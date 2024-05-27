"""Build CSV files from OAS raw data w/ sequence-level data subset splitting (slower than v2)."""
"""From tfold_utils"""

import os
import re
import json
import random
import logging
from collections import defaultdict
from multiprocessing import Pool, Manager

import pandas as pd


def get_cgz_fpaths(csv_fpath, cgz_dpath):
    """Get GZ-compressed CSV files, grouped by species names."""

    df_meta = pd.read_csv(csv_fpath)
    cgz_fpaths_dict = defaultdict(list)
    regex = re.compile(r'(camel|human|mouse|rabbit|rat|rhesus)')
    for _, row_data in df_meta.iterrows():
        spc_name = re.search(regex, row_data['Species'].lower()).group()
        chn_type = row_data['Chain'].lower()  # heavy / light
        cgz_fpath = os.path.join(cgz_dpath, f'{row_data["Entry"]}.csv.gz')
        cgz_fpaths_dict[(spc_name, chn_type)].append(cgz_fpath)

    return cgz_fpaths_dict


def build_csv_file(cgz_fpath, csv_fpath):
    """Build a CSV file of per-segment amino-acid sequences."""

    # configurations
    n_seqs_max = 100000  # maximal number of sequences in a single CSV file
    seg_names_pri = ['fw1', 'cdr1', 'fw2', 'cdr2', 'fw3', 'cdr3', 'fw4']
    seg_names_sec_dict = {
        'H': ['fwh1', 'cdrh1', 'fwh2', 'cdrh2', 'fwh3', 'cdrh3', 'fwh4'],
        'K': ['fwk1', 'cdrk1', 'fwk2', 'cdrk2', 'fwk3', 'cdrk3', 'fwk4'],
        'L': ['fwl1', 'cdrl1', 'fwl2', 'cdrl2', 'fwl3', 'cdrl3', 'fwl4'],
    }

    # initialization
    if os.path.exists(csv_fpath):
        return
    logging.info('parsing the GZ-compressed CSV file: %s', cgz_fpath)

    # parse the GZ-compressed CSV file
    try:
        data_frame = pd.read_csv(cgz_fpath, header=1, nrows=n_seqs_max, compression='gzip')
    except EOFError:
        logging.warning('corrupted GZ-compressed CSV file: %s', cgz_fpath)
        return

    # obtain a list of (per-chain sequence, per-segment sequences)-tuples
    seq_list = []
    chn_seqs = set()  # set of per-chain sequences (for duplicate removal)
    for _, row_data in data_frame.iterrows():
        seg_names_sec = seg_names_sec_dict[row_data['locus']]
        chn_seq = row_data['sequence_alignment_aa']
        anarci_outputs = json.loads(row_data['ANARCI_numbering'].replace('\'', '"'))
        seg_seqs = []
        for seg_name in seg_names_sec:
            seg_seq = ''.join(anarci_outputs[seg_name].values())
            seg_seqs.append(seg_seq)
        if any(len(x) == 0 for x in seg_seqs):  # discard sequences w/ whole region(s) missing
            continue
        assert ''.join(seg_seqs) in chn_seq, f'{chn_seq} vs. {seg_seqs} ({cgz_fpath})'
        chn_seq = ''.join(seg_seqs)
        if chn_seq not in chn_seqs:
            chn_seqs.add(chn_seq)
            seq_list.append((chn_seq, seg_seqs))

    # build a CSV file of per-segment amino-acid sequences
    os.makedirs(os.path.dirname(csv_fpath), exist_ok=True)
    with open(csv_fpath, 'w', encoding='UTF-8') as o_file:
        o_file.write(','.join(['chn'] + seg_names_pri) + '\n')
        for chn_seq, seg_seqs in seq_list:
            o_file.write(','.join([chn_seq] + seg_seqs) + '\n')


def get_file_size(path, file_size_dict):
    """Get the number of lines (excluding the header line) in the CSV file."""

    with open(path, 'r', encoding='UTF-8') as i_file:
        file_size_dict[path] = len(i_file.readlines())


def get_assignment(idx_file, idx_seq, config):
    """Get the subset & CSV file assignment."""

    rand_val_int = idx_file * 101 + idx_seq * 97
    rand_val_flt = rand_val_int % 65536 / 65535.0

    if rand_val_flt < config['ratio_trn']:
        subset = 'train'
        idx_file = rand_val_int % config['n_files_trn']
    elif rand_val_flt < config['ratio_trn'] + config['ratio_val']:
        subset = 'valid'
        idx_file = rand_val_int % config['n_files_val']
    else:
        subset = 'test'
        idx_file = rand_val_int % config['n_files_tst']

    return subset, idx_file


def build_csv_file_impl(csv_fpaths_src, csv_fpath_dst, subset_dst, idx_file_dst, config):
    """Build a CSV file from multiple source CSV files - core implementation."""

    # configurations
    seg_names = ['fw1', 'cdr1', 'fw2', 'cdr2', 'fw3', 'cdr3', 'fw4']

    # build a CSV file
    logging.info('building a CSV file: %s', csv_fpath_dst)
    with open(csv_fpath_dst, 'w', encoding='UTF-8') as o_file:
        o_file.write(','.join(['chn'] + seg_names) + '\n')
        for idx_file_src, csv_fpath_src in enumerate(csv_fpaths_src):
            with open(csv_fpath_src, 'r', encoding='UTF-8') as i_file:
                i_file.readline()  # skip the first line
                for idx_seq, i_line in enumerate(i_file):
                    subset, idx_file = get_assignment(idx_file_src, idx_seq, config)
                    if (subset == subset_dst) and (idx_file == idx_file_dst):
                        o_file.write(i_line)


def build_subset(csv_fpaths_src, spc_name, chn_type, file_size_dict, csv_dpath_root):
    """Build CSV files for the specified (species, chain type) combination."""

    # configurations
    config = {
        'ratio_trn': 0.8,
        'ratio_val': 0.1,
        'ratio_tst': 0.1,
    }
    n_threads = 40
    n_seqs_per_file = 1000000

    # determine the number of CSV files in each subset
    n_seqs = sum(file_size_dict[x] for x in csv_fpaths_src)
    n_seqs_val = int(n_seqs * config['ratio_val'] + 0.5)
    n_seqs_tst = int(n_seqs * config['ratio_tst'] + 0.5)
    n_seqs_trn = n_seqs - n_seqs_val - n_seqs_tst
    config['n_files_trn'] = (n_seqs_trn + n_seqs_per_file - 1) // n_seqs_per_file
    config['n_files_val'] = (n_seqs_val + n_seqs_per_file - 1) // n_seqs_per_file
    config['n_files_tst'] = (n_seqs_tst + n_seqs_per_file - 1) // n_seqs_per_file

    # build CSV files for the specified (species, chain type) combination
    args_list = []
    for idx_file in range(config['n_files_trn']):
        subset = 'train'
        csv_fname_dst = f'{chn_type}-{idx_file:04d}-of-{config["n_files_trn"]:04d}.csv'
        csv_fpath_dst = os.path.join(csv_dpath_root, subset, spc_name, csv_fname_dst)
        args_list.append((csv_fpaths_src, csv_fpath_dst, subset, idx_file, config))
    for idx_file in range(config['n_files_val']):
        subset = 'valid'
        csv_fname_dst = f'{chn_type}-{idx_file:04d}-of-{config["n_files_val"]:04d}.csv'
        csv_fpath_dst = os.path.join(csv_dpath_root, subset, spc_name, csv_fname_dst)
        args_list.append((csv_fpaths_src, csv_fpath_dst, subset, idx_file, config))
    for idx_file in range(config['n_files_tst']):
        subset = 'test'
        csv_fname_dst = f'{chn_type}-{idx_file:04d}-of-{config["n_files_tst"]:04d}.csv'
        csv_fpath_dst = os.path.join(csv_dpath_root, subset, spc_name, csv_fname_dst)
        args_list.append((csv_fpaths_src, csv_fpath_dst, subset, idx_file, config))
    with Pool(processes=n_threads) as pool:
        pool.starmap(build_csv_file_impl, args_list)


def main():
    """Main entry."""

    # configurations
    n_threads = 50
    n_seqs_min = 1000  # minimal number of sequences in a (species, chain type) combination
    data_dir = '/data/home/waitma/antibody_proj/encoder/data/oas_pair_human_data'
    csv_fpath = os.path.join(data_dir, 'meta_data.csv')
    cgz_dpath = os.path.join(data_dir, 'raw.data')
    csv_dpath_root = os.path.join(data_dir, 'csv.files.v3')

    # initialization
    # tfold_init()
    random.seed(42)  # so that all the workers share the same train/valid/test split

    # configure the logging format
    logging.basicConfig(
        format='[%(asctime)-15s %(levelname)s %(filename)s:L%(lineno)d] %(message)s', level='INFO')

    # find all the GZ-compressed CSV files
    cgz_fpaths_dict = get_cgz_fpaths(csv_fpath, cgz_dpath)

    # convert GZ-compressed CSV files into plain CSV files
    logging.info('converting GZ-compressed CSV files into plain CSV files ...')
    args_list = []
    csv_fpaths = []
    for (spc_name, chn_type), cgz_fpaths in cgz_fpaths_dict.items():
        csv_dpath = os.path.join(csv_dpath_root, 'unmixed', f'{spc_name}-{chn_type}')
        for cgz_fpath in cgz_fpaths:
            csv_fpath = os.path.join(csv_dpath, os.path.basename(cgz_fpath).replace('.gz', ''))
            csv_fpaths.append(csv_fpath)
            args_list.append((cgz_fpath, csv_fpath))
    csv_fpaths.sort()
    with Pool(processes=n_threads) as pool:
        pool.starmap(build_csv_file, args_list)

    # determine the number of lines in each CSV file
    logging.info('determining the number of lines in each CSV file ...')
    txt_fpath = os.path.join(csv_dpath_root, 'unmixed/file_size.txt')
    if not os.path.exists(txt_fpath):
        file_size_dict = Manager().dict()
        args_list = [(x, file_size_dict) for x in csv_fpaths]
        with Pool(processes=n_threads) as pool:
            pool.starmap(get_file_size, args_list)
        with open(txt_fpath, 'w', encoding='UTF-8') as o_file:
            for csv_fpath in csv_fpaths:
                o_file.write(f'{csv_fpath} {file_size_dict[csv_fpath]}\n')
    else:
        file_size_dict = {}
        with open(txt_fpath, 'r', encoding='UTF-8') as i_file:
            for i_line in i_file:
                csv_fpath, n_seqs = i_line.split()
                file_size_dict[csv_fpath] = int(n_seqs)

    # group CSV files based on the (species, chain type) combination
    csv_fpaths_per_subset_dict = defaultdict(list)
    for csv_fpath in csv_fpaths:
        spc_name, chn_type = os.path.basename(os.path.dirname(csv_fpath)).split('-')
        csv_fpaths_per_subset_dict[(spc_name, chn_type)].append(csv_fpath)

    # build CSV files for each (species, chain type) combination
    for (spc_name, chn_type), csv_fpaths_sel in csv_fpaths_per_subset_dict.items():
        n_seqs = sum(file_size_dict[x] for x in csv_fpaths_sel)
        if n_seqs < n_seqs_min:
            continue
        logging.info('building CSV files for (%s, %s) ...', spc_name, chn_type)
        build_subset(csv_fpaths_sel, spc_name, chn_type, file_size_dict, csv_dpath_root)


if __name__ == '__main__':
    main()
