"""Build a CSV file of paired heavy & light chain sequences."""

import os
import json
import logging

import pandas as pd
from tqdm import tqdm

from tfold_utils.common_utils import tfold_init


def parse_cgz_file(path):
    """Parse the GZ-compressed CSV file."""

    # configurations
    seg_names_dict = {
        'H': ['fwh1', 'cdrh1', 'fwh2', 'cdrh2', 'fwh3', 'cdrh3', 'fwh4'],
        'K': ['fwk1', 'cdrk1', 'fwk2', 'cdrk2', 'fwk3', 'cdrk3', 'fwk4'],
        'L': ['fwl1', 'cdrl1', 'fwl2', 'cdrl2', 'fwl3', 'cdrl3', 'fwl4'],
    }

    # parse the GZ-compressed CSV file
    try:
        data_frame = pd.read_csv(path, header=1, compression='gzip')
    except EOFError:
        logging.warning('corrupted GZ-compressed CSV file: %s', path)
        return []

    # obtain a list of (CSV file name, record index, heavy chain seq., light chain seq.)-tuples
    seq_list = []
    chn_seqs = set()  # to remove duplicated sequences
    name = os.path.basename(path).replace('.csv.gz', '')
    for idx, row_data in data_frame.iterrows():
        # heavy chain
        seg_names = seg_names_dict[row_data['locus_heavy']]
        chn_seq = row_data['sequence_alignment_aa_heavy']
        anarci_outputs = json.loads(row_data['ANARCI_numbering_heavy'].replace('\'', '"'))
        seg_seqs_hc = [''.join(anarci_outputs[x].values()) for x in seg_names]
        assert ''.join(seg_seqs_hc) in chn_seq
        chn_seq_hc = ''.join(seg_seqs_hc)  # remove redundanat leading & trailing AAs

        # light chain
        seg_names = seg_names_dict[row_data['locus_light']]
        chn_seq = row_data['sequence_alignment_aa_light']
        anarci_outputs = json.loads(row_data['ANARCI_numbering_light'].replace('\'', '"'))
        seg_seqs_lc = [''.join(anarci_outputs[x].values()) for x in seg_names]
        assert ''.join(seg_seqs_lc) in chn_seq
        chn_seq_lc = ''.join(seg_seqs_lc)  # remove redundanat leading & trailing AAs

        # record the current data entry
        if (chn_seq_hc, chn_seq_lc) not in chn_seqs:
            chn_seqs.add((chn_seq_hc, chn_seq_lc))
            seq_list.append((name, idx, seg_seqs_hc, seg_seqs_lc))

    return seq_list


def export_csv_file(seq_list, path, keep_fr_cdr_def=False):
    """Export the sequence data into a CSV file."""

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w', encoding='UTF-8') as o_file:
        o_file.write('ENTRY,SEQ\n')
        for name, idx, seg_seqs_hc, seg_seqs_lc in seq_list:
            prot_id_hc = f'{name}_{idx}_H'
            prot_id_lc = f'{name}_{idx}_L'
            if not keep_fr_cdr_def:
                chn_seq_hc = ''.join(seg_seqs_hc)
                chn_seq_lc = ''.join(seg_seqs_lc)
            else:
                chn_seq_hc = ''.join(
                    [seq if idx % 2 == 0 else seq.lower() for idx, seq in enumerate(seg_seqs_hc)])
                chn_seq_lc = ''.join(
                    [seq if idx % 2 == 0 else seq.lower() for idx, seq in enumerate(seg_seqs_lc)])
            o_file.write(f'{prot_id_hc}|{prot_id_lc},{chn_seq_hc}|{chn_seq_lc}\n')


def main():
    """Main entry."""

    # configurations
    root_dir = '/mnt/exDisk/Datasets/OAS-20230320/paired'
    cgz_dpath = os.path.join(root_dir, 'raw.data')
    csv_fpath_pri = os.path.join(root_dir, 'oas_paired.csv')  # w/o FR-CDR definitions
    csv_fpath_sec = os.path.join(root_dir, 'oas_paired_fr_cdr.csv')  # w/ FR-CDR definitions

    # initialization
    tfold_init()

    # parse all the GZ-compressed CSV files
    seq_list = []
    cgz_names = sorted(os.listdir(cgz_dpath))
    for cgz_fname in tqdm(cgz_names, desc='parsing GZ-compressed CSV files'):
        cgz_fpath = os.path.join(cgz_dpath, cgz_fname)
        seq_list.extend(parse_cgz_file(cgz_fpath))
    export_csv_file(seq_list, csv_fpath_pri, keep_fr_cdr_def=False)
    export_csv_file(seq_list, csv_fpath_sec, keep_fr_cdr_def=True)


if __name__ == '__main__':
    main()