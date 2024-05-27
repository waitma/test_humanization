"""Build a CSV file of paired heavy & light chain sequences."""

import os
import pandas as pd
# import dask.dataframe as dd
import random
import numpy
import re
from tqdm import tqdm
import json
import pickle
import lmdb
# from concurrent.futures import ProcessPoolExecutor
import logging
import torch
import joblib

# from abnumber import Chain
# from anarci import anarci, number

HEAVY_POSITIONS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
       '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
       '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
       '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
       '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
       '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
       '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
       '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
       '101', '102', '103', '104', '105', '106', '107', '108', '109',
       '110', '111', '111A', '111B', '111C', '111D', '111E', '111F',
       '111G', '111H', '111I', '111J', '111K', '111L', '112L', '112K',
       '112J', '112I', '112H', '112G', '112F', '112E', '112D', '112C',
       '112B', '112A', '112', '113', '114', '115', '116', '117', '118',
       '119', '120', '121', '122', '123', '124', '125', '126', '127',
       '128']

HEAVY_POSITIONS_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, '11': 10,
                        '12': 11, '13': 12, '14': 13, '15': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19,
                        '21': 20, '22': 21, '23': 22, '24': 23, '25': 24, '26': 25, '27': 26, '28': 27, '29': 28,
                        '30': 29, '31': 30, '32': 31, '33': 32, '34': 33, '35': 34, '36': 35, '37': 36, '38': 37,
                        '39': 38, '40': 39, '41': 40, '42': 41, '43': 42, '44': 43, '45': 44, '46': 45, '47': 46,
                        '48': 47, '49': 48, '50': 49, '51': 50, '52': 51, '53': 52, '54': 53, '55': 54, '56': 55,
                        '57': 56, '58': 57, '59': 58, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64,
                        '66': 65, '67': 66, '68': 67, '69': 68, '70': 69, '71': 70, '72': 71, '73': 72, '74': 73,
                        '75': 74, '76': 75, '77': 76, '78': 77, '79': 78, '80': 79, '81': 80, '82': 81, '83': 82,
                        '84': 83, '85': 84, '86': 85, '87': 86, '88': 87, '89': 88, '90': 89, '91': 90, '92': 91,
                        '93': 92, '94': 93, '95': 94, '96': 95, '97': 96, '98': 97, '99': 98, '100': 99, '101': 100,
                        '102': 101, '103': 102, '104': 103, '105': 104, '106': 105, '107': 106, '108': 107, '109': 108,
                        '110': 109, '111': 110, '111A': 111, '111B': 112, '111C': 113, '111D': 114, '111E': 115,
                        '111F': 116, '111G': 117, '111H': 118, '111I': 119, '111J': 120, '111K': 121, '111L': 122,
                        '112L': 123, '112K': 124, '112J': 125, '112I': 126, '112H': 127, '112G': 128, '112F': 129,
                        '112E': 130, '112D': 131, '112C': 132, '112B': 133, '112A': 134, '112': 135, '113': 136,
                        '114': 137, '115': 138, '116': 139, '117': 140, '118': 141, '119': 142, '120': 143,
                        '121': 144, '122': 145, '123': 146, '124': 147, '125': 148, '126': 149, '127': 150, '128': 151}

HEAVY_CDR_INDEX = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

LIGHT_POSITIONS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
       '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
       '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
       '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
       '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
       '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
       '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
       '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
       '101', '102', '103', '104', '105', '106', '107', '108', '109',
       '110', '111', '111A', '111B', '111C', '111D', '111E', '111F',
       '112F', '112E', '112D', '112C', '112B', '112A', '112', '113',
       '114', '115', '116', '117', '118', '119', '120', '121', '122',
       '123', '124', '125', '126', '127']

LIGHT_POSITIONS_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, '11': 10,
                        '12': 11, '13': 12, '14': 13, '15': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19,
                        '21': 20, '22': 21, '23': 22, '24': 23, '25': 24, '26': 25, '27': 26, '28': 27, '29': 28,
                        '30': 29, '31': 30, '32': 31, '33': 32, '34': 33, '35': 34, '36': 35, '37': 36, '38': 37,
                        '39': 38, '40': 39, '41': 40, '42': 41, '43': 42, '44': 43, '45': 44, '46': 45, '47': 46,
                        '48': 47, '49': 48, '50': 49, '51': 50, '52': 51, '53': 52, '54': 53, '55': 54, '56': 55,
                        '57': 56, '58': 57, '59': 58, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64,
                        '66': 65, '67': 66, '68': 67, '69': 68, '70': 69, '71': 70, '72': 71, '73': 72, '74': 73,
                        '75': 74, '76': 75, '77': 76, '78': 77, '79': 78, '80': 79, '81': 80, '82': 81, '83': 82,
                        '84': 83, '85': 84, '86': 85, '87': 86, '88': 87, '89': 88, '90': 89, '91': 90, '92': 91,
                        '93': 92, '94': 93, '95': 94, '96': 95, '97': 96, '98': 97, '99': 98, '100': 99, '101': 100,
                        '102': 101, '103': 102, '104': 103, '105': 104, '106': 105, '107': 106, '108': 107, '109': 108,
                        '110': 109, '111': 110, '111A': 111, '111B': 112, '111C': 113, '111D': 114, '111E': 115,
                        '111F': 116, '112F': 117, '112E': 118, '112D': 119, '112C': 120, '112B': 121, '112A': 122,
                        '112': 123, '113': 124, '114': 125, '115': 126, '116': 127, '117': 128, '118': 129, '119': 130,
                        '120': 131, '121': 132, '122': 133, '123': 134, '124': 135, '125': 136, '126': 137, '127': 138}

LIGHT_CDR_INDEX = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3, 3, 3, 3,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


Chn_seqs = set()
Special_count = list()

SEG_names_dict = {
    'H': ['fwh1', 'cdrh1', 'fwh2', 'cdrh2', 'fwh3', 'cdrh3', 'fwh4'],
    'K': ['fwk1', 'cdrk1', 'fwk2', 'cdrk2', 'fwk3', 'cdrk3', 'fwk4'],
    'L': ['fwl1', 'cdrl1', 'fwl2', 'cdrl2', 'fwl3', 'cdrl3', 'fwl4'],
}

def extract_before_dash(s):
    pattern = r"(.+?)-"
    match = re.search(pattern, s)
    if match:
        return match.group(1)
    else:
        return None

def parse_unpair_cgz_file(data_frame, path, length=149):
    """Parse the GZ-compressed CSV file."""

    # obtain a list of (CSV file name, record index, heavy chain seq., light chain seq.)-tuples
    seq_list = []
    name = os.path.basename(path).replace('.csv', '')
    # try:
    for row_data in tqdm(data_frame.itertuples(), total=len(data_frame)):
        chain_type = row_data.locus
        if chain_type == 'H':
            pos_dict = HEAVY_POSITIONS_dict
            cdr_index = HEAVY_CDR_INDEX
        else:
            pos_dict = LIGHT_POSITIONS_dict
            cdr_index = LIGHT_CDR_INDEX
        try:
            seg_names = SEG_names_dict[row_data.locus]
            chn_seq = row_data.sequence_alignment_aa
            anarci_outputs = json.loads(row_data.ANARCI_numbering.replace('\'', '"'))
            seg_seqs_c = [''.join(anarci_outputs[x].values()) for x in seg_names]
            assert ''.join(seg_seqs_c) in chn_seq
            if chn_seq not in Chn_seqs:
                # print(chn_seq)
                Chn_seqs.add(chn_seq)
                if '1' in [i.strip() for i in anarci_outputs[seg_names[0]].keys()]:
                    pad_initial_seg_seq = ['-'] * len(cdr_index)
                    merged_dict = {key: value for sub_dict in anarci_outputs.values()
                                   for key, value in sub_dict.items()}
                    for key, value in merged_dict.items():
                        key = key.strip()
                        pos_idx = pos_dict[key]
                        pad_initial_seg_seq[pos_idx] = value
                    pad_seq = ''.join(pad_initial_seg_seq)
                    assert len(pad_seq) == len(cdr_index), 'Pad has problem.'
                # Append seq for training.
                    V_gene = extract_before_dash(row_data.v_call)
                    # print(V_gene)
                    seq_list.append((name, chn_seq, pad_seq, chain_type, V_gene))

            else:
                raise AttributeError(f'Chain {chain_type} Length is special.')

        except (AttributeError, KeyError):
            # print(f'May not suit for imgt.')
            # print(f'Chain {chain_type} Length is special.')
            continue

    return seq_list



def main(chain_type='Heavy', num_processes=4):
    """Main entry."""

    # configurations
    # root_dir = '/data/home/waitma/antibody_proj/antidiff/data/oas_unpair_human_data/'
    root_dir = '/apdcephfs/share_1364275/waitma/anti_proj/data/'
    cgz_dpath = os.path.join(root_dir, 'unpair_select_cgzdata')

    save_heavy_fpath = os.path.join(root_dir, 'heavy_unpair_limit.pkl')
    save_light_fpath = os.path.join(root_dir, 'light_unpair_limit.pkl')
    limit_number = 5000000


    # Filter set
    # filter_fpath = '/apdcephfs/share_1364275/waitma/anti_proj/data/oas_pair_data/processed/pair_sequence.pkl'
    # with open(filter_fpath, 'rb') as f:
    #     filter_list = pickle.load(f)
    #     f.close()
    # print('Loading pair data')
    # for seq in tqdm(filter_list):
    #     if seq not in Chn_seqs:
    #         Chn_seqs.add(seq)

    # parse all the GZ-compressed CSV files
    heavy_seq_list = []
    light_seq_list = []
    chunksize = 40000
    cgz_names = os.listdir(cgz_dpath)
    cgz_fpath_list = []
    for cgz_fname in tqdm(cgz_names, desc=f'parsing CSV files'):
        cgz_fpath = os.path.join(cgz_dpath, cgz_fname)
        cgz_fpath_list.append(cgz_fpath)

    for cgz_fpath in cgz_fpath_list:
        if '_Heavy_' in cgz_fpath:
            if len(heavy_seq_list) >= limit_number:
                continue
        if '_Light_' in cgz_fpath:
            if len(light_seq_list) >= limit_number:
                continue

        for data_frame in pd.read_csv(cgz_fpath,
                                              header=1,
                                              usecols=['locus', 'sequence_alignment_aa', 'ANARCI_numbering', 'v_call'],
                                              chunksize=chunksize,
                                              engine='python',
                                              # compression='gzip'
                                              # on_bad_lines='warn'
                                              ):
            seq_list = parse_unpair_cgz_file(data_frame, cgz_fpath)
            if '_Heavy_' in cgz_fpath:
                if len(heavy_seq_list) < limit_number:
                    heavy_seq_list.extend(seq_list)
            elif '_Light_' in cgz_fpath:
                if len(light_seq_list) < limit_number:
                    light_seq_list.extend(seq_list)
            else:
                print('Unkown seq type')
                continue
        if len(heavy_seq_list) >= limit_number and len(light_seq_list) >= limit_number:
            print(len(heavy_seq_list))
            print(len(light_seq_list))
            break

    print(len(heavy_seq_list))
    print(len(light_seq_list))
    heavy_seq_list = heavy_seq_list[:limit_number]
    light_seq_list = light_seq_list[:limit_number]

    print('All chnseq: ', len(Chn_seqs))
    with open(save_heavy_fpath, 'wb') as f:
        joblib.dump(heavy_seq_list, f)
        f.close()

    with open(save_light_fpath, 'wb') as f:
        joblib.dump(light_seq_list, f)
        f.close()


#############################

if __name__ == '__main__':
    # cgz_path = '/data/home/waitma/antibody_proj/encoder/data/oas_pair_human_data/cgz_data/1287167_1_Paired_All.csv.gz'
    # parse_cgz_file(cgz_path)
    # main()
    # devide_train_and_valid(csv_path)
    # filter_length(csv_path, out_path)

    main()

    root_dir = '/apdcephfs/share_1364275/waitma/anti_proj/data/light_unpair_limit.pkl'

    # with open(root_dir, 'rb') as f:
    #     file_list = pickle.load(f)
    #
    # for line in file_list:
    #     print(line)
