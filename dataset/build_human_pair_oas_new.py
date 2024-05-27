"""Build CSV files from OAS dataset, Split the """
import pickle
import os
import json
import logging

import pandas as pd
import torch
from tqdm import tqdm



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


class LengthError(Exception):
    def __init__(self, message=None):
        if message is not None:
            self.message = message
        else:
            self.message = 'Pre difined Length has wrong.'


def region_padding_fix(ori_sequence, fix_lgth):
    ori_lgth = len(ori_sequence)
    if ori_lgth <= fix_lgth:
        padding_lgth = fix_lgth - ori_lgth
        pad_sequence = ori_sequence + '-' * padding_lgth
        return pad_sequence
    elif ori_lgth > fix_lgth:
        print("Ori length: ", ori_lgth)
        print("Fix length: ", fix_lgth)
        raise LengthError
    else:
        pad_sequence = ori_sequence
        return pad_sequence


def parse_cgz_file(path, Chn_set, length=128):
    """Parse the GZ-compressed CSV file."""

    # parse the GZ-compressed CSV file
    try:
        data_frame = pd.read_csv(path, header=1, compression='gzip')
    except EOFError:
        logging.warning('corrupted GZ-compressed CSV file: %s', path)
        return []

    # obtain a list of (CSV file name, record index, heavy chain seq., light chain seq.)-tuples
    seq_list = []
    # chn_seqs = set()  # to remove duplicated sequences
    name = os.path.basename(path).replace('.csv.gz', '')
    for row_data in tqdm(data_frame.itertuples(), total=len(data_frame)):
        # heavy chain
        if row_data.locus_heavy == 'L' or row_data.locus_heavy == 'K':
            continue
        else:
            try:
                pos_dict = HEAVY_POSITIONS_dict
                cdr_index = HEAVY_CDR_INDEX
                seg_names = SEG_names_dict[row_data.locus_heavy]
                chn_seq = row_data.sequence_alignment_aa_heavy
                anarci_outputs = json.loads(row_data.ANARCI_numbering_heavy.replace('\'', '"'))
                seg_seqs_hc = [''.join(anarci_outputs[x].values()) for x in seg_names]
                assert ''.join(seg_seqs_hc) in chn_seq
                chn_seq_hc = ''.join(seg_seqs_hc)
                # if '1' in [i.strip() for i in anarci_outputs[seg_names[0]].keys()]:
                if 'X' not in chn_seq_hc:
                    h_pad_initial_seg_seq = ['-'] * len(cdr_index)
                    merged_dict = {key: value for sub_dict in anarci_outputs.values()
                                   for key, value in sub_dict.items()}
                    for key, value in merged_dict.items():
                        key = key.strip()
                        pos_idx = pos_dict[key]
                        h_pad_initial_seg_seq[pos_idx] = value
                    pad_seq_hc = ''.join(h_pad_initial_seg_seq)
                    assert len(pad_seq_hc) == len(cdr_index), 'Pad has problem.'
                else:
                    raise AttributeError()

            except:
                # print('H Length is special.')
                continue

        # light chain
        if row_data.locus_light == 'H':
            continue
        else:
            try:
                pos_dict = LIGHT_POSITIONS_dict
                cdr_index = LIGHT_CDR_INDEX
                seg_names = SEG_names_dict[row_data.locus_light]
                chn_seq = row_data.sequence_alignment_aa_light
                anarci_outputs = json.loads(row_data.ANARCI_numbering_light.replace('\'', '"'))
                seg_seqs_lc = [''.join(anarci_outputs[x].values()) for x in seg_names]
                assert ''.join(seg_seqs_lc) in chn_seq
                chn_seq_lc = ''.join(seg_seqs_lc)  # remove redundanat leading & trailing AAs
                # if '1' in [i.strip() for i in anarci_outputs[seg_names[0]].keys()]:
                if 'X' not in chn_seq_lc:
                    l_pad_initial_seg_seq = ['-'] * len(cdr_index)
                    merged_dict = {key: value for sub_dict in anarci_outputs.values()
                                   for key, value in sub_dict.items()}
                    for key, value in merged_dict.items():
                        key = key.strip()
                        pos_idx = pos_dict[key]
                        l_pad_initial_seg_seq[pos_idx] = value
                    pad_seq_lc = ''.join(l_pad_initial_seg_seq)
                    assert len(pad_seq_lc) == len(cdr_index), 'Pad has problem.'
                else:
                    raise AttributeError
            except:
                # print('L length is special.')
                continue

        # record the current data entry
        if (chn_seq_hc, chn_seq_lc) not in Chn_set:  # and len(chn_seq_hc) <= length:
            Chn_set.add((chn_seq_hc, chn_seq_lc))
            type_hc, type_lc = row_data.locus_heavy, row_data.locus_light
            seq_list.append((name, chn_seq_hc, chn_seq_lc,
                             pad_seq_hc, pad_seq_lc,
                             type_hc, type_lc))

    return seq_list, Chn_set


def zero_batch(seq):
    """
    Get batch zero
    :param seq: str
    :return: len(str) * 0
    """
    str_num = len(seq)
    return str_num * [0]


def cdr_batch(idx, seq):
    '''
    :param idx: int
    :param seq: str
    :return: [int] * len(str)
    '''
    if idx == 1:
        return len(seq) * [idx]
    elif idx == 3:
        return len(seq) * [idx-1]
    elif idx == 5:
        return len(seq) * [idx-2]


def export_csv_file(seq_list, path, keep_fr_cdr_def=False):
    """Export the sequence data into a CSV file."""

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w', encoding='UTF-8') as o_file:
        o_file.write('ENTRY,HSEQ,LSEQ,HCDR,LCDR\n')
        for name, idx, seg_seqs_hc, seg_seqs_lc in seq_list:
            prot_id_hc = f'{name}_{idx}_H'
            prot_id_lc = f'{name}_{idx}_L'
            # seq_hc = ''.join(
            #     [seq if idx % 2 == 0 else seq.lower() for idx, seq in enumerate(seg_seqs_hc)])
            # seq_lc = ''.join(
            #     [seq if idx % 2 == 0 else seq.lower() for idx, seq in enumerate(seg_seqs_lc)])
            seq_hc = ''.join(seg_seqs_hc)
            seq_lc = ''.join(seg_seqs_lc)
            hc_cdr = sum([zero_batch(seq)
                      if idx % 2 == 0 else cdr_batch(idx, seq) for idx, seq in enumerate(seg_seqs_hc)], [])
            lc_cdr = sum([zero_batch(seq)
                      if idx % 2 == 0 else cdr_batch(idx, seq) for idx, seq in enumerate(seg_seqs_lc)], [])
            o_file.write(f'{prot_id_hc}|{prot_id_lc},{seq_hc},{seq_lc},{hc_cdr},{lc_cdr}\n')


def devide_train_and_valid(csv_path, valid_sample=50000):
    """
    Partition the csv file.
    :param csv_path:
    :return: None, export two csv path.
    """
    df = pd.read_csv(csv_path)
    valid_df = df.sample(n=valid_sample)
    train_df = df.drop(valid_df.index)

    valid_df.to_csv(os.path.join(os.path.dirname(csv_path), 'pair_valid_set.csv'), index=False)
    train_df.to_csv(os.path.join(os.path.dirname(csv_path), 'pair_train_set.csv'), index=False)


def filter_length(csv_path, out_path, length=128):
    """ Filter the sequence length. """
    data = pd.read_csv(csv_path, sep='\n')
    column_name = 'HSEQ'
    long_string_indices = data[data[column_name].str.len() > 128].index
    data.drop(long_string_indices, inplace=True)
    data.to_csv('cleaned_file.csv', index=False)


def main():
    """Main entry."""

    # configurations
    root_dir = '/data/home/waitma/antibody_proj/antidiff/data/oas_pair_human_data/'
    cgz_dpath = os.path.join(root_dir, 'cgz_data')
    csv_fpath_pri = os.path.join(root_dir, 'oas_paired.csv')  # w/o FR-CDR definitions
    csv_fpath_sec = os.path.join(root_dir, 'oas_paired_fr_cdr_v0.csv')  # w/ FR-CDR definitions

    # initialization
    # tfold_init()

    # parse all the GZ-compressed CSV files
    seq_list = []
    cgz_names = sorted(os.listdir(cgz_dpath))
    for cgz_fname in tqdm(cgz_names, desc='parsing GZ-compressed CSV files'):
        cgz_fpath = os.path.join(cgz_dpath, cgz_fname)
        if not os.path.isdir(cgz_fpath):
            seq_list.extend(parse_cgz_file(cgz_fpath, Chn_seqs))
    # export_csv_file(seq_list, csv_fpath_pri, keep_fr_cdr_def=False)
    export_csv_file(seq_list, csv_fpath_sec, keep_fr_cdr_def=True)

#############################
H_dict_distribution = {
    'fwh1': [],
    'cdrh1': [],
    'fwh2': [],
    'cdrh2': [],
    'fwh3': [],
    'cdrh3': [],
    'fwh4': []
}

K_dict_distribution = {
    'fwk1': [],
    'cdrk1': [],
    'fwk2': [],
    'cdrk2': [],
    'fwk3': [],
    'cdrk3': [],
    'fwk4': []
}

L_dict_distribution = {
    'fwl1': [],
    'cdrl1': [],
    'fwl2': [],
    'cdrl2': [],
    'fwl3': [],
    'cdrl3': [],
    'fwl4': []
}

def statistic_different_region_length():
    """ Need to know the length for model design. """
    root_dir = '/data/home/waitma/antibody_proj/antidiff/data/oas_pair_human_data'
    cgz_dpath = os.path.join(root_dir, 'cgz_data')
    save_dpath = os.path.join(cgz_dpath, 'statistic_distribution')

    seg_names_dict = {
        'H': ['fwh1', 'cdrh1', 'fwh2', 'cdrh2', 'fwh3', 'cdrh3', 'fwh4'],
        'K': ['fwk1', 'cdrk1', 'fwk2', 'cdrk2', 'fwk3', 'cdrk3', 'fwk4'],
        'L': ['fwl1', 'cdrl1', 'fwl2', 'cdrl2', 'fwl3', 'cdrl3', 'fwl4'],
    }

    seq_list = []
    cgz_names = sorted(os.listdir(cgz_dpath))
    for cgz_fname in tqdm(cgz_names, desc='parsing GZ-compressed CSV files'):
        cgz_fpath = os.path.join(cgz_dpath, cgz_fname)
        if not os.path.isdir(cgz_fpath):
            seq_list.extend(parse_cgz_file(cgz_fpath))

    for a_seq in tqdm(seq_list):
        H_type = a_seq[3]
        L_type = a_seq[4]
        for region_i, (H_seg_seq, L_seg_seq) in enumerate(zip(a_seq[1], a_seq[2])):
            H_dict_distribution[seg_names_dict['H'][region_i]].append(len(H_seg_seq))
            if L_type == 'L':
                L_dict_distribution[seg_names_dict['L'][region_i]].append(len(L_seg_seq))
            elif L_type == 'K':
                K_dict_distribution[seg_names_dict['K'][region_i]].append(len(L_seg_seq))
            elif L_type == 'H':
                print(L_type)
                print('Unknown Type')

    save_fpath = os.path.join(save_dpath, 'length_distribution_no_length_limit.pkl')

    save_dict = {
        'H_dist': H_dict_distribution,
        'K_dist': K_dict_distribution,
        'L_dist': L_dict_distribution
    }

    with open(save_fpath, 'wb') as f:
        pickle.dump(save_dict, f)


def draw_the_different_length_distribution():
    """ Draw the different region length's distribution of different chain. """
    data_draw_path = '/data/home/waitma/antibody_proj/antidiff/data/oas_pair_human_data/' \
                     'cgz_data/statistic_distribution/length_distribution_no_length_limit.pkl'

    with open(data_draw_path, 'rb') as f:
        data_draw_dict = pickle.load(f)

    save_name = [
        'H_distribution_length_nolimit',
        'L_distribution_length_nolimit',
        'K_distribution_length_nolimit'
    ]

    for chain_idx, chain_dict in enumerate(data_draw_dict.values()):
        fig, axs = plt.subplots(7, 1, figsize=(10, 20))
        for i, (chain_region, lengths) in enumerate(chain_dict.items()):

            s = pd.Series(lengths)
            print(s.size)
            freqs = s.value_counts(normalize=True)

            # test dra
            axs[i].bar(freqs.index, freqs.values, alpha=0.7, color='c')  # 7行1列的subplot布局，当前是第1个subplot
            axs[i].set_title(f'Length distribution of {chain_region}')
            axs[i].set_xlabel('Length')
            axs[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(data_draw_path), f'{save_name[chain_idx]}.pdf'), dpi=300)
        plt.show()

if __name__ == '__main__':
    # cgz_path = '/data/home/waitma/antibody_proj/encoder/data/oas_pair_human_data/cgz_data/1287167_1_Paired_All.csv.gz'
    # parse_cgz_file(cgz_path)
    # main()
    # devide_train_and_valid(csv_path)
    # filter_length(csv_path, out_path)

    # statistic_different_region_length()
    # draw_the_different_length_distribution()
    # main()
    new_v = torch.tensor(HEAVY_CDR_INDEX) != 0
    mask_v = 152 - new_v.sum()
    new_x = torch.tensor(LIGHT_CDR_INDEX) != 0
    print(new_x.sum())
    mask_x = 139 - new_x.sum()
    print(mask_v)
    print(mask_x)