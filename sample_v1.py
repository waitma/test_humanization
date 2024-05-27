import os.path

import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from abnumber import Chain
from anarci import anarci, number
from copy import deepcopy
import re

from utils.anti_numbering import get_regions
from dataset.build_human_pair_oas_new import (region_padding_fix,
                                              HEAVY_POSITIONS_dict, LIGHT_POSITIONS_dict,
                                              HEAVY_CDR_INDEX, LIGHT_CDR_INDEX)
from dataset.oas_pair_dataset_new import light_pad_cdr
from utils.tokenizer import Tokenizer
from utils.train_utils import model_selected
from utils.misc import get_new_log_dir, get_logger
from multi_train import convert_multi_gpu_checkpoint_to_single_gpu

REGION_LENGTH = (26, 12, 17, 10, 38, 30, 11)

def compare_length(length_list):
    small = True
    for i, lg in enumerate(length_list):
        if lg <= REGION_LENGTH[i]:
            continue
        else:
            small = False
    return small

def get_diff_region_aa_seq(raw_seq, length_list):
    split_aa_seq_list = []
    start_lg = 0
    for lg in length_list:
        end_lg = start_lg + lg
        aa_seq = raw_seq[start_lg:end_lg]
        split_aa_seq_list.append(aa_seq)
        start_lg = end_lg
    assert ''.join(split_aa_seq_list) == raw_seq, 'Split length has wrong.'
    return split_aa_seq_list


def get_pad_seq(aa_seq):
    """
    :param aa_seq: AA seqs.
    :return: the pading AA seqs.
    """
    seq_dict = {}
    results = number(aa_seq, scheme='imgt')

    for key, value in results[0]:
        str_key = str(key[0]) + key[1].strip()
        seq_dict[str_key] = value
    seq_chain_type = results[1]
    return seq_dict, seq_chain_type



def get_input_element(mouse_aa_h, mouse_aa_l):
    """
    :param mouse_h:
    :param mouse_l:
    :return:
    """
    # 1. Make sure the length of the sequence;
    # 2. Get the index of different region;
    # 3. Padding the sequence;
    # 4. Mask the sequence and get mask index (need shuffle);
    # 5. sample aa by aa.

    h_seq_dict, h_chain_type = get_pad_seq(mouse_aa_h)
    l_seq_dict, l_chain_type = get_pad_seq(mouse_aa_l)

    # if not compare_length(h_length_list):
    #     raise AttributeError("H Length is too larger than predefined.")
    #
    # if not compare_length(l_length_list):
    #     raise AttributeError('L length is too larger than predefined.')
    h_cdr_index = deepcopy(HEAVY_CDR_INDEX)
    l_cdr_index = light_pad_cdr(deepcopy(LIGHT_CDR_INDEX), pad_v=0)

    h_pad_cdr_index = torch.tensor(h_cdr_index)
    l_pad_cdr_index = torch.tensor(l_cdr_index)

    h_pad_initial_seq = ['-'] * len(HEAVY_CDR_INDEX)
    for key, value in h_seq_dict.items():
        try:
            pos_idx = HEAVY_POSITIONS_dict[key]
            h_pad_initial_seq[pos_idx] = value
        except KeyError:
            nkey = re.findall(r'\d+', key)
            nkey = int(nkey[0])
            if (27 <= nkey <= 38) or (56 <= nkey <= 65) or ( 105 <= nkey <= 117):
                print("Heavy CDR has problem.")
            else:
                print('H Position {} is not in predefine dict, which can be ignored.'.format(key))

    l_pad_initial_seq = ['-'] * len(HEAVY_CDR_INDEX)
    for key, value in l_seq_dict.items():
        try:
            pos_idx = LIGHT_POSITIONS_dict[key]
            l_pad_initial_seq[pos_idx] = value
        except KeyError:
            nkey = re.findall(r'\d+', key)
            nkey = int(nkey[0])
            if (27 <= nkey <= 38) or (56 <= nkey <= 65) or ( 105 <= nkey <= 117):
                print("Light CDR has problem.")
            else:
                print('L Position {} is not in predefine dict, which can be ignored.'.format(key))


    # chain_type = torch.tensor([ms_tokenizer.chain_type_idx(c) for c in h_chain_type+l_chain_type]).to(device)
    chain_type = [h_chain_type, l_chain_type]
    # batch.
    h_l_ms_batch = torch.tensor([0, 1])

    return (h_pad_cdr_index, l_pad_cdr_index,
            h_pad_initial_seq, l_pad_initial_seq,
            chain_type, h_l_ms_batch)


def batch_input_element(mouse_sq_h, mouse_sq_l, batch_size=10):
    (pad_ms_hc_cdr_index, pad_ms_lc_cdr_index,
     h_ms_pad_seq, l_ms_pad_seq,
     chain_type, h_l_ms_batch) = get_input_element(mouse_sq_h, mouse_sq_l)

    # Get mask. Do not change CDR region.
    h_mask = torch.tensor(HEAVY_CDR_INDEX) == 0
    l_mask = torch.tensor(LIGHT_CDR_INDEX) == 0
    # print(h_mask)
    h_seq = np.arange(len(HEAVY_CDR_INDEX))
    l_seq = np.arange(len(LIGHT_CDR_INDEX))
    h_loc = h_seq[h_mask]
    l_loc = l_seq[l_mask]
    l_loc = np.append(l_loc, len(HEAVY_CDR_INDEX)-1)
    np.random.shuffle(h_loc)
    np.random.shuffle(l_loc)
    # h_loc == 93, l_loc == 92. for consistent.
    # l_loc.append(151)

    # initial mask.
    ms_tokenizer = Tokenizer()
    h_ms_pad_seq_tokenize = ms_tokenizer.seq2idx(h_ms_pad_seq)
    l_ms_pad_seq_tokenize = ms_tokenizer.seq2idx(l_ms_pad_seq)
    h_ms_pad_seq_tokenize[h_mask] = ms_tokenizer.idx_msk

    l_mask = torch.cat((l_mask, torch.tensor([1]*13).bool()), dim=0)
    l_ms_pad_seq_tokenize[l_mask] = ms_tokenizer.idx_msk

    pad_ms_h_l_cdr_index = torch.cat(
        (
            pad_ms_hc_cdr_index.unsqueeze(0).expand(batch_size, -1),
            pad_ms_lc_cdr_index.unsqueeze(0).expand(batch_size, -1)
        ), dim=0
    )
    h_l_pad_seq_sample = torch.cat(
        (
            h_ms_pad_seq_tokenize.unsqueeze(0).expand(batch_size, -1),
            l_ms_pad_seq_tokenize.unsqueeze(0).expand(batch_size, -1)
        ), dim=0
    )
    chain_type = torch.tensor([ms_tokenizer.chain_type_idx(c) for c in chain_type])
    chain_type = chain_type.view(-1, 1).repeat(1, batch_size).view(-1)
    h_l_ms_batch = h_l_ms_batch.view(-1, 1).repeat(1, batch_size).view(-1)
    print(h_l_pad_seq_sample)


    return h_l_pad_seq_sample, pad_ms_h_l_cdr_index, chain_type, h_l_ms_batch, h_loc, l_loc, ms_tokenizer


def get_mouse_line(fpath):
    df_humanization = pd.read_csv(fpath)
    mouse_df = df_humanization[df_humanization['type'] == 'mouse']
    return mouse_df


if __name__ == '__main__':
    # mouse_aa_h = 'EVKLEESGGGLVQPGGSMKLSCVASGFTFSNFWMDWVRQSPEKGLEWI' \
    #              'AGIRLKSYNYATHYAESVKGRFTISRDDSKSSVYLQMNNLRAEDTGIYYCTDWDGAYWGQGTLVTVSA'
    # mouse_aa_l = 'DIVMTQSHKFMSTSVGDRVSITCKASQDVSTDVAWYQQKPGQSPKLLI' \
    #              'YSASYRYTGVPDRFTGSGSGTDFTFTISSVQAEDLAVYYCQQHYSTPFTFGSGTKLEIK'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='/data/home/waitma/antibody_proj/antidiff/checkpoints/108.pt'
                        default='/apdcephfs/share_1364275/waitma/anti_proj/log/v11_mul_pair_test_step2_2023_12_10__22_19_49/checkpoints/108.pt'
                    )
    parser.add_argument('--data_fpath', type=str,
                        # default='/data/home/waitma/antibody_proj/antidiff/data/lab_data/humanization_pair_data_filter.csv'
                        default = '/apdcephfs/share_1364275/waitma/anti_proj/data/lab_data/humanization_pair_data_filter.csv'
                    )
    parser.add_argument('--batch_size', type=int,
                        default=1
                    )
    parser.add_argument('--sample_tag', type=str,
                        default='batch_one_sample'
                    )

    args = parser.parse_args()

    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # log dir
    # This is tmp, only for test the result.
    # log_path = '/data/home/waitma/antibody_proj/antidiff/checkpoints/sample_test'
    log_path = os.path.dirname(os.path.dirname(args.ckpt))
    # log_path = os.path.dirname(args.ckpt)
    log_dir = get_new_log_dir(
        root=log_path,
        prefix=args.sample_tag
    )
    logger = get_logger('test', log_dir)
    # load model check point.

    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    model = model_selected(config).to(device)

    ckpt_model = convert_multi_gpu_checkpoint_to_single_gpu(ckpt)
    model.load_state_dict(ckpt_model)
    # model.load_state_dict(ckpt['model'])

    # save path
    save_fpath = os.path.join(log_dir, 'sample_humanization_result.csv')
    with open(save_fpath, 'a', encoding='UTF-8') as f:
        f.write('Specific,name,hseq,lseq,\n')

    wrong_idx_list = []
    mouse_df = get_mouse_line(args.data_fpath)
    for idx, mouse_line in tqdm(enumerate(mouse_df.itertuples()), total=len(mouse_df.index)):
        sample_batch = 1
        try_num = 5
        mouse_aa_h = Chain(mouse_line.h_seq, scheme='imgt').seq
        mouse_aa_l = Chain(mouse_line.l_seq, scheme='imgt').seq
        # mouse_aa_h = mouse_line.h_seq
        # mouse_aa_l = mouse_line.l_seq
        print(mouse_aa_h)
        print(mouse_aa_l)
        origin = 'mouse'
        name = mouse_line.name
        with open(save_fpath, 'a', encoding='UTF-8') as f:
            f.write(f'{origin},{name},{mouse_aa_h},{mouse_aa_l}\n')

        try:
            (h_l_pad_seq_sample, pad_ms_h_l_cdr_index,
            chain_type, h_l_ms_batch, h_loc, l_loc, ms_tokenizer) = batch_input_element(mouse_aa_h, mouse_aa_l, batch_size)
        except:
            logger.info('Wrong idx {}'.format(idx))
            wrong_idx_list.append(idx)
            continue
        while sample_batch > 0 and try_num > 0:
            all_token = ms_tokenizer.toks
            with torch.no_grad():
                for h_i, l_i in tqdm(zip(h_loc, l_loc), total=len(h_loc)):
                    h_l_prediction = model(
                        h_l_pad_seq_sample.to(device),
                        pad_ms_h_l_cdr_index.to(device),
                        chain_type.to(device),
                        h_l_ms_batch.to(device),
                        'pair'
                    )
                    h_prediction = h_l_prediction[:batch_size]
                    h_pred = h_prediction[:, h_i, :len(all_token) - 1]
                    h_soft = torch.nn.functional.softmax(h_pred, dim=1)
                    h_sample = torch.multinomial(h_soft, num_samples=1)
                    h_pad_seq_sample = h_l_pad_seq_sample[:batch_size]
                    h_pad_seq_sample[:, h_i] = h_sample.squeeze()

                    l_prediction = h_l_prediction[batch_size:]
                    if l_i != len(HEAVY_CDR_INDEX)-1:
                        l_pred = l_prediction[:, l_i, :len(all_token) - 1]
                        l_soft = torch.nn.functional.softmax(l_pred, dim=1)
                        l_sample = torch.multinomial(l_soft, num_samples=1)
                        l_pad_seq_sample = h_l_pad_seq_sample[batch_size:]
                        l_pad_seq_sample[:, l_i] = l_sample.squeeze()
                    else:
                        l_pad_seq_sample = h_l_pad_seq_sample[batch_size:]

                    h_l_pad_seq_sample = torch.cat(
                        (h_pad_seq_sample, l_pad_seq_sample),
                        dim=0
                    )
                    # if i > len(LIGHT_CDR_INDEX):


            h_l_untokenized = [ms_tokenizer.idx2seq(s) for s in h_l_pad_seq_sample]
            h_untokenized = h_l_untokenized[:batch_size]
            l_untokenized = h_l_untokenized[batch_size:]
            for i, (g_h, g_l) in enumerate(zip(h_untokenized, l_untokenized)):
                # if len(g_h) == len(mouse_aa_h) and len(g_l) == len(mouse_aa_l) and sample_batch > 0:
                with open(save_fpath, 'a', encoding='UTF-8') as f:
                    sample_origin = 'humanization'
                    sample_name = str(name) + 'human_' + f'{i}'
                    f.write(f'{sample_origin},{sample_name},{g_h},{g_l}\n')

                    sample_batch -= 1
                if sample_batch == 0:
                    break
    print(wrong_idx_list)
                # else:
                #     # try_num -= 1
                #     print('Generate length is wrong, need to retry sample batchsize: {}'.format(sample_batch))
            # try_num -= 1
            # if try_num == 0:
            #     g_h, g_l = h_untokenized[0], l_untokenized[0]
            #     with open(save_fpath, 'a', encoding='UTF-8') as f:
            #         sample_origin = 'humanization'
            #         sample_name = str(name) + 'human_' + f'{i}'
            #         f.write(f'{sample_origin},{sample_name},{g_h},{g_l}\n')
            #         sample_batch -= 1



