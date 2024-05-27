import os.path

import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from abnumber import Chain

from utils.anti_numbering import get_regions
from dataset.build_human_pair_oas_new import region_padding_fix
from dataset.oas_pair_dataset_new import zero_batch, cdr_batch
from utils.tokenizer import Tokenizer
from utils.train_utils import model_selected
from utils.misc import get_new_log_dir, get_logger

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

    h_regions, h_length_list, h_chain_type = get_regions(mouse_aa_h)
    l_regions, l_length_list, l_chain_type = get_regions(mouse_aa_l)

    if not compare_length(h_length_list):
        raise AttributeError("H Length is too larger than predefined.")

    if not compare_length(l_length_list):
        raise AttributeError('L length is too larger than predefined.')

    seg_seqs_ms_h = get_diff_region_aa_seq(mouse_aa_h, h_length_list)
    seg_seqs_ms_l = get_diff_region_aa_seq(mouse_aa_l, l_length_list)

    pad_seg_seqs_ms_h = [
        region_padding_fix(seg_sq_h, fix_lgth=seg_lgth)
        for seg_sq_h, seg_lgth in zip(seg_seqs_ms_h, REGION_LENGTH)
    ]
    pad_seg_seqs_ms_l = [
        region_padding_fix(seg_sq_l, fix_lgth=seg_lgth)
        for seg_sq_l, seg_lgth in zip(seg_seqs_ms_l, REGION_LENGTH)
    ]

    pad_ms_hc_cdr = sum([zero_batch(seq)
                      if idx % 2 == 0 else cdr_batch(idx, seq) for idx, seq in enumerate(pad_seg_seqs_ms_h)], [])
    pad_ms_lc_cdr = sum([zero_batch(seq)
                      if idx % 2 == 0 else cdr_batch(idx, seq) for idx, seq in enumerate(pad_seg_seqs_ms_l)], [])

    # cdr index
    pad_ms_hc_cdr_index = torch.tensor(pad_ms_hc_cdr)
    pad_ms_lc_cdr_index = torch.tensor(pad_ms_lc_cdr)

    # residue type tokenizer.
    ms_tokenizer = Tokenizer()
    h_ms_pad_seq_tokenize = ms_tokenizer.seq2idx(''.join(pad_seg_seqs_ms_h))
    l_ms_pad_seq_tokenize = ms_tokenizer.seq2idx(''.join(pad_seg_seqs_ms_l))

    chain_type = torch.tensor([ms_tokenizer.chain_type_idx(c) for c in h_chain_type+l_chain_type]).to(device)

    # batch.
    h_l_ms_batch = torch.tensor([0, 1]).to(device)

    return (pad_ms_hc_cdr_index, pad_ms_lc_cdr_index,
            h_ms_pad_seq_tokenize, l_ms_pad_seq_tokenize,
            chain_type, h_l_ms_batch, ms_tokenizer)

def batch_input_element(mouse_sq_h, mouse_sq_l, batch_size=10):
    (pad_ms_hc_cdr_index, pad_ms_lc_cdr_index,
     h_ms_pad_seq_tokenize, l_ms_pad_seq_tokenize,
     chain_type, h_l_ms_batch, ms_tokenizer) = get_input_element(mouse_sq_h, mouse_sq_l)

    # Get mask.
    mask = pad_ms_hc_cdr_index == 0
    seq = np.arange(sum(REGION_LENGTH))
    loc = seq[mask]
    np.random.shuffle(loc)

    # initial mask.
    h_ms_pad_seq_tokenize[mask] = ms_tokenizer.idx_msk
    l_ms_pad_seq_tokenize[mask] = ms_tokenizer.idx_msk

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
    chain_type = chain_type.view(-1, 1).repeat(1, batch_size).view(-1)
    h_l_ms_batch = h_l_ms_batch.view(-1, 1).repeat(1, batch_size).view(-1)


    return h_l_pad_seq_sample, pad_ms_h_l_cdr_index, chain_type, h_l_ms_batch, loc, ms_tokenizer

def get_mouse_line(fpath):
    df_humanization = pd.read_csv(fpath)
    mouse_df = df_humanization[df_humanization['type']=='mouse']
    return mouse_df


if __name__ == '__main__':
    # mouse_aa_h = 'EVKLEESGGGLVQPGGSMKLSCVASGFTFSNFWMDWVRQSPEKGLEWI' \
    #              'AGIRLKSYNYATHYAESVKGRFTISRDDSKSSVYLQMNNLRAEDTGIYYCTDWDGAYWGQGTLVTVSA'
    # mouse_aa_l = 'DIVMTQSHKFMSTSVGDRVSITCKASQDVSTDVAWYQQKPGQSPKLLI' \
    #              'YSASYRYTGVPDRFTGSGSGTDFTFTISSVQAEDLAVYYCQQHYSTPFTFGSGTKLEIK'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='/data/home/waitma/antibody_proj/antidiff/checkpoints/582.pt'
                        default='/apdcephfs/share_1364275/waitma/anti_proj/log/v08_2023_11_09__20_22_02/checkpoints/582.pt'
                    )
    parser.add_argument('--data_fpath', type=str,
                        # default='/data/home/waitma/antibody_proj/antidiff/data/lab_data/humanization_pair_data_filter.csv'
                        default = '/apdcephfs/share_1364275/waitma/anti_proj/data/lab_data/humanization_pair_data_filter.csv'
                    )
    parser.add_argument('--batch_size', type=int,
                        default=1
                    )
    parser.add_argument('--sample_tag', type=str,
                        default='test_sample_mouse_1'
                    )

    args = parser.parse_args()

    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # log dir
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
    model.load_state_dict(ckpt['model'])

    # save path
    save_fpath = os.path.join(log_dir, 'sample_humanization_result.csv')
    with open(save_fpath, 'a', encoding='UTF-8') as f:
        f.write('Specific,name,hseq,lseq,\n')

    wrong_idx_list = []
    mouse_df = get_mouse_line(args.data_fpath)
    for m_i, mouse_line in tqdm(enumerate(mouse_df.itertuples()), total=len(mouse_df.index)):

        sample_batch = 1
        try_num = 5
        mouse_aa_h = Chain(mouse_line.h_seq, scheme='imgt').seq
        mouse_aa_l = Chain(mouse_line.l_seq, scheme='imgt').seq
        # mouse_aa_h = mouse_line.h_seq
        # mouse_aa_l = mouse_line.l_seq
        origin = 'mouse'
        name = mouse_line.name

        with open(save_fpath, 'a', encoding='UTF-8') as f:
            f.write(f'{origin},{name},{mouse_aa_h},{mouse_aa_l}\n')

        try:
            (h_l_pad_seq_sample, pad_ms_h_l_cdr_index,
            chain_type, h_l_ms_batch, loc, ms_tokenizer) = batch_input_element(mouse_aa_h, mouse_aa_l, batch_size)
        except:
            logger.info('Wrong idx {}'.format(m_i))
            wrong_idx_list.append(m_i)
            continue
        while sample_batch > 0 and try_num > 0:
            all_token = ms_tokenizer.toks
            with torch.no_grad():
                for i in tqdm(loc):
                    h_l_prediction = model(
                        h_l_pad_seq_sample.to(device),
                        pad_ms_h_l_cdr_index.to(device),
                        chain_type.to(device),
                        h_l_ms_batch.to(device),
                        'pair'
                    )
                    h_l_pred = h_l_prediction[:, i, :len(all_token)-1]
                    h_l_soft = torch.nn.functional.softmax(h_l_pred, dim=1)
                    h_l_sample = torch.multinomial(h_l_soft, num_samples=1)
                    h_l_pad_seq_sample[:, i] = h_l_sample.squeeze()

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
            # if try_num == 1:
            #     g_h, g_l = h_untokenized[0], l_untokenized[0]
            #     with open(save_fpath, 'a', encoding='UTF-8') as f:
            #         sample_origin = 'humanization'
            #         sample_name = str(name) + 'human_' + f'{i}'
            #         f.write(f'{sample_origin},{sample_name},{g_h},{g_l}\n')
            #         sample_batch -= 1
            # with open(save_fpath, 'a', encoding='UTF-8') as f:
            #     f.write(f'{origin},{name},{mouse_aa_h},{mouse_aa_l}\n')
            #     for i, (human_h, human_l) in enumerate(zip(h_untokenized, l_untokenized)):
            #         sample_origin = 'humanization'
            #         sample_name = str(name) + 'human_' + f'{i}'
            #         f.write(f'{sample_origin},{sample_name},{human_h},{human_l}\n')




