import os

import requests
import sys
import pandas as pd
import time
from tqdm import tqdm
import re
from abnumber import Chain
import concurrent.futures



T20_REGEX = re.compile('<td>T20 Score:</td><td>([0-9.]+)</td>')
def get_t20_online(seq):
    chain = Chain(seq, scheme='imgt')
    chain_type = 'vh' if chain.chain_type == 'H' else ('vl' if chain.chain_type == 'L' else 'vk')
    html = None
    for retry in range(5):
        url = f'https://sam.curiaglobal.com/t20/cgi-bin/blast.py?chain={chain_type}&region=1&output=3&seqs={seq}'
        try:
            request = requests.get(url)
            if request.ok:
                html = request.text
                break
        except Exception as e:
            print(e)
        time.sleep(0.5 + retry * 5)
        print('Retry', retry+1)
    if not html:
        sys.exit(1)
    # print(html)
    matches = T20_REGEX.findall(html)
    time.sleep(1)
    if not matches:
        print(html)
        # raise ValueError(f'Error calling url {url}')
        return None, None
    return float(matches[0]), chain_type

def get_pair_data_t20(h_seq, l_seq):
    h_score, h_type = get_t20_online(h_seq)
    l_score, l_type = get_t20_online(l_seq)
    print(h_score, l_score)
    return [h_score, h_type, l_score, l_type, h_seq, l_seq]

def process_line(line):
    h_seq = line[1]['hseq']
    l_seq = line[1]['lseq']
    name = [line[1]['name']]
    for retry in range(10):
        try:
            data = get_pair_data_t20(h_seq, l_seq)
            if len(data) > 2:
                break
        except:
            time.sleep(5)
            continue
    if len(data) != 2:
        new_data = name + data
        new_line_df = pd.DataFrame([new_data], columns=['Raw_name', 'h_score', 'h_gene', 'l_score', 'l_gene', 'h_seq', 'l_seq'])
        return new_line_df
    else:
        return None

def main():
    """
    Gather the T20 score from the website.
    :return:
    """
    sample_fpath = '/apdcephfs/share_1364275/waitma/anti_proj/log/' \
                   'v11_mul_pair_test_step2_2023_12_10__22_19_49/no_limit_sample_1_2023_12_14__17_41_18/sample_humanization_result.csv'
    # sample_fpath = '/data/home/waitma/antibody_proj/antidiff/data/lab_data/humanization_pair_data_filter.csv'
    save_fpath = os.path.join(os.path.dirname(sample_fpath), 'sample_t20_score.csv')

    sample_df = pd.read_csv(sample_fpath)
    # print(sample_df)
    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    # sample_human_df = sample_df[sample_df['type'] == 'mouse'].reset_index(drop=True)
    # print(sample_human_df)

    save_t20_df = pd.DataFrame(columns=['Raw_name', 'h_score', 'h_gene', 'l_score', 'l_gene', 'h_seq', 'l_seq'])
    # print(save_t20_df.columns)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_line, sample_human_df.iterrows()), total=len(sample_human_df)))

    save_t20_df = pd.concat([result for result in results if result is not None], ignore_index=True)
    Not_successful_index = [i for i, result in enumerate(results) if result is None]

    print(Not_successful_index)
    save_t20_df.to_csv(save_fpath, index=False)


if __name__ == '__main__':
    main()