"""Build a CSV file containing the meta data."""

import os
import json

import pandas as pd
from tqdm import tqdm


def main():
    """Main entry."""

    # configurations
    data_dir = '/data/home/waitma/antibody_proj/encoder/data/oas_pair_human_data/'
    cgz_dpath = os.path.join(data_dir, 'cgz_data')
    csv_fpath = os.path.join(data_dir, 'meta_data.csv')

    # fetch the meta data from GZ-compressed CSV files
    meta_keys = None
    meta_data_dict = {}
    for cgz_fname in tqdm(os.listdir(cgz_dpath), desc='fetching the meta data'):
        cgz_fpath = os.path.join(cgz_dpath, cgz_fname)
        header = ','.join(pd.read_csv(cgz_fpath, skiprows=1, nrows=0).columns)
        meta_data = json.loads(header)
        meta_data_dict[cgz_fname.replace('.csv.gz', '')] = meta_data
        if meta_keys is None:
            meta_keys = list(meta_data.keys())

    # build a CSV file containing the meta data
    with open(csv_fpath, 'w', encoding='UTF-8') as o_file:
        sub_strs = ['Entry'] + meta_keys
        o_file.write(','.join([x if ',' not in x else f'"{x}"' for x in sub_strs]) + '\n')
        for entry, meta_data in meta_data_dict.items():
            sub_strs = [entry] + [str(meta_data[x]) for x in meta_keys]
            o_file.write(','.join([x if ',' not in x else f'"{x}"' for x in sub_strs]) + '\n')


if __name__ == '__main__':
    main()