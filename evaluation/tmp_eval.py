import os.path

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':
    pass

    # Lab data.
    lab_root_path = '/data/home/waitma/antibody_proj/antidiff/data/lab_data'
    t20_lab_fpath = os.path.join(lab_root_path, 't20_score_lab_filter_result.csv')
    z_lab_fpath = os.path.join(lab_root_path, 'z_score_lab_filter_result.csv')

    # Sample data.
    sample_root_path = '/data/home/waitma/antibody_proj/antidiff/checkpoints/batch_one_sample_2023_11_16__22_44_23'

    t20_sample_raw_fpath = os.path.join(sample_root_path, 't20_score_result.csv')
    z_sample_raw_fpath = os.path.join(sample_root_path, 'z_score_result.csv')

    t20_sample_filter_fpath = os.path.join(sample_root_path, 't20_score_filter_result.csv')
    z_sample_filter_fpath = os.path.join(sample_root_path, 'z_score_filter_result.csv')


    # Compare t20 H score.

    # lab_t20_df = pd.read_csv(t20_lab_fpath)
    # raw_t20_df = pd.read_csv(t20_sample_raw_fpath)
    # filter_t20_df = pd.read_csv(t20_sample_filter_fpath)
    #
    # sns.kdeplot(lab_t20_df['L_score'], fill=True, label='lab_VL')
    # sns.kdeplot(raw_t20_df['L_score'], fill=True, label='raw_VL')
    # sns.kdeplot(filter_t20_df['L_score'], fill=True, label='filter_VL')
    #
    # plt.xlabel("Score")
    # plt.ylabel("Density")
    # plt.title("L chain T20 Score Distributions")
    # plt.legend()
    # plt.show()

    # Compare z score.
    lab_z_df = pd.read_csv(z_lab_fpath)
    raw_z_df = pd.read_csv(z_sample_raw_fpath)
    filter_z_df = pd.read_csv(z_sample_filter_fpath)

    print(lab_z_df.keys())
    sns.kdeplot(lab_z_df['H_score'], fill=True, label='lab_VH')
    sns.kdeplot(raw_z_df['H_Score'], fill=True, label='raw_VH')
    sns.kdeplot(filter_z_df['H_score'], fill=True, label='filter_VH')

    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("H chain Z Score Distributions")
    plt.legend()
    plt.show()

