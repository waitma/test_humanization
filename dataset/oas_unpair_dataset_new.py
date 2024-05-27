import os
import pickle
import lmdb
import numpy as np

import torch
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F


from tfold_utils.prot_constants import RESD_NAMES_1C, RESD_WITH_X
from utils.tokenizer import Tokenizer
# from utils.train_utils import split_data

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


def _pad(tokenized, value, dim=2):
    """
    Utility function that pads batches to the same length.

    tokenized: list of tokenized sequences
    value: pad index
    """
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    if dim == 3: # dim = 3 (one hot)
        categories = tokenized[0].shape[-1]
        output = torch.zeros((batch_size, max_len, categories)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t), :] = t
    elif dim == 2: # dim = 2 (tokenized)
        output = torch.zeros((batch_size, max_len)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t)] = t
    else:
        print("padding not supported for dim > 3")
    return output


class OasUnPairDataset(Dataset):

    def __init__(self,
                data_dpath='/data/home/waitma/antibody_proj/antidiff/data/oas_unpair_human_data/heavy_unpair_order.pkl',
                chaintype='heavy', transform=None
                 ):
        super().__init__()
        self.raw_path = os.path.dirname(data_dpath)
        self.data_path = data_dpath
        self.processed_path = os.path.join(self.raw_path,
                                           f'{chaintype}_test_order.lmdb')
        self.index_path = os.path.join(self.raw_path,
                                       f'{chaintype}_idx.pt')
        self.transform = transform
        self.db = None

        self.keys = None

        # Filter set.
        self.Chn_seqs = set()

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None


    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=15 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )

        # # Deal Chain data.
        line_data_list = None
        with open(self.data_path, 'rb') as f:
            line_data_list = pickle.load(f)
            f.close()

        with db.begin(write=True, buffers=True) as txn:
            for line_idx, line in tqdm(enumerate(line_data_list)):
                name, chn_seq, pad_seq, chain_type, _ = line
                line_data = {
                    'name': name,
                    'seq': chn_seq,
                    'pad_seq': pad_seq,
                    'chain': chain_type
                }
                txn.put(
                    key=str(line_idx).encode(),
                    value=pickle.dumps(line_data)
                )
                line_idx += 1
        db.close()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        return data


def merge_stack(H_d, L_d):
    """
    Cat tensor for training.
    :param H_d: list of tensor,
    :param L_d: list of tensor
    :return: cat tensor
    """
    return torch.cat(
        (torch.stack(H_d), torch.stack(L_d)), dim=0
    )


class OasUnpairMaskCollater(object):
    """
    OrderAgnosic Mask Collater for masking batch data according to Hoogeboom et al. OA ARDMS
    inputs:
        list_sequences_dict : dict of H/L chain torch tensor, including the CDR regions.
        inputs_padded: if inputs are padded (due to truncation in Simple_Collater) set True (default False)

    OA-ARM variables:
        D : possible permutations from 0.. max length
        t : randomly selected timestep

    outputs:
        src : source  masked sequences (model input)
        timesteps: (D-t+1) term
        tokenized: tokenized sequences (target seq)
        masks: masks used to generate src
    """
    def __init__(self, tokenizer=Tokenizer()):
        self.tokenizer = tokenizer

    def __call__(self, list_sequences_dict, light_pad_v=0):
        chain_tokenized = [self.tokenizer.seq2idx(s_dict['pad_seq']) for s_dict in list_sequences_dict]
        chain_type = torch.tensor(
            [self.tokenizer.chain_type_idx(s_dict['chain']) for s_dict in list_sequences_dict]
        )
        type = list_sequences_dict[0]['chain']

        if type == 'L' or type == 'K':
            pad_length = len(HEAVY_CDR_INDEX) - len(LIGHT_CDR_INDEX)
            chain_tokenized = [F.pad(chain_t, (0, pad_length), 'constant', self.tokenizer.idx_pad)
                               for chain_t in chain_tokenized]
            chain_cdr_index = [F.pad(torch.tensor(LIGHT_CDR_INDEX), (0, pad_length), 'constant', light_pad_v)
                               for _ in range(len(list_sequences_dict))]
        else:
            chain_cdr_index = [torch.tensor(HEAVY_CDR_INDEX) for _ in range(len(list_sequences_dict))]
        chain_max_len = len(HEAVY_CDR_INDEX)
        chain_src = []
        chain_timesteps = []
        chain_masks = []
        chain_cdr_mask = []

        mask_id = torch.tensor(self.tokenizer.idx_msk, dtype=torch.int64)
        for i, ch_x in enumerate(chain_tokenized):
            # Randomly generate timestep and indices to mask
            ch_D = len(ch_x)   # D should have the same dimensions as each sequence length
            if ch_D <= 1:  # for sequence length = 1 in dataset
                ch_t = 1
            else:
                ch_t = np.random.randint(1, ch_D) # randomly sample timestep

            ch_num_mask = (ch_D-ch_t+1) # from OA-ARMS

            # Generate chain mask.
            ch_mask_arr = np.random.choice(ch_D, ch_num_mask, replace=False) # Generates array of len num_mask
            ch_index_arr = np.arange(0, chain_max_len) #index array [1...seq_len]
            ch_mask = np.isin(ch_index_arr, ch_mask_arr, invert=False).reshape(ch_index_arr.shape) # True represents mask, vice versa
            ch_cdr_mask = chain_cdr_index[i] != 0
            ch_mask = torch.tensor(ch_mask, dtype=torch.bool)
            ch_before_fix_true_number = ch_mask[:ch_D].sum()
            ch_mask[:ch_D] = ch_mask[:ch_D] * ~ch_cdr_mask
            ch_after_fix_true_number = ch_mask[:ch_D].sum()
            assert ch_before_fix_true_number >= ch_after_fix_true_number, 'H chain Mask has problem'
            ch_num_mask = ch_after_fix_true_number
            chain_masks.append(ch_mask)
            chain_cdr_mask.append(ch_cdr_mask)

            # Generate timestep H.
            ch_x_t = ~ch_mask[0:ch_D] * ch_x + ch_mask[0:ch_D] * mask_id
            chain_src.append(ch_x_t)

            # Append timestep
            chain_timesteps.append(ch_num_mask)

        # PAD src out
        # H_src = _pad(H_src, self.tokenizer.idx_pad)
        # L_src = _pad(L_src, self.tokenizer.idx_pad)

        # Pad mask out
        # H_masks = _pad(H_masks*1, 0) #, self.seq_length, 0)
        # L_masks = _pad(L_masks*1, 0)

        # Pad token out
        # H_tokenized = _pad(H_tokenized, self.tokenizer.idx_pad)
        # L_tokenized = _pad(L_tokenized, self.tokenizer.idx_pad)

        # tensor.
        chain_src = torch.stack(chain_src)
        chain_tokenized = torch.stack(chain_tokenized)
        chain_cdr_token = torch.stack(chain_cdr_index)
        chain_cdr_mask = torch.stack(chain_cdr_mask)
        chain_masks = torch.stack(chain_masks)
        chain_timesteps = torch.tensor(chain_timesteps)

        assert len(list_sequences_dict) == chain_src.shape[0], print("Wrong data collater!")
        return (chain_src, chain_tokenized,
                chain_cdr_token, chain_type, chain_masks,
                chain_cdr_mask, chain_timesteps)


if __name__ == '__main__':

    def inf_iterator(iterable):
        iterator = iterable.__iter__()
        while True:
            try:
                yield iterator.__next__()
            except StopIteration:
                iterator = iterable.__iter__()


    def split_data(path, dataset):
        split = torch.load(path)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return subsets
    # heavy_path = '/apdcephfs/share_1364275/waitma/anti_proj/data/oas_unpair_data/'
    root_path = '/data/home/waitma/antibody_proj/antidiff/data/oas_unpair_human_data/light_unpair_order.pkl'
    dataset = OasUnPairDataset(root_path, chaintype='light')


    # h_dataset = OasUnPairDataset(root_path, data_path=None, chaintype='Heavy')
    # l_dataset = OasUnPairDataset(root_path, data_path=None, chaintype='Light')
    # h_split_path = h_dataset.index_path
    split_path = dataset.index_path
    # h_subsets = split_data(h_split_path, h_dataset)
    subsets = split_data(split_path, dataset)
    #
    # h_train_dataset, h_val_dataset = h_subsets['train'], h_subsets['val']
    train_dataset, val_dataset = subsets['train'], subsets['val']
    collater = OasUnpairMaskCollater()

    test_data_loader = DataLoader(
        train_dataset,
        batch_size=24,
        collate_fn=lambda x: x
    )

    for batch in test_data_loader:
        collater(batch)
