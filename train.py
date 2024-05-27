import os.path
import pickle
import numpy as np
import argparse
import yaml
from easydict import EasyDict
import shutil

import torch
import torch.utils.tensorboard
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from dataset.oas_pair_dataset_new import OasPairMaskCollater
from dataset.oas_unpair_dataset_new import OasUnpairMaskCollater
from torch.utils.data import DataLoader
from utils.train_utils import model_selected, optimizer_selected, scheduler_selected
from utils.misc import seed_all, get_new_log_dir, get_logger, inf_iterator, count_parameters
from utils.loss import OasMaskedCrossEntropyLoss, MaskedAccuracy
from utils.train_utils import get_dataset


def convert_multi_gpu_checkpoint_to_single_gpu(checkpoint):
    if 'module' in list(checkpoint['model'].keys())[0]:
        new_state_dict = {}
        for key, value in checkpoint['model'].items():
            new_key = key.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        checkpoint['model'] = new_state_dict
    return checkpoint['model']


def freeze_parameters(block):
    for x in block:
        x.requires_grad = False

def unfreeze_parameters(block):
    for x in block:
        x.requires_grad = True

def train(it, train_type):
    sum_loss = 0
    H_L_sum_loss, H_L_sum_nll = 0., 0.
    H_L_sum_cdr_loss = 0.
    H_L_sum_acc_loss = 0.
    sum_roc_auc = 0.

    model.train()
    if train_type == 'unpair':
        for _ in range(config.train.batch_acc):
            optimizer.zero_grad()
            h_src, h_tgt, h_cdr_index, h_type, h_masks, h_cdr_masks, h_timesteps = next(h_train_iterator)
            l_src, l_tgt, l_cdr_index, l_type, l_masks, l_cdr_masks, l_timesteps = next(l_train_iterator)
            h_l_src = torch.cat((h_src, l_src), dim=0).to(device)
            h_l_tgt = torch.cat((h_tgt, l_tgt), dim=0).to(device)
            h_l_cdr_index = torch.cat((h_cdr_index, l_cdr_index), dim=0).to(device)
            h_l_chain = torch.cat((h_type, l_type), dim=0).to(device)
            h_l_masks = torch.cat((h_masks, l_masks), dim=0).to(device)
            h_l_cdr_masks = torch.cat((h_cdr_masks, l_cdr_masks), dim=0).to(device)
            h_l_timesteps = torch.cat((h_timesteps, l_timesteps), dim=0).to(device)
            batch = torch.zeros(len(h_l_src)).to(device)
            batch[len(l_src):] = 1

            h_l_pred = model(h_l_src, h_l_cdr_index, h_l_chain, batch, train_type)

            h_l_loss, h_l_nll, h_l_cdr_loss = cross_loss(
                                                h_l_pred,
                                                h_l_tgt,
                                                h_l_masks,
                                                h_l_cdr_masks,
                                                h_l_timesteps
                                              )
            h_l_acc_loss, roc_auc = mask_acc_loss(h_l_pred, h_l_tgt, h_l_masks)

            loss = h_l_loss + h_l_cdr_loss
            loss.mean()
            loss.backward()

            # clip grad norm

            optimizer.step()

            sum_loss += loss
            H_L_sum_loss += h_l_loss
            H_L_sum_nll += h_l_nll
            H_L_sum_cdr_loss += h_l_cdr_loss

            # Not backward.
            H_L_sum_acc_loss += h_l_acc_loss
            sum_roc_auc += roc_auc



    elif train_type == 'pair':
        for _ in range(config.train.batch_acc):
            optimizer.zero_grad()
            (H_L_src, H_L_tgt, H_L_region, chain_type, batch,
             H_L_masks, H_L_cdr_masks,
             H_L_timesteps) = next(train_iterator)
            H_L_src, H_L_tgt = H_L_src.to(device), H_L_tgt.to(device)
            H_L_region = H_L_region.to(device)
            chain_type, batch = chain_type.to(device), batch.to(device)
            H_L_masks, H_L_cdr_masks = H_L_masks.to(device), H_L_cdr_masks.to(device)
            H_L_timesteps = H_L_timesteps.to(device)

            H_L_pred = model(H_L_src, H_L_region, chain_type, batch, train_type)

            H_L_loss, H_L_nll, H_L_cdr_loss = cross_loss(
                                                H_L_pred,
                                                H_L_tgt,
                                                H_L_masks,
                                                H_L_cdr_masks,
                                                H_L_timesteps
                                              )
            # Those value indicate whether the pred equl to tgt. max may is 1.
            H_L_acc_loss, roc_auc = mask_acc_loss(H_L_pred, H_L_tgt, H_L_masks)

            loss = H_L_loss + H_L_cdr_loss
            loss.mean()
            loss.backward()

            # clip grad norm

            optimizer.step()

            sum_loss += loss
            H_L_sum_loss += H_L_loss
            H_L_sum_nll += H_L_nll
            H_L_sum_cdr_loss += H_L_cdr_loss

            # Not backward.
            H_L_sum_acc_loss += H_L_acc_loss
            sum_roc_auc += roc_auc

    mean_loss = sum_loss / config.train.batch_acc
    mean_H_L_loss = H_L_sum_loss / config.train.batch_acc
    mean_H_L_nll = H_L_sum_nll / config.train.batch_acc

    mean_H_L_cdr_loss = H_L_sum_cdr_loss / config.train.batch_acc

    # Not backward.
    mean_H_L_acc_loss = H_L_sum_acc_loss / config.train.batch_acc
    mean_roc_auc = sum_roc_auc / config.train.batch_acc

    logger.info('Training iter {}, Loss is: {:.6f} | H_L_loss: {:.6f} | H_L_nll: {:.6f} '
                '| H_L_cdr_loss: {:.6f} | H_L_acc: {:.6f} | ROC_AUC: {:.6f}'.
                format(it, mean_loss, mean_H_L_loss, mean_H_L_nll, mean_H_L_cdr_loss, mean_H_L_acc_loss, mean_roc_auc))
    writer.add_scalar('train/loss', mean_loss, it)
    writer.add_scalar('train/H_L_loss', mean_H_L_loss, it)
    writer.add_scalar('train/H_L_nll', mean_H_L_nll, it)
    writer.add_scalar('train/H_L_cdr_loss', mean_H_L_cdr_loss, it)
    writer.add_scalar('train/H_L_acc', mean_H_L_acc_loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/roc_auc', mean_roc_auc, it)


def valid(it, valid_type):
    sum_valid_loss = 0.
    H_L_sum_loss, H_L_sum_nll = 0., 0.
    H_L_sum_acc_loss = 0.
    H_L_sum_cdr_loss = 0.
    sum_roc_auc = 0.
    model.eval()
    if valid_type == 'unpair':
        with torch.no_grad():
            for h_batch, l_batch in tqdm(zip(h_val_loader, l_val_loader), desc='Val'):
                h_src, h_tgt, h_cdr_index, h_type, h_masks, h_cdr_masks, h_timesteps = h_batch
                l_src, l_tgt, l_cdr_index, l_type, l_masks, l_cdr_masks, l_timesteps = l_batch
                h_l_src = torch.cat((h_src, l_src), dim=0).to(device)
                h_l_tgt = torch.cat((h_tgt, l_tgt), dim=0).to(device)
                h_l_cdr_index = torch.cat((h_cdr_index, l_cdr_index), dim=0).to(device)
                h_l_chain = torch.cat((h_type, l_type), dim=0).to(device)
                h_l_masks = torch.cat((h_masks, l_masks), dim=0).to(device)
                h_l_cdr_masks = torch.cat((h_cdr_masks, l_cdr_masks), dim=0).to(device)
                h_l_timesteps = torch.cat((h_timesteps, l_timesteps), dim=0).to(device)
                batch = torch.zeros(len(h_l_src))
                batch[len(l_src):] = 1

                h_l_pred = model(h_l_src, h_l_cdr_index, h_l_chain, batch, valid_type)

                h_l_loss, h_l_nll, h_l_cdr_loss = cross_loss(
                    h_l_pred,
                    h_l_tgt,
                    h_l_masks,
                    h_l_cdr_masks,
                    h_l_timesteps
                )
                h_l_acc_loss, roc_auc = mask_acc_loss(h_l_pred, h_l_tgt, h_l_masks)

                loss = h_l_loss + h_l_cdr_loss

                sum_valid_loss += loss
                H_L_sum_loss += h_l_loss
                H_L_sum_nll += h_l_nll
                H_L_sum_cdr_loss += h_l_cdr_loss

                # Not backward.
                H_L_sum_acc_loss += h_l_acc_loss
                sum_roc_auc += roc_auc

                mean_loss = sum_valid_loss / len(h_val_loader)
                mean_H_L_loss = H_L_sum_loss / len(h_val_loader)
                mean_H_L_nll = H_L_sum_nll / len(h_val_loader)
                mean_H_L_cdr_loss = H_L_sum_cdr_loss / len(h_val_loader)

                # Not backward.
                mean_H_L_acc_loss = H_L_sum_acc_loss / len(h_val_loader)
                mean_roc_auc = sum_roc_auc / len(h_val_loader)


    elif valid_type == 'pair':
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val'):
                (H_L_src, H_L_tgt, H_L_region, chain_type, batch,
                 H_L_masks, H_L_cdr_masks,
                 H_L_timesteps) = batch
                H_L_src, H_L_tgt = H_L_src.to(device), H_L_tgt.to(device)
                H_L_region = H_L_region.to(device)
                chain_type, batch = chain_type.to(device), batch.to(device)
                H_L_masks, H_L_cdr_masks = H_L_masks.to(device), H_L_cdr_masks.to(device)
                H_L_timesteps = H_L_timesteps.to(device)

                H_L_pred = model(H_L_src, H_L_region, chain_type, batch, valid_type)

                H_L_loss, H_L_nll, H_L_cdr_loss = cross_loss(
                    H_L_pred,
                    H_L_tgt,
                    H_L_masks,
                    H_L_cdr_masks,
                    H_L_timesteps
                )
                # Those value indicate whether the pred equl to tgt. max may is 1.
                H_L_acc_loss, roc_auc = mask_acc_loss(H_L_pred, H_L_tgt, H_L_masks)

                loss = H_L_loss + H_L_cdr_loss

                sum_valid_loss += loss
                H_L_sum_loss += H_L_loss
                H_L_sum_nll += H_L_nll
                H_L_sum_cdr_loss += H_L_cdr_loss

                # Not backward.
                H_L_sum_acc_loss += H_L_acc_loss
                sum_roc_auc += roc_auc

                mean_loss = sum_valid_loss / len(val_loader)
                mean_H_L_loss = H_L_sum_loss / len(val_loader)
                mean_H_L_nll = H_L_sum_nll / len(val_loader)
                mean_H_L_cdr_loss = H_L_sum_cdr_loss / len(val_loader)

                # Not backward.
                mean_H_L_acc_loss = H_L_sum_acc_loss / len(val_loader)
                mean_roc_auc = sum_roc_auc /len(val_loader)

    scheduler.step(mean_loss)

    logger.info('Validation iter {}, Loss is: {:.6f} | H_L_loss: {:.6f} | H_L_nll: {:.6f} '
                '| H_L_cdr_loss: {:.6f} | H_L_acc: {:.6f} | ROC_AUC: {:.6f}'.
                format(it, mean_loss, mean_H_L_loss, mean_H_L_nll,
                       mean_H_L_cdr_loss, mean_H_L_acc_loss, mean_roc_auc))
    writer.add_scalar('val/loss', mean_loss, it)
    writer.add_scalar('val/H_L_loss', mean_H_L_loss, it)
    writer.add_scalar('val/H_L_nll', mean_H_L_nll, it)
    writer.add_scalar('val/H_L_cdr_loss', mean_H_L_cdr_loss, it)
    writer.add_scalar('val/H_L_acc', mean_H_L_acc_loss, it)
    writer.add_scalar('val/roc_auc', mean_roc_auc, it)

    return mean_loss


if __name__ == '__main__':
    # Required args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_data_path', type=str,
                        default='/data/home/waitma/antibody_proj/antidiff/data/oas_pair_human_data/'
                        # default='/apdcephfs/share_1364275/waitma/anti_proj/data/oas_pair_data/'
                        )
    parser.add_argument('--unpair_data_path', type=str,
                        default='/data/home/waitma/antibody_proj/antidiff/data/oas_unpair_human_data/'
                        # default='/apdcephfs/share_1364275/waitma/anti_proj/data/oas_unpair_data/'
                        )
    parser.add_argument('--data_name', type=str,
                        default='pair', choices=['pair', 'unpair'])
    parser.add_argument('--train_model', type=str,
                        default='not_pretrain', choices=['pretrain', 'not_pretrain'])
    parser.add_argument('--data_version', type=str,
                        default='test')
    parser.add_argument('--config_path', type=str,
                        default='/data/home/waitma/antibody_proj/antidiff/configs/training.yml'
                        # default='/apdcephfs/share_1364275/waitma/anti_proj/v001/configs/training.yml'
                        )
    parser.add_argument('--log_path', type=str,
                        default='/data/home/waitma/antibody_proj/antidiff/log'
                        # default='/apdcephfs/share_1364275/waitma/anti_proj/log'
                        )
    parser.add_argument('--version', type=str,
                        default='v11_unpair_test')
    parser.add_argument('--resume', type=bool,
                        default=False)
    parser.add_argument('--checkpoint', type=str,
                        default='/data/home/waitma/antibody_proj/antidiff/checkpoints/290.pt')

    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Config parameters.
    if not args.resume:
        with open(args.config_path, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
    else:
        assert args.checkpoint != '', "Need Specified Checkpoint."
        ckpt_path = args.checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        config = ckpt['config']

    # Create Log dir.
    log_dir = get_new_log_dir(
        root=args.log_path,
        prefix=args.version
    )

    # Checkpoints dir.
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # logger and writer
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    logger.info(args)
    logger.info(config)


    # Copy files for checking.
    shutil.copyfile(args.config_path, os.path.join(log_dir, os.path.basename(args.config_path)))
    shutil.copyfile('./train.py', os.path.join(log_dir, 'train.py'))
    shutil.copytree('./model', os.path.join(log_dir, 'model'))


    # Fixed
    seed_all(config.train.seed)

    # Create dataloader.
    if args.data_name == 'pair':
        subsets = get_dataset(args.pair_data_path, args.data_name, args.data_version)
        train_dataset, val_dataset = subsets['train'], subsets['val']
        collater = OasPairMaskCollater()

        # Pair
        train_iterator = inf_iterator(DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            # num_workers=config.train.num_workers,
            collate_fn=collater
        ))
        logger.info(f'Training: {len(train_dataset)} Validation: {len(val_dataset)}')
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            # num_workers=config.train.num_workers,
            collate_fn=collater
        )
        logger.info('Dataloader has created!')

    elif args.data_name == 'unpair':
        h_subsets, l_subsets = get_dataset(args.unpair_data_path, args.data_name, args.data_version)
        h_train_dataset, h_val_dataset = h_subsets['train'], h_subsets['val']
        l_train_dataset, l_val_dataset = l_subsets['train'], l_subsets['val']
        collater = OasUnpairMaskCollater()

        # Unpair
        h_train_iterator = inf_iterator(
            DataLoader(
                h_train_dataset,
                batch_size=config.train.batch_size,
                num_workers=config.train.num_workers,
                shuffle=True,
                collate_fn=collater
            )
        )
        l_train_iterator = inf_iterator(
            DataLoader(
                l_train_dataset,
                batch_size=config.train.batch_size,
                num_workers=config.train.num_workers,
                shuffle=True,
                collate_fn=collater
            )
        )
        logger.info(f'Training H and L: {len(h_train_dataset), len(l_train_dataset)} '
                    f'Validation H and L: {len(h_val_dataset), len(l_val_dataset)}')
        # validation.
        h_val_loader = DataLoader(
            h_val_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            collate_fn=collater
        )
        l_val_loader = DataLoader(
            l_val_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            collate_fn=collater
        )
        logger.info('Dataloader has created!')

    # Build model.
    logger.info('Building model and initializing!')

    model = model_selected(config).to(device)
    if args.resume:
        ckpt_model = convert_multi_gpu_checkpoint_to_single_gpu(ckpt)
        model.load_state_dict(ckpt_model)

    if args.train_model == 'pretrain':
        if args.data_name == 'unpair':
            # Can be visualized as pre-train.
            freeze_parameters(model.cross_at.parameters())

        elif args.data_name == 'pair':
            # Fine-tuning.
            unfreeze_parameters(model.cross_at.parameters())

            # Freezed
            freeze_parameters(model.aa_encoder.parameters())
            freeze_parameters(model.side_encoder.parameters())
            freeze_parameters(model.pos_encoder.parameters())
            freeze_parameters(model.decoder.parameters())
            freeze_parameters(model.dual_conv_block.parameters())
            freeze_parameters(model.last_norm.parameters())
    elif args.train_model == 'not_pretrain':
        print(f'Do not need to freeze any parameters! train model:{args.train_model}')
    else:
        raise AttributeError('Train model not exists.')

    # Build optimizer and scheduler.
    optimizer = optimizer_selected(config, model)
    scheduler = scheduler_selected(config, optimizer)

    # Config the type of loss.
    cross_loss = OasMaskedCrossEntropyLoss()
    mask_acc_loss = MaskedAccuracy()       # Do not be considered during backward, only make sure the correction of mask.

    if args.resume:
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        """Do not use the ckpt optimizer, because other layer has freezed."""
        it_sum = ckpt['iteration']
        logger.info('The re iteration start from {}'.format(it_sum))

    logger.info(f'# trainable parameters: {count_parameters(model) / 1e6:.4f} M')
    logger.info('Training...')
    best_val_loss = torch.inf
    best_iter = 0
    for it in range(0, config.train.max_iter+1):
        train(it, train_type=args.data_name)
        if it % config.train.valid_step == 0 or it == config.train.max_iter:
            valid_loss = valid(it, valid_type=args.data_name)
            # valid_loss = torch.inf
            if valid_loss < best_val_loss:
                best_val_loss, best_iter = valid_loss, it
                logger.info(f'Bset validate loss achieved: {best_val_loss:.6f}')
                ckpt_path = os.path.join(ckpt_dir, '%d.pt'%it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
            else:
                logger.info(f'[Validate] Val loss is not improved. '
                            f'Best val loss: {best_val_loss:.6f} at iter {best_iter}')

