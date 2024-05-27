import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot, softmax
from torch.nn.functional import cross_entropy
from sklearn.metrics import roc_auc_score

from .tokenizer import Tokenizer


class MaskedAccuracy(object):
    """Masked accuracy.

    Inputs:
        pred (N, L, C)
        tgt (N, L)
        mask (N, L)
    """

    def __call__(self, pred, tgt, mask):
        _, p = torch.max(pred, -1)
        masked_tgt = torch.masked_select(tgt, mask.bool())
        p = torch.masked_select(p, mask.bool())

        # multi_p = torch.masked_select(pred, mask.bool())
        # r_p = multi_p.detach().cpu().numpy()
        # r_tgt = masked_tgt.detach().cpu().numpy()
        #
        # roc_auc = roc_auc_score(r_tgt, r_p)

        # multi_logits = pred[mask][:, :22]
        # multi_p = softmax(multi_logits, dim=-1).clone().detach().cpu().numpy()
        # multi_t = one_hot(masked_tgt, num_classes=22).cpu().numpy()
        # roc_auc = roc_auc_score(multi_t, multi_p,  multi_class="ovr")

        return torch.mean((p == masked_tgt).float()), 0


class OasMaskedCrossEntropyLoss(CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.
    Evaluates the cross-entropy loss at specified locations in a sequence
    When reweight = True, reweights CE according to Hoogeboom et al.;
    reweight term = 1/(D-t+1)
    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - timestep (N, L) output from OAMaskCollater
            - input mask (N, L)
            - weight: (C, ): class weights for nn.CrossEntropyLoss

    Returns
        ce_losses
        nll_losses
    """
    def __init__(self, weight=None, reduction='none', reweight=True, tokenizer=Tokenizer()):
        self.reweight=reweight
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, H_L_pred, H_L_tgt, H_L_mask, H_L_cdr_mask, H_L_timesteps):
        # Make sure we have that empty last dimension
        if len(H_L_mask.shape) == len(H_L_pred.shape) - 1:
            H_L_mask = H_L_mask.unsqueeze(-1)



        # Make sure mask is boolean
        H_L_mask = H_L_mask.bool()

        H_L_mask_tokens = H_L_mask.sum()  # masked tokens
        H_L_cdr_mask_tokens = H_L_cdr_mask.sum().int()
        # H_nonpad_tokens, L_nonpad_tokens = H_input_mask.sum(dim=1), L_input_mask.sum(dim=1) # nonpad tokens

        # Cal H & L loss.
        H_L_p = torch.masked_select(H_L_pred, H_L_mask).view(H_L_mask_tokens, -1)  # [T x K] predictions for each mask char
        H_L_t = torch.masked_select(H_L_tgt, H_L_mask.squeeze())  # [ T ] true mask char
        H_L_loss = super().forward(H_L_p, H_L_t.long()) # [ T ] loss per mask char
        H_L_nll_losses = H_L_loss.mean()

        # Cal H & L cdr loss.
        H_L_cdr_mask = H_L_cdr_mask.bool()
        H_L_cdr_p = torch.masked_select(H_L_pred, H_L_cdr_mask.unsqueeze(-1)).view(H_L_cdr_mask_tokens, -1)
        H_L_cdr_t = torch.masked_select(H_L_tgt, H_L_cdr_mask)
        H_L_cdr_loss = super().forward(H_L_cdr_p, H_L_cdr_t.long())
        H_L_cdr_losses = H_L_cdr_loss.mean()

        if self.reweight: # Uses Hoogeboom OARDM reweighting term
            H_L_rwt_term = 1. / H_L_timesteps

            no_pad_number = torch.tensor([H_L_pred.size(1)]).repeat(H_L_pred.size(0)).to(H_L_rwt_term.device)
            H_L_rwt_term = H_L_rwt_term.repeat_interleave(H_L_timesteps)
            H_L_n_tokens = no_pad_number.repeat_interleave(H_L_timesteps)
            H_L_ce_loss = H_L_n_tokens * H_L_rwt_term * H_L_loss
            H_L_ce_losses = H_L_ce_loss.mean()  # reduce mean

        else:
            H_L_ce_losses = H_L_nll_losses
        return H_L_ce_losses, H_L_nll_losses.to(torch.float64), H_L_cdr_losses


class OasMaskedSplitCrossEntropyLoss(CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.
    Evaluates the cross-entropy loss at specified locations in a sequence
    When reweight = True, reweights CE according to Hoogeboom et al.;
    reweight term = 1/(D-t+1)
    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - timestep (N, L) output from OAMaskCollater
            - input mask (N, L)
            - weight: (C, ): class weights for nn.CrossEntropyLoss

    Returns
        ce_losses
        nll_losses
    """
    def __init__(self, weight=None, reduction='none', reweight=True, tokenizer=Tokenizer()):
        self.reweight=reweight
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, H_L_pred, H_L_tgt, H_L_mask, H_L_cdr_mask, H_L_timesteps):
        # Make sure we have that empty last dimension
        if len(H_L_mask.shape) == len(H_L_pred.shape) - 1:
            H_L_mask = H_L_mask.unsqueeze(-1)



        # Make sure mask is boolean
        H_L_mask = H_L_mask.bool()

        H_L_mask_tokens = H_L_mask.sum()  # masked tokens
        H_L_cdr_mask_tokens = H_L_cdr_mask.sum().int()
        # H_nonpad_tokens, L_nonpad_tokens = H_input_mask.sum(dim=1), L_input_mask.sum(dim=1) # nonpad tokens

        # Cal H & L loss.
        H_L_p = torch.masked_select(H_L_pred, H_L_mask).view(H_L_mask_tokens, -1)  # [T x K] predictions for each mask char
        H_L_t = torch.masked_select(H_L_tgt, H_L_mask.squeeze())  # [ T ] true mask char
        H_L_loss = super().forward(H_L_p, H_L_t.long()) # [ T ] loss per mask char
        H_L_nll_losses = H_L_loss.mean()

        # Cal H & L cdr loss.
        H_L_cdr_mask = H_L_cdr_mask.bool()
        H_L_cdr_p = torch.masked_select(H_L_pred, H_L_cdr_mask.unsqueeze(-1)).view(H_L_cdr_mask_tokens, -1)
        H_L_cdr_t = torch.masked_select(H_L_tgt, H_L_cdr_mask)
        H_L_cdr_loss = super().forward(H_L_cdr_p, H_L_cdr_t.long())
        H_L_cdr_losses = H_L_cdr_loss.mean()

        if self.reweight: # Uses Hoogeboom OARDM reweighting term
            H_L_rwt_term = 1. / H_L_timesteps

            no_pad_number = torch.tensor([H_L_pred.size(1)]).repeat(H_L_pred.size(0)).to(H_L_rwt_term.device)
            H_L_rwt_term = H_L_rwt_term.repeat_interleave(H_L_timesteps)
            H_L_n_tokens = no_pad_number.repeat_interleave(H_L_timesteps)
            H_L_ce_loss = H_L_n_tokens * H_L_rwt_term * H_L_loss
            H_L_ce_losses = H_L_ce_loss.mean()  # reduce mean

        else:
            H_L_ce_losses = H_L_nll_losses
        return H_L_ce_losses, H_L_nll_losses.to(torch.float64), H_L_cdr_losses