from torch import nn 
import torch 

def hinge(x):
    return torch.clamp(x, min=0.)

def paired_hinge_rank_loss(
        lang_output: torch.Tensor,
        visn_output: torch.Tensor,
        lang_mask: torch.Tensor,
        margin: float,
):
    """
    Consider the first half as positive and the second half as negative.
    :param lang_output: [batch_size, max_len, hid_dim]
    :param visn_output: [batch_size, hid_dim]
    :param lang_mask: Int Tensor [batch_size, max_len], 1 for tokens, 0 for paddings.
    :param margin: margin in the ranking loss
    :return: a scalar loss
    """
    
    batch_size, lang_len, dim = lang_output.shape
    assert batch_size % 2 == 0 and batch_size == visn_output.shape[0]
    assert margin > 0.

    # Expand the visn_output to match each word
    visn_output = visn_output.unsqueeze(1)      # [b, 1, hid_dim]

    # Split to positive and negative sets.
    half_batch_size = batch_size // 2
    pos_lang, neg_lang = torch.split(lang_output, half_batch_size, dim=0)
    pos_visn, neg_visn = torch.split(visn_output, half_batch_size, dim=0)

    # Calculate positive and negative scores.
    true_pos_score = (pos_lang * pos_visn).sum(-1)           # [batch_size / 2, max_len]
    true_neg_score = (neg_lang * neg_visn).sum(-1)           # [batch_size / 2, max_len]
    false_pos_score = (pos_lang * neg_visn).sum(-1)          # [batch_size / 2, max_len]
    false_neg_score = (neg_lang * pos_visn).sum(-1)          # [batch_size / 2, max_len]

    # Hinge Loss
    pos_loss = hinge(margin - true_pos_score + false_pos_score)
    neg_loss = hinge(margin - true_neg_score + false_neg_score)

    # Averaging
    loss = (pos_loss.sum() + neg_loss.sum()) / batch_size / lang_len

    return loss

class LossComputer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss_func = paired_hinge_rank_loss
    def forward(self, embeddings):
        v_e, l_e = embeddings

        return self.loss_func(l_e, v_e, l_e[1], 0.5)

