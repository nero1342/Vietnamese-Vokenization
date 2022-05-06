from torchmetrics import Metric
import torch 
import torch 
def batchwise_recall(lang_output, visn_output, recall = 1):
    """
    Calculate the accuracy of contextual word retrieval, average by batch.
    :param lang_output: [batch_size, max_len, hid_dim]
    :param visn_output: [batch_size, hid_dim]
    :param lang_mask: Int Tensor [batch_size, max_len], 1 for tokens, 0 for paddings.
    :param recall: a list, which are the number of recalls to be evaluated.
    :return:
    """
    batch_size, lang_len, dim = lang_output.shape
    assert batch_size % 2 == 0 and batch_size == visn_output.shape[0]

    # Expand the visn_output to match each word
    visn_output = visn_output.unsqueeze(1)                  # [b, 1, dim]

    # The score of positive pairs
    positive_score = (lang_output * visn_output).sum(-1)    # [b, max_len]

    # The score of negative pairs. Note that the diagonal is actually the positive score,
    # but it would be zero-graded in calculating the loss below.
    negative_scores = (lang_output.reshape(batch_size, 1, lang_len, dim) *
                       visn_output.reshape(1, batch_size, 1, dim)).sum(-1)    # [b(lang), b(visn), max_len]
    # negative_scores = torch.einsum('ikd,jd->ijk', lang_output, visn_output)

    kthscore, kthidx = torch.kthvalue(negative_scores, batch_size - recall, dim=1)     # [b, max_len]
    # print(kthscore.shape) print(positive_score.shape)
    correct = (positive_score >= kthscore)                                # [b, max_len]
    correct_num = correct.sum()
    
    return (correct_num * 1. / batch_size / lang_len).item()

class AverageBatchwiseRecall(Metric):
    def __init__(self, recall = 1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.recall = recall 
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, embeddings):
        visn_output, lang_output = embeddings
        self.correct += batchwise_recall(lang_output, visn_output, self.recall)
        self.total += 1

    def compute(self):
        return self.correct / self.total
