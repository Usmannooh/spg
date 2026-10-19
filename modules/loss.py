import torch
import torch.nn as nn


class SupervisedLearningLoss(nn.Module):

    def __init__(self):
        super(SupervisedLearningLoss, self).__init__()

    def forward(self, logits, target_labels, mask):

        # Truncate the target labels and mask to the same size as the input logits
        target_labels = target_labels[:, :logits.size(1)]
        mask = mask[:, :logits.size(1)]
        gathered_logits = -logits.gather(2, target_labels.long().unsqueeze(2)).squeeze(2) 
        masked_logits = gathered_logits * mask # Apply the mask to the gathered logits
        sum_masked_logits = torch.sum(masked_logits)# Compute the sum of the masked logits
        sum_mask = torch.sum(mask) # Compute the sum of the mask
        epsilon = 0.0001 # Add a small epsilon to avoid division by zero
        sup_loss = sum_masked_logits / (sum_mask + epsilon)

        return sup_loss

def compute_loss(output_logits, target_reports_ids, target_reports_masks, criterion=None):

    if criterion is None:
        criterion = SupervisedLearningLoss()
    sup_loss = criterion(output_logits, target_reports_ids[:, 1:], target_reports_masks[:, 1:]).mean() 
    return sup_loss



#     return sup_loss
