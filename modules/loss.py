import torch
import torch.nn as nn


class SupervisedLearningLoss(nn.Module):

    def __init__(self):
        super(SupervisedLearningLoss, self).__init__()

    def forward(self, logits, target_labels, mask):

        # Truncate the target labels and mask to the same size as the input logits
        target_labels = target_labels[:, :logits.size(1)]
        mask = mask[:, :logits.size(1)]
        gathered_logits = -logits.gather(2, target_labels.long().unsqueeze(2)).squeeze(2) # Gather the logits corresponding to the target labels
        masked_logits = gathered_logits * mask # Apply the mask to the gathered logits
        sum_masked_logits = torch.sum(masked_logits)# Compute the sum of the masked logits
        sum_mask = torch.sum(mask) # Compute the sum of the mask
        epsilon = 0.0001 # Add a small epsilon to avoid division by zero
        sup_loss = sum_masked_logits / (sum_mask + epsilon)

        return sup_loss

def compute_loss(output_logits, target_reports_ids, target_reports_masks, criterion=None):

    if criterion is None:
        criterion = SupervisedLearningLoss()
    sup_loss = criterion(output_logits, target_reports_ids[:, 1:], target_reports_masks[:, 1:]).mean() # Compute the loss using the criterion
    return sup_loss


# import torch
# import torch.nn as nn
#
#
# class SupervisedLearningLoss(nn.Module):
#     def __init__(self, token_weights=None, pad_idx=0):
#         super().__init__()
#         self.pad_idx = pad_idx
#
#         if token_weights is not None:
#             max_token = max(token_weights.keys()) + 1
#             weights_tensor = torch.ones(max_token)
#             for tid, w in token_weights.items():
#                 weights_tensor[tid] = w
#             self.register_buffer('token_weights', weights_tensor)
#         else:
#             self.token_weights = None
#
#     def forward(self, logits, target_labels, mask):
#         target_labels = target_labels[:, :logits.size(1)]
#         mask = mask[:, :logits.size(1)]
#
#         gathered_logits = -logits.gather(2, target_labels.long().unsqueeze(2)).squeeze(2)
#
#         # Token-level weighting (MIMIC only)
#         if self.token_weights is not None:
#             B, T = target_labels.shape
#             flat_targets = target_labels.view(-1)
#             valid_mask = flat_targets < self.token_weights.size(0)
#             token_w = torch.ones_like(flat_targets, dtype=torch.float)
#             token_w[valid_mask] = self.token_weights[flat_targets[valid_mask].long()]
#             token_w = token_w.view(B, T).to(logits.device)
#             mask = mask * token_w
#
#         masked_logits = gathered_logits * mask
#         sum_masked_logits = torch.sum(masked_logits)
#         sum_mask = torch.sum(mask)
#         epsilon = 0.0001
#         sup_loss = sum_masked_logits / (sum_mask + epsilon)
#
#         return sup_loss


# def compute_loss(output_logits, target_reports_ids, target_reports_masks,
#                  criterion=None, args=None):
#     if criterion is None:
#         token_weights = None
#         if args is not None and getattr(args, 'token_weighting', False):
#             token_weights = getattr(args, 'precomputed_token_weights', None)
#
#         criterion = SupervisedLearningLoss(
#             token_weights=token_weights,
#             pad_idx=getattr(args, 'pad_idx', 0)
#         )
#
#     sup_loss = criterion(output_logits, target_reports_ids[:, 1:],
#                          target_reports_masks[:, 1:]).mean()
#     return sup_loss