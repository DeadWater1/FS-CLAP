import torch
from torch import nn
import torch.nn.functional as F





class AudioTextContrastiveLoss_HN(nn.Module):

    def __init__(self):
        super().__init__()

    def reweight(self, sim_mat, sim_targets, beta=0.15):
        # this beta is gamma in the paper
        beta_sim_mat = sim_mat * beta
        exp_beta_sim_mat = torch.exp(beta_sim_mat)
        bs = sim_mat.size(0)
        exp_beta_sim_mat_1 = (bs - 1) * exp_beta_sim_mat
        exp_beta_sim_mat_2 = torch.sum(exp_beta_sim_mat, 1) - exp_beta_sim_mat
        weights = exp_beta_sim_mat_1 / exp_beta_sim_mat_2
        weights[sim_targets > 0] = 1.0
        return weights


    def forward(self,
                sim_a2t,
                sim_t2a,
                sim_targets=None):
        if sim_targets is None:
            sim_targets = torch.zeros(sim_a2t.size()).to(
                sim_a2t.device
            )
            sim_targets.fill_diagonal_(1)

        logits_max_a2t, _ = torch.max(sim_a2t, dim=1, keepdim=True)
        sim_a2t = sim_a2t - logits_max_a2t.detach()
        weights_a2t = self.reweight(sim_a2t, sim_targets)

        logits_max_t2a, _ = torch.max(sim_t2a, dim=1, keepdim=True)
        sim_t2a = sim_t2a - logits_max_t2a.detach()
        weights_t2a = self.reweight(sim_t2a, sim_targets)

        exp_logits_a2t = torch.exp(sim_a2t)
        log_prob_a2t = sim_a2t - torch.log(1e-7 + (weights_a2t * exp_logits_a2t).sum(1, keepdim=True))

        exp_logits_t2a = torch.exp(sim_t2a)
        log_prob_t2a = sim_t2a - torch.log(1e-7 + (weights_t2a * exp_logits_t2a).sum(1, keepdim=True))

        loss_a2t = -(sim_targets * log_prob_a2t).sum(1).mean()
        loss_t2a = -(sim_targets * log_prob_t2a).sum(1).mean()

        loss_atc = (loss_a2t + loss_t2a) / 2
            
        return loss_atc
    


