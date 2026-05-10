import torch
import torch.nn as nn


class DecentralizedAgentMixin:
    """Expects ``dtype``, ``perturb`` on the host class."""

    def is_commit(self, commitment):
        w = torch.tensor([0.0, 1.0], dtype=self.dtype, device=commitment.device).unsqueeze(1)
        return torch.matmul(commitment, w) ** 4

    def apply_grad_clamped(self, loss, module: nn.Module, optimizer, retain_graph=False):
        optimizer.zero_grad()
        params = list(module.parameters())
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph)
        for p, g in zip(params, grads):
            p.grad = g
            p.grad.data.clamp_(-1, 1)
        optimizer.step()

    def policy_entropy(self, probs):
        return -torch.mean(probs * torch.log(probs + self.perturb))

    def mixed_joint_actions(self, self_proposal, coplayer_proposal, self_action, coplayer_action, is_mutual_commitment):
        """Blend proposals vs unconstrained actions when both sides mutually commit."""
        exp = is_mutual_commitment.expand(-1, self_proposal.shape[1])
        real_self = exp * self_proposal + (1 - exp) * self_action
        real_cp = exp * coplayer_proposal + (1 - exp) * coplayer_action
        return real_self, real_cp
