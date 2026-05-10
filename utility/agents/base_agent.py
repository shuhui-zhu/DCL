"""Unified DCL agent shared by repeated games and grid games.

Game-specific subclasses live in ``agent_class_repeated_games.py`` and
``agent_class_grid_game.py``; everything algorithmic lives here.
"""
import numpy as np
import torch
import torch.nn.functional as F

from utility.agents.nets import CriticNet, SoftmaxNet

EPS = 1e-8


class DCL_Agent:
    def __init__(
        self,
        *,
        state_dim,
        action_dim,
        num_agents,
        temperature,
        temperature_decay,
        hidden_dim,
        lr_critic,
        lr_actor,
        with_constraints,
        gamma,
        is_entropy,
        mega_step=1,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.with_constraints = with_constraints
        self.is_entropy = is_entropy
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.mega_step = mega_step
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.dtype = torch.float32

        proposal_dim = action_dim * mega_step
        self.proposal_dim = proposal_dim
        joint_input_dim = state_dim + proposal_dim * num_agents

        self.proposing_actor = SoftmaxNet(state_dim, proposal_dim, hidden_dim).to(device)
        self.commit_actor = SoftmaxNet(joint_input_dim, 2, hidden_dim).to(device)
        self.unconstrained_actor = SoftmaxNet(state_dim, proposal_dim, hidden_dim).to(device)
        self.critic = CriticNet(joint_input_dim, hidden_dim).to(device)

        self.proposing_actor_optimizer = torch.optim.Adam(self.proposing_actor.parameters(), lr=lr_actor)
        self.commit_actor_optimizer = torch.optim.Adam(self.commit_actor.parameters(), lr=lr_actor)
        self.unconstrained_actor_optimizer = torch.optim.Adam(self.unconstrained_actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def build_connection(self, co_player):
        self.co_player = co_player

    # ------------------------------------------------------------------ helpers
    def _sample_categorical(self, probs, explore):
        logits = torch.log(probs + EPS)
        if explore:
            return F.gumbel_softmax(logits, hard=True, tau=self.temperature)
        return F.one_hot(torch.argmax(logits, dim=1), num_classes=logits.size(-1))

    @staticmethod
    def _entropy(probs):
        return -torch.mean(probs * torch.log(probs + EPS))

    @staticmethod
    def _selected_logit(probs, onehot):
        logits = torch.log(probs + EPS)
        return logits, logits[torch.arange(logits.shape[0]), onehot.argmax(dim=1)]

    @staticmethod
    def _apply_grads(loss, params, optimizer, retain_graph=False):
        optimizer.zero_grad()
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph)
        for p, g in zip(params, grads):
            p.grad = g
            p.grad.data.clamp_(-1, 1)
        optimizer.step()

    def is_commit(self, commitment):
        # one-hot in -> [1] if commit (idx 1), [0] otherwise.
        # The **4 keeps the original behavior (hard 0/1 with smooth gradient through Gumbel-softmax).
        weights = torch.tensor([0, 1], dtype=self.dtype).unsqueeze(1)
        return torch.matmul(commitment, weights) ** 4

    def int_to_onehot(self, integer):
        onehot = torch.zeros(1, self.action_dim, dtype=self.dtype)
        onehot[0][integer] = 1
        return onehot

    # -------------------------------------------------------------- act methods
    def get_proposal(self, state, explore=False):
        return self._sample_categorical(self.proposing_actor(state), explore)

    def get_commitment(self, state, self_proposal, coplayer_proposal, explore=False):
        x = torch.cat((state, self_proposal, coplayer_proposal), dim=1)
        return self._sample_categorical(self.commit_actor(x), explore)

    def get_unconstrained_action(self, state, explore=False):
        return self._sample_categorical(self.unconstrained_actor(state), explore)

    # ------------------------------------------------------------ learn methods
    def update_unconstrained_policy(self, state, self_commitment, coplayer_commitment,
                                    self_action, coplayer_action, entropy_coeff):
        is_mutual = (self.is_commit(self_commitment) * self.co_player.is_commit(coplayer_commitment)).detach().squeeze()

        probs = self.unconstrained_actor(state)
        _, action_logit = self._selected_logit(probs, self_action)
        q_a = self.critic(torch.cat((state, self_action, coplayer_action), dim=1)).detach().squeeze()

        loss = (-q_a * action_logit * is_mutual).mean()
        if self.is_entropy:
            loss = loss - entropy_coeff * self._entropy(probs)

        self._apply_grads(loss, list(self.unconstrained_actor.parameters()),
                          self.unconstrained_actor_optimizer)

    def update_commitment_policy(self, state, self_proposal, coplayer_proposal,
                                 self_commitment, coplayer_commitment,
                                 self_action, coplayer_action, entropy_coeff):
        # Resample commitment with Gumbel-softmax so we can backprop through it.
        probs = self.commit_actor(torch.cat((state, self_proposal, coplayer_proposal), dim=1))
        logits = torch.log(probs + EPS)
        sampled = F.gumbel_softmax(logits, hard=True, tau=self.temperature)
        sampled_logit = logits[torch.arange(logits.shape[0]), sampled.clone().detach().argmax(dim=1)]

        self_is_commit = self.is_commit(sampled).squeeze()
        coplayer_is_commit = self.co_player.is_commit(coplayer_commitment).squeeze()
        is_mutual = (self_is_commit * coplayer_is_commit).detach()

        q_sa = self.critic(torch.cat((state, self_action, coplayer_action), dim=1)).detach().squeeze()
        q_sm = self.critic(torch.cat((state, self_proposal, coplayer_proposal), dim=1)).detach().squeeze()

        loss = (
            is_mutual * (-q_sm * sampled_logit + (q_sa - q_sm) * self_is_commit)
            + (1 - is_mutual) * (-q_sa * sampled_logit)
        ).mean()
        if self.is_entropy:
            loss = loss - entropy_coeff * self._entropy(probs)

        self._apply_grads(loss, list(self.commit_actor.parameters()),
                          self.commit_actor_optimizer, retain_graph=True)

    def update_proposal_policy(self, state, entropy_coeff):
        # Self proposal (with grad).
        sp_probs = self.proposing_actor(state)
        sp_logits = torch.log(sp_probs + EPS)
        sp = F.gumbel_softmax(sp_logits, hard=True, tau=self.temperature)
        sp_logit = sp_logits[torch.arange(sp_logits.shape[0]), sp.argmax(dim=1)]

        # Co-player proposal (no grad through their network here).
        cp_probs = self.co_player.proposing_actor(state)
        cp_logits = torch.log(cp_probs + EPS)
        cp = F.gumbel_softmax(cp_logits, hard=True, tau=self.temperature)

        # Self commitment given (sp, cp).
        sc_probs = self.commit_actor(torch.cat((state, sp, cp), dim=1))
        sc_logits = torch.log(sc_probs + EPS)
        sc = F.gumbel_softmax(sc_logits, hard=True, tau=self.temperature)
        sc_logit = sc_logits[torch.arange(sc_logits.shape[0]), sc.argmax(dim=1)]
        self_is_commit = self.is_commit(sc).squeeze()

        # Co-player commitment given (cp, sp).
        cc_probs = self.co_player.commit_actor(torch.cat((state, cp, sp), dim=1))
        cc_logits = torch.log(cc_probs + EPS)
        cc = F.gumbel_softmax(cc_logits, hard=True, tau=self.temperature)
        cc_logit = cc_logits[torch.arange(cc_logits.shape[0]), cc.argmax(dim=1)]
        coplayer_is_commit = self.co_player.is_commit(cc).squeeze()

        is_mutual = (self_is_commit * coplayer_is_commit).detach()

        # Sampled (unconstrained) actions, no grad.
        self_action = F.gumbel_softmax(
            torch.log(self.unconstrained_actor(state) + EPS),
            hard=True, tau=self.temperature,
        ).detach()
        coplayer_action = F.gumbel_softmax(
            torch.log(self.co_player.unconstrained_actor(state) + EPS),
            hard=True, tau=self.temperature,
        ).detach()

        q_sa = self.critic(torch.cat((state, self_action, coplayer_action), dim=1)).detach().squeeze()
        q_sm = self.critic(torch.cat((state, sp, cp), dim=1)).detach().squeeze()
        q_sa_cp = self.co_player.critic(torch.cat((state, coplayer_action, self_action), dim=1)).detach().squeeze()
        q_sm_cp = self.co_player.critic(torch.cat((state, cp, sp), dim=1)).detach().squeeze()

        loss = (
            is_mutual * (-q_sm * (sp_logit + sc_logit + cc_logit))
            + (q_sa - q_sm) * (self_is_commit + coplayer_is_commit)
            + (1 - is_mutual) * (-q_sa * (sp_logit + sc_logit + cc_logit))
        )

        if self.with_constraints:
            loss = loss + torch.abs(q_sa - q_sm) * torch.clamp(q_sa - q_sm, min=0.0) * sp_logit
            loss = loss + torch.abs(q_sa_cp - q_sm_cp) * torch.clamp(q_sa_cp - q_sm_cp, min=0.0) * sp_logit

        if self.is_entropy:
            loss = loss.mean() - entropy_coeff * self._entropy(sp_probs)
        else:
            loss = loss.mean()

        self._apply_grads(loss, list(self.proposing_actor.parameters()),
                          self.proposing_actor_optimizer, retain_graph=True)
        self.temperature = float(np.maximum(1.0, self.temperature - self.temperature_decay))

    def update_critic(self, state, self_proposal, coplayer_proposal,
                      self_action, coplayer_action, is_mutual_commitment, self_return):
        is_mutual_expanded = is_mutual_commitment.expand(-1, self_proposal.shape[1])
        real_action = is_mutual_expanded * self_proposal + (1 - is_mutual_expanded) * self_action
        real_cp_action = is_mutual_expanded * coplayer_proposal + (1 - is_mutual_expanded) * coplayer_action

        actual = self.critic(torch.cat((state, real_action, real_cp_action), dim=1)).squeeze()
        loss = F.mse_loss(actual, self_return.squeeze())

        self._apply_grads(loss, list(self.critic.parameters()), self.critic_optimizer)
