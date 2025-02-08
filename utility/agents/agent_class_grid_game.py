import torch
import torch.nn.functional as F
import numpy as np

class ProposalActorNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim):
        super(ProposalActorNet, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layer2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.outputlayer = torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.ReLU()(x)
        x = self.layer2(x)
        x = torch.nn.ReLU()(x)
        x = self.outputlayer(x)
        x = torch.nn.Softmax(dim=-1)(x)
        return x
    
class CommitActorNet(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super(CommitActorNet, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layer2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.outputlayer = torch.nn.Linear(in_features=hidden_dim, out_features=2)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.ReLU()(x)
        x = self.layer2(x)
        x = torch.nn.ReLU()(x)
        x = self.outputlayer(x)
        x = torch.nn.Softmax(dim=-1)(x)
        return x
    
class UnconstrainedActorNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim):
        super(UnconstrainedActorNet, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layer2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.outputlayer = torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.ReLU()(x)
        x = self.layer2(x)
        x = torch.nn.ReLU()(x)
        x = self.outputlayer(x)
        x = torch.nn.Softmax(dim=-1)(x)
        return x

class CriticNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layer2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.outputlayer =  torch.nn.Linear(in_features=hidden_dim, out_features=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.ReLU()(x)
        x = self.layer2(x)
        x = torch.nn.ReLU()(x)
        x = self.outputlayer(x)
        return x


class DCL_Agent_Grid_Game():
    def __init__(self, temperature, hidden_dim, lr_critic, lr_actor, with_constraints, gamma, is_entropy, temperature_decay, action_dim, num_agents, grid_size=3, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.with_constraints = with_constraints
        self.is_entropy = is_entropy
        self.temperature_decay = temperature_decay
        self.action_dim = action_dim
        self.state_dim = grid_size*num_agents
        self.num_agents = num_agents        
        self.grid_size = grid_size
        self.temperature = temperature
        self.dtype = torch.float32

        # Actor net
        self.proposing_actor = ProposalActorNet(input_dim=self.state_dim, output_dim=action_dim, hidden_dim=hidden_dim).to(device)
        self.commit_actor = CommitActorNet(input_dim=self.state_dim+action_dim*num_agents,hidden_dim=hidden_dim).to(device)
        self.unconstrained_actor = UnconstrainedActorNet(input_dim=self.state_dim, output_dim=action_dim,hidden_dim=hidden_dim).to(device)
        # Critic net
        self.critic = CriticNet(input_dim=self.state_dim+action_dim*num_agents,hidden_dim=hidden_dim).to(device)

        # Initialize optimizers
        self.proposing_actor_optimizer = torch.optim.Adam(self.proposing_actor.parameters(), lr=lr_actor)
        self.commit_actor_optimizer = torch.optim.Adam(self.commit_actor.parameters(), lr=lr_actor)
        self.unconstrained_actor_optimizer = torch.optim.Adam(self.unconstrained_actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def build_connection(self, co_player):
        self.co_player = co_player

    def get_proposal(self, state, explore=False):
        proposal_probs = self.proposing_actor(state)
        proposal_logits = torch.log(proposal_probs+1e-8)
        if explore:
            proposal = F.gumbel_softmax(proposal_logits, hard=True, tau=self.temperature)
        else:
            proposal = F.one_hot(torch.argmax(proposal_logits,dim=1),num_classes=proposal_logits.size(-1))
        return proposal
    
    def get_commitment(self, state, self_proposal, coplayer_proposal, explore=False):        
        commitment_probs = self.commit_actor(torch.cat((state, self_proposal, coplayer_proposal),dim=1))
        commitment_logits = torch.log(commitment_probs+1e-8)
        if explore:
            commitment = F.gumbel_softmax(commitment_logits, hard=True, tau=self.temperature)
        else:
            commitment = F.one_hot(torch.argmax(commitment_logits,dim=1), num_classes=commitment_logits.size(-1))
        return commitment
    
    def get_unconstrained_action(self, state, explore=False):
        action_probs = self.unconstrained_actor(state)
        action_logits = torch.log(action_probs+1e-8)
        if explore:
            action = F.gumbel_softmax(action_logits, hard=True, tau=self.temperature)
        else:
            action = F.one_hot(torch.argmax(action_logits,dim=1), num_classes=action_logits.size(-1))
        return action
    
    def is_commit(self, commitment):
        unsqueezed_tensor = torch.tensor([0,1],dtype=self.dtype).unsqueeze(1)
        return torch.matmul(commitment, unsqueezed_tensor)**4 # input: one-hot; output: [1] if commit, [0] if not commit

    def int_to_onehot(self, variable_int_list, k):
        variable_onehot = torch.zeros(len(variable_int_list), k, dtype=self.dtype)
        variable_onehot[range(len(variable_int_list)), variable_int_list] = 1
        return variable_onehot
    
    # def calculate_unconstrained_value(self, state, action_probs, coplayer_action_probs):
    #     v = torch.zeros(len(state), dtype=torch.float64, device=self.device)
    #     for a_i in range(2):
    #             for a_j in range(2):
    #                 action_self_onehot_i = self.int_to_onehot(variable_int_list=[a_i], k=2, device=self.device).expand(len(state),-1)
    #                 action_coplayer_onehot_i = self.int_to_onehot([a_j], k=2, device=self.device).expand(len(state),-1)
    #                 q_i = self.critic(torch.cat((state, action_self_onehot_i, action_coplayer_onehot_i),dim=1)).squeeze()
    #                 v += q_i * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
    #     return v.detach()
    
    # def calculate_state_value(self, state, proposal_probs, coplayer_proposal_probs, commitment_probs, coplayer_commitment_probs, action_probs, coplayer_action_probs):
    #     v = torch.zeros(len(state), dtype=torch.float64, device=self.device)
    #     for m_i in range(2):
    #         for m_j in range(2):
    #             proposal_self_onehot = self.int_to_onehot(variable_int_list=[m_i], k=2, device=self.device).expand(len(state),-1)
    #             proposal_coplayer_onehot = self.int_to_onehot([m_j], k=2, device=self.device).expand(len(state),-1)
    #             q_i = self.critic(torch.cat((state, proposal_self_onehot, proposal_coplayer_onehot),dim=1)).squeeze()
    #             v += q_i * proposal_probs[:,m_i] * coplayer_proposal_probs[:,m_j] * commitment_probs[:,1] * coplayer_commitment_probs[:,1]
    #     for a_i in range(2):
    #         for a_j in range(2):
    #             action_self_onehot = self.int_to_onehot([a_i], k=2, device=self.device).expand(len(state),-1)
    #             action_coplayer_onehot = self.int_to_onehot([a_j], k=2, device=self.device).expand(len(state),-1)
    #             q_i = self.critic(torch.cat((state, action_self_onehot, action_coplayer_onehot),dim=1)).squeeze()
    #             v += q_i * (1-commitment_probs[:,1] * coplayer_commitment_probs[:,1]) * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
    #     return v.detach()

    def update_unconstrained_policy(self, state, self_commitment, coplayer_commitment ,self_action, coplayer_action, entropy_coeff):
        """
        Update policy for each agent
        """
        is_mutual_commitment = (self.is_commit(self_commitment)*self.co_player.is_commit(coplayer_commitment)).detach().squeeze()
        self.unconstrained_actor_optimizer.zero_grad() # Zero the gradients
        self_policy_probs = self.unconstrained_actor(state)
        self_policy_logits = torch.log(self_policy_probs+1e-8)
        self_action_logit = self_policy_logits[torch.arange(self_policy_logits.shape[0]),self_action.argmax(dim=1)]
        q_a = self.critic(torch.cat((state, self_action, coplayer_action),dim=1)).detach().squeeze()
        loss_unconstrained_actor = (-q_a * self_action_logit * is_mutual_commitment).mean() 
        if self.is_entropy:
            entropy = -torch.mean(self_policy_probs * torch.log(self_policy_probs + 1e-8))
            loss_unconstrained_actor -= entropy_coeff * entropy

        unconstrained_policy_grads = torch.autograd.grad(loss_unconstrained_actor, list(self.unconstrained_actor.parameters())) # Compute the gradients
        unconstrained_policy_params = list(self.unconstrained_actor.parameters())
        for layer in range(len(unconstrained_policy_params)):
            unconstrained_policy_params[layer].grad = unconstrained_policy_grads[layer]
            unconstrained_policy_params[layer].grad.data.clamp_(-1, 1)
        self.unconstrained_actor_optimizer.step() # Perform an optimization step

    def update_commitment_policy(self, state, self_proposal, coplayer_proposal, self_commitment, coplayer_commitment, self_action, coplayer_action, entropy_coeff):
        """
        Update policy for each agent
        """
        # We need Gumbel-softmax sample for commitment, because we need to take derivative \partial commitment / \partial parameters
        self_commitment_probs = self.commit_actor(torch.cat((state, self_proposal, coplayer_proposal),dim=1))
        self_commitment_logits = torch.log(self_commitment_probs+1e-8)
        self_commitment = F.gumbel_softmax(self_commitment_logits, hard=True, tau=self.temperature)
        self_commitment_logit = self_commitment_logits[torch.arange(self_commitment_logits.shape[0]), self_commitment.clone().detach().argmax(dim=1)]
        self_is_commitment = self.is_commit(self_commitment).squeeze()
        coplayer_is_commitment = self.co_player.is_commit(coplayer_commitment).squeeze()
        is_mutual_commitment = (self_is_commitment*coplayer_is_commitment).detach()

        q_sa = self.critic(torch.cat((state, self_action, coplayer_action),dim=1)).detach().squeeze()
        q_sm = self.critic(torch.cat((state, self_proposal, coplayer_proposal),dim=1)).detach().squeeze()
        commitment_loss = (is_mutual_commitment * (-q_sm * self_commitment_logit + (q_sa-q_sm)*self_is_commitment)+(1-is_mutual_commitment) * (-q_sa * self_commitment_logit)).mean()
        if self.is_entropy:
            entropy = -torch.mean(self_commitment_probs * torch.log(self_commitment_probs + 1e-8))
            commitment_loss -= entropy_coeff * entropy

        self.commit_actor_optimizer.zero_grad() # Zero the gradients
        commitment_grads = torch.autograd.grad(commitment_loss, list(self.commit_actor.parameters()), retain_graph=True) # Compute the gradients
        commitment_params = list(self.commit_actor.parameters())
        for layer in range(len(commitment_params)):
            commitment_params[layer].grad = commitment_grads[layer]
            commitment_params[layer].grad.data.clamp_(-1, 1)
        self.commit_actor_optimizer.step() # Perform an optimization step

    def update_proposal_policy(self, state, entropy_coeff):
        # We need Gumbel-softmax sample for proposal and commitment, the derivatives should be retained in these samples.
        self_proposal_probs = self.proposing_actor(state)
        self_proposal_logits = torch.log(self_proposal_probs+1e-8)
        self_proposal = F.gumbel_softmax(self_proposal_logits, hard=True, tau=self.temperature)
        self_proposal_logit = self_proposal_logits[torch.arange(self_proposal_logits.shape[0]), self_proposal.argmax(dim=1)]
        coplayer_proposal_probs = self.co_player.proposing_actor(state)
        coplayer_proposal_logits = torch.log(coplayer_proposal_probs+1e-8)
        coplayer_proposal = F.gumbel_softmax(coplayer_proposal_logits, hard=True, tau=self.temperature)

        self_commitment_probs = self.commit_actor(torch.cat((state, self_proposal, coplayer_proposal),dim=1))
        self_commitment_logits = torch.log(self_commitment_probs+1e-8)
        self_commitment = F.gumbel_softmax(self_commitment_logits, hard=True, tau=self.temperature)
        self_is_commitment = self.is_commit(self_commitment).squeeze()
        self_commitment_logit = self_commitment_logits[torch.arange(self_commitment_logits.shape[0]), self_commitment.argmax(dim=1)]
        coplayer_commitment_probs = self.co_player.commit_actor(torch.cat((state, coplayer_proposal, self_proposal),dim=1))
        coplayer_commitment_logits = torch.log(coplayer_commitment_probs+1e-8)
        coplayer_commitment = F.gumbel_softmax(coplayer_commitment_logits, hard=True, tau=self.temperature)
        coplayer_commitment_logit = coplayer_commitment_logits[torch.arange(coplayer_commitment_logits.shape[0]), coplayer_commitment.argmax(dim=1)]
        coplayer_is_commitment = self.co_player.is_commit(coplayer_commitment).squeeze()
        is_mutual_commitment = (self_is_commitment*coplayer_is_commitment).detach()

        self_action_probs = self.unconstrained_actor(state)
        self_action_logits = torch.log(self_action_probs+1e-8)
        self_action = F.gumbel_softmax(self_action_logits, hard=True, tau=self.temperature).detach()
        coplayer_action_probs = self.co_player.unconstrained_actor(state)
        coplayer_action_logits = torch.log(coplayer_action_probs+1e-8)
        coplayer_action = F.gumbel_softmax(coplayer_action_logits, hard=True, tau=self.temperature).detach()

        q_sa = self.critic(torch.cat((state, self_action, coplayer_action),dim=1)).detach().squeeze()
        q_sm = self.critic(torch.cat((state, self_proposal, coplayer_proposal),dim=1)).detach().squeeze()
        q_sa_coplayer = self.co_player.critic(torch.cat((state, coplayer_action, self_action),dim=1)).detach().squeeze()
        q_sm_coplayer = self.co_player.critic(torch.cat((state, coplayer_proposal, self_proposal),dim=1)).detach().squeeze()

        self.proposing_actor_optimizer.zero_grad() # Zero the gradients
        # proposing gradients part
        proposal_loss = is_mutual_commitment * (-q_sm * (self_proposal_logit + self_commitment_logit + coplayer_commitment_logit)) + (q_sa-q_sm)*(self_is_commitment+coplayer_is_commitment) + (1-is_mutual_commitment) * (-q_sa * (self_proposal_logit + self_commitment_logit+ coplayer_commitment_logit))
        if self.with_constraints==True:
            proposal_loss += torch.abs((q_sa-q_sm))*torch.maximum((q_sa-q_sm),torch.tensor(0.0))*self_proposal_logit
            proposal_loss += torch.abs((q_sa_coplayer-q_sm_coplayer))*torch.maximum((q_sa_coplayer-q_sm_coplayer),torch.tensor(0.0))*self_proposal_logit
        if self.is_entropy: # Get entropy of the proposal network
            entropy = -torch.mean(self_proposal_probs * torch.log(self_proposal_probs + 1e-8))
            proposal_loss = proposal_loss.mean()- entropy_coeff * entropy
        else:
            proposal_loss = proposal_loss.mean()
        
        proposal_grads = torch.autograd.grad(proposal_loss, list(self.proposing_actor.parameters()),retain_graph=True) # Compute the gradients
        proposal_params = list(self.proposing_actor.parameters())
        for layer in range(len(proposal_params)):
            proposal_params[layer].grad = proposal_grads[layer]
            proposal_params[layer].grad.data.clamp_(-1, 1)
        self.proposing_actor_optimizer.step() # Perform an optimization step
        self.temperature = np.maximum(1.0, self.temperature - self.temperature_decay)

    def update_critic(self, state, self_proposal, coplayer_proposal, self_action, coplayer_action, is_mutual_commitment, self_return):
        """
        Update critic Q^i(s,a^i,a^j)
        """
        # Compute the actual value
        is_mutual_commitment_expanded = is_mutual_commitment.expand(-1, self_proposal.shape[1])
        real_action = is_mutual_commitment_expanded * self_proposal + (1-is_mutual_commitment_expanded) * self_action
        real_coplayer_action = is_mutual_commitment_expanded * coplayer_proposal + (1-is_mutual_commitment_expanded) * coplayer_action
        actual_value = self.critic(torch.cat((state, real_action, real_coplayer_action),dim=1)).squeeze()
        target_value = self_return.squeeze()

        loss_func = torch.nn.MSELoss()
        critic_loss = loss_func(actual_value, target_value)
        self.critic_optimizer.zero_grad()  #  Zero the gradients       
        critic_grads = torch.autograd.grad(critic_loss, list(self.critic.parameters())) # Compute the gradients
        critic_params = list(self.critic.parameters())
        for layer in range(len(critic_params)):
            critic_params[layer].grad = critic_grads[layer]
            critic_params[layer].grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()