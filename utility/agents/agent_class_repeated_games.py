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


class DCL_Agent_Repeated_Games():
    def __init__(self, perturb, temperature, hidden_dim, lr_critic, lr_actor, with_constraints, gamma, is_entropy, temperature_decay, mega_step, state_dim, action_dim, num_agents,device="cpu"):
        self.device = device
        self.gamma = gamma
        self.with_constraints = with_constraints
        self.is_entropy = is_entropy
        self.temperature_decay = temperature_decay
        self.mega_step = mega_step
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_agents = num_agents

        # Actor net
        self.proposing_actor = ProposalActorNet(input_dim=state_dim, output_dim=action_dim*mega_step, hidden_dim=hidden_dim).to(device)
        self.commit_actor = CommitActorNet(input_dim=state_dim+mega_step*action_dim*num_agents,hidden_dim=hidden_dim).to(device)
        self.unconstrained_actor = UnconstrainedActorNet(input_dim=state_dim, output_dim=action_dim*mega_step,hidden_dim=hidden_dim).to(device)
        # Critic net
        self.critic = CriticNet(input_dim=state_dim+mega_step*action_dim*num_agents,hidden_dim=hidden_dim).to(device)
        # Initialize optimizers
        self.proposing_actor_optimizer = torch.optim.Adam(self.proposing_actor.parameters(), lr=lr_actor)
        self.commit_actor_optimizer = torch.optim.Adam(self.commit_actor.parameters(), lr=lr_actor)
        self.unconstrained_actor_optimizer = torch.optim.Adam(self.unconstrained_actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.coplayer_proposing_actor = ProposalActorNet(input_dim=state_dim, output_dim=action_dim*mega_step, hidden_dim=hidden_dim).to(device)
        self.coplayer_commit_actor = CommitActorNet(input_dim=state_dim+mega_step*action_dim*num_agents,hidden_dim=hidden_dim).to(device)
        self.coplayer_unconstrained_actor = UnconstrainedActorNet(input_dim=state_dim, output_dim=action_dim*mega_step,hidden_dim=hidden_dim).to(device)
        self.coplayer_critic = CriticNet(input_dim=state_dim+mega_step*action_dim*num_agents,hidden_dim=hidden_dim).to(device)
        self.coplayer_proposing_actor_optimizer = torch.optim.Adam(self.coplayer_proposing_actor.parameters(), lr=lr_actor)
        self.coplayer_commit_actor_optimizer = torch.optim.Adam(self.coplayer_commit_actor.parameters(), lr=lr_actor)
        self.coplayer_unconstrained_actor_optimizer = torch.optim.Adam(self.coplayer_unconstrained_actor.parameters(), lr=lr_actor)
        self.coplayer_critic_optimizer = torch.optim.Adam(self.coplayer_critic.parameters(), lr=lr_critic)

        self.temperature = temperature
        self.dtype = torch.float32
        self.perturb = perturb

    def int_to_onehot(self, integer):
        """
        Convert integer to one-hot tensor
        """
        onehot = torch.zeros(1,self.action_dim, dtype=torch.float32)
        onehot[0][integer] = 1
        return onehot

    def epsilon_greedy_probs(self, probs, epsilon=0.0):
        probs = (1 - epsilon) * probs + epsilon / probs.shape[1]
        return probs

    def get_proposal(self, state, epsilon, explore=False):
        proposal_probs = self.proposing_actor(state)
        proposal_probs = self.epsilon_greedy_probs(proposal_probs, epsilon)
        proposal_probs = (1 - epsilon) * proposal_probs + epsilon / proposal_probs.shape[1]
        proposal_logits = torch.log(proposal_probs+self.perturb)
        if explore:
            proposal = F.gumbel_softmax(proposal_logits, hard=True, tau=self.temperature)
        else:
            proposal = F.one_hot(torch.argmax(proposal_logits,dim=1),num_classes=proposal_logits.size(-1))
        return proposal.detach()
    
    def get_commitment(self, state, self_proposal, coplayer_proposal, epsilon, explore=False):        
        commitment_probs = self.commit_actor(torch.cat((state, self_proposal, coplayer_proposal),dim=1))
        commitment_probs = self.epsilon_greedy_probs(commitment_probs, epsilon)
        commitment_probs = (1 - epsilon) * commitment_probs + epsilon / commitment_probs.shape[1]
        commitment_logits = torch.log(commitment_probs+self.perturb)
        if explore:
            commitment = F.gumbel_softmax(commitment_logits, hard=True, tau=self.temperature)
        else:
            commitment = F.one_hot(torch.argmax(commitment_logits,dim=1), num_classes=commitment_logits.size(-1))
        return commitment.detach()
    
    def get_unconstrained_action(self, state, epsilon, explore=False):
        action_probs = self.unconstrained_actor(state)
        action_probs = self.epsilon_greedy_probs(action_probs, epsilon)
        action_probs = (1 - epsilon) * action_probs + epsilon / action_probs.shape[1]
        action_logits = torch.log(action_probs+self.perturb)
        if explore:
            action = F.gumbel_softmax(action_logits, hard=True, tau=self.temperature)
        else:
            action = F.one_hot(torch.argmax(action_logits,dim=1), num_classes=action_logits.size(-1))
        return action.detach()
    
    def is_commit(self, commitment):
        unsqueezed_tensor = torch.tensor([0,1],dtype=self.dtype).unsqueeze(1)
        return torch.matmul(commitment, unsqueezed_tensor)**4 # input: one-hot; output: [1] if commit, [0] if not commit

    def ints_to_mega_one_hot(self, ints):
        if self.mega_step == 2:
            # Map integers to one-hot vectors tensors
            if np.array_equal(ints, np.array([0, 0])):
                return torch.tensor(np.array([[1, 0, 0, 0]]), dtype=self.dtype)
            elif np.array_equal(ints, np.array([0, 1])):
                return torch.tensor(np.array([[0, 1, 0, 0]]), dtype=self.dtype)
            elif np.array_equal(ints, np.array([1, 0])):
                return torch.tensor(np.array([[0, 0, 1, 0]]), dtype=self.dtype)
            elif np.array_equal(ints, np.array([1, 1])):
                return torch.tensor(np.array([[0, 0, 0, 1]]), dtype=self.dtype)
            else:
                raise ValueError("Input integers are not valid for the given one-hot vector")
        elif self.mega_step == 1:
            if np.array_equal(ints, np.array([0])):
                return torch.tensor(np.array([[1, 0]]), dtype=self.dtype)
            elif np.array_equal(ints, np.array([1])):
                return torch.tensor(np.array([[0, 1]]), dtype=self.dtype)
            else:
                raise ValueError("Input integers are not valid for the given one-hot vector")

    def calculate_mega_unconstrained_value(self, state, action_probs, coplayer_action_probs):
        v = torch.zeros(len(state), dtype=self.dtype, device=self.device)
        action_list = [[0, 0],[0, 1],[1, 0],[1, 1]]
        for a_i in range(len(action_list)):
            for a_j in range(len(action_list)):
                action_self_onehot_i = self.ints_to_mega_one_hot(action_list[a_i]).expand(len(state),-1)
                action_coplayer_onehot_i = self.ints_to_mega_one_hot(action_list[a_j]).expand(len(state),-1)
                q_i = self.critic(torch.cat((state, action_self_onehot_i, action_coplayer_onehot_i),dim=1)).squeeze()
                v += q_i * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
        return v.detach()

    def calculate_mega_unconstrained_value_coplayer(self, state, coplayer_action_probs, action_probs):
        v = torch.zeros(len(state), dtype=self.dtype, device=self.device)
        action_list = [[0, 0],[0, 1],[1, 0],[1, 1]]
        for a_i in range(len(action_list)):
            for a_j in range(len(action_list)):
                action_self_onehot_i = self.ints_to_mega_one_hot(action_list[a_i]).expand(len(state),-1)
                action_coplayer_onehot_i = self.ints_to_mega_one_hot(action_list[a_j]).expand(len(state),-1)
                q_i = self.coplayer_critic(torch.cat((state, action_coplayer_onehot_i, action_self_onehot_i),dim=1)).squeeze()
                v += q_i * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
        return v.detach().squeeze()
    
    def calculate_mega_state_value(self, state, proposal_probs, coplayer_proposal_probs, commitment_probs, coplayer_commitment_probs, action_probs, coplayer_action_probs):
        v = torch.zeros(len(state), dtype=self.dtype, device=self.device)
        action_list = [[0, 0],[0, 1],[1, 0],[1, 1]]
        for m_i in range(len(action_list)):
            for m_j in range(len(action_list)):
                proposal_self_onehot = self.ints_to_mega_one_hot(action_list[m_i]).expand(len(state),-1)
                proposal_coplayer_onehot = self.ints_to_mega_one_hot(action_list[m_j]).expand(len(state),-1)
                q_i = self.critic(torch.cat((state, proposal_self_onehot, proposal_coplayer_onehot),dim=1)).squeeze()
                v += q_i * proposal_probs[:,m_i] * coplayer_proposal_probs[:,m_j] * commitment_probs[:,1] * coplayer_commitment_probs[:,1]
        for a_i in range(2):
            for a_j in range(2):
                action_self_onehot = self.ints_to_mega_one_hot(action_list[a_i]).expand(len(state),-1)
                action_coplayer_onehot = self.ints_to_mega_one_hot(action_list[a_j]).expand(len(state),-1)
                q_i = self.critic(torch.cat((state, action_self_onehot, action_coplayer_onehot),dim=1)).squeeze()
                v += q_i * (1-commitment_probs[:,1] * coplayer_commitment_probs[:,1]) * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
        return v.detach().squeeze()
    
    def calculate_mega_state_value_coplayer(self, state, coplayer_proposal_probs, proposal_probs, coplayer_commitment_probs, commitment_probs, coplayer_action_probs, action_probs):
        v = torch.zeros(len(state), dtype=self.dtype, device=self.device)
        action_list = [[0, 0],[0, 1],[1, 0],[1, 1]]
        for m_i in range(2):
            for m_j in range(2):
                proposal_self_onehot = self.ints_to_mega_one_hot(action_list[m_i]).expand(len(state),-1)
                proposal_coplayer_onehot = self.ints_to_mega_one_hot(action_list[m_j]).expand(len(state),-1)
                q_i = self.coplayer_critic(torch.cat((state, proposal_coplayer_onehot, proposal_self_onehot),dim=1)).squeeze()
                v += q_i * proposal_probs[:,m_i] * coplayer_proposal_probs[:,m_j] * commitment_probs[:,1] * coplayer_commitment_probs[:,1]
        for a_i in range(2):
            for a_j in range(2):
                action_self_onehot = self.ints_to_mega_one_hot(action_list[a_i]).expand(len(state),-1)
                action_coplayer_onehot = self.ints_to_mega_one_hot(action_list[a_j]).expand(len(state),-1)
                q_i = self.coplayer_critic(torch.cat((state, action_coplayer_onehot, action_self_onehot),dim=1)).squeeze()
                v += q_i * (1-commitment_probs[:,1] * coplayer_commitment_probs[:,1]) * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
        return v.detach().squeeze()
    
    def calculate_unconstrained_value(self, state, action_probs, coplayer_action_probs):
        v = torch.zeros(len(state), dtype=self.dtype, device=self.device)
        for a_i in range(2):
            for a_j in range(2):
                action_self_onehot_i = self.int_to_onehot(a_i).expand(len(state),-1)
                action_coplayer_onehot_i = self.int_to_onehot(a_j).expand(len(state),-1)
                q_i = self.critic(torch.cat((state, action_self_onehot_i, action_coplayer_onehot_i),dim=1)).squeeze()
                v += q_i * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
        return v.detach()
    
    def calculate_unconstrained_value_coplayer(self, state, coplayer_action_probs, action_probs):
        v = torch.zeros(len(state), dtype=self.dtype, device=self.device)
        for a_i in range(2):
                for a_j in range(2):
                    action_self_onehot_i = self.int_to_onehot(a_i).expand(len(state),-1)
                    action_coplayer_onehot_i = self.int_to_onehot(a_j).expand(len(state),-1)
                    q_i = self.coplayer_critic(torch.cat((state, action_coplayer_onehot_i, action_self_onehot_i),dim=1)).squeeze()
                    v += q_i * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
        return v.detach().squeeze()
    
    def calculate_state_value(self, state, proposal_probs, coplayer_proposal_probs, commitment_probs, coplayer_commitment_probs, action_probs, coplayer_action_probs):
        v = torch.zeros(len(state), dtype=self.dtype, device=self.device)
        for m_i in range(2):
            for m_j in range(2):
                proposal_self_onehot = self.int_to_onehot(m_i).expand(len(state),-1)
                proposal_coplayer_onehot = self.int_to_onehot(m_j).expand(len(state),-1)
                q_i = self.critic(torch.cat((state, proposal_self_onehot, proposal_coplayer_onehot),dim=1)).squeeze()
                v += q_i * proposal_probs[:,m_i] * coplayer_proposal_probs[:,m_j] * commitment_probs[:,1] * coplayer_commitment_probs[:,1]
        for a_i in range(2):
            for a_j in range(2):
                action_self_onehot = self.int_to_onehot(a_i).expand(len(state),-1)
                action_coplayer_onehot = self.int_to_onehot(a_j).expand(len(state),-1)
                q_i = self.critic(torch.cat((state, action_self_onehot, action_coplayer_onehot),dim=1)).squeeze()
                v += q_i * (1-commitment_probs[:,1] * coplayer_commitment_probs[:,1]) * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
        return v.detach().squeeze()
    
    def calculate_state_value_coplayer(self, state, coplayer_proposal_probs, proposal_probs, coplayer_commitment_probs, commitment_probs, coplayer_action_probs, action_probs):
        v = torch.zeros(len(state), dtype=self.dtype, device=self.device)
        for m_i in range(2):
            for m_j in range(2):
                proposal_self_onehot = self.int_to_onehot(m_i).expand(len(state),-1)
                proposal_coplayer_onehot = self.int_to_onehot(m_j).expand(len(state),-1)
                q_i = self.coplayer_critic(torch.cat((state, proposal_coplayer_onehot, proposal_self_onehot),dim=1)).squeeze()
                v += q_i * proposal_probs[:,m_i] * coplayer_proposal_probs[:,m_j] * commitment_probs[:,1] * coplayer_commitment_probs[:,1]
        for a_i in range(2):
            for a_j in range(2):
                action_self_onehot = self.int_to_onehot(a_i).expand(len(state),-1)
                action_coplayer_onehot = self.int_to_onehot(a_j).expand(len(state),-1)
                q_i = self.coplayer_critic(torch.cat((state, action_coplayer_onehot, action_self_onehot),dim=1)).squeeze()
                v += q_i * (1-commitment_probs[:,1] * coplayer_commitment_probs[:,1]) * action_probs[:,a_i] * coplayer_action_probs[:,a_j]
        return v.detach().squeeze()

    def update_unconstrained_policy(self, state, self_commitment, coplayer_commitment ,self_action, coplayer_action, entropy_coeff):
        """
        Update policy for each agent
        """
        is_mutual_commitment = (self.is_commit(self_commitment)*self.is_commit(coplayer_commitment)).detach().squeeze()
        self.unconstrained_actor_optimizer.zero_grad() # Zero the gradients
        self_policy_probs = self.unconstrained_actor(state)
        self_policy_logits = torch.log(self_policy_probs+self.perturb)
        self_action_logit = self_policy_logits[torch.arange(self_policy_logits.shape[0]),self_action.argmax(dim=1)]
        q_a = self.critic(torch.cat((state, self_action, coplayer_action),dim=1)).detach().squeeze()
        coplayer_action_probs = self.coplayer_unconstrained_actor(state).detach()
        advantage_a = q_a - self.calculate_mega_unconstrained_value(state, self_policy_probs, coplayer_action_probs)
        # loss_unconstrained_actor = (-q_a * self_action_logit * is_mutual_commitment).mean() 
        loss_unconstrained_actor = (-advantage_a * self_action_logit * is_mutual_commitment).mean()
        if self.is_entropy:
            entropy = -torch.mean(self_policy_probs * torch.log(self_policy_probs + self.perturb))
            # entropy = entropy/torch.log(torch.tensor(self.action_dim, dtype=self.dtype))
            loss_unconstrained_actor -= entropy_coeff * entropy

        unconstrained_policy_grads = torch.autograd.grad(loss_unconstrained_actor, list(self.unconstrained_actor.parameters())) # Compute the gradients
        unconstrained_policy_params = list(self.unconstrained_actor.parameters())
        for layer in range(len(unconstrained_policy_params)):
            unconstrained_policy_params[layer].grad = unconstrained_policy_grads[layer]
            unconstrained_policy_params[layer].grad.data.clamp_(-1, 1)
        # Perform an optimization step
        self.unconstrained_actor_optimizer.step()
        return 
    
    def update_coplayer_unconstrained_policy(self, state, self_commitment, coplayer_commitment ,self_action, coplayer_action, entropy_coeff):
        """
        update estimate for coplayer
        """
        is_mutual_commitment = (self.is_commit(self_commitment)*self.is_commit(coplayer_commitment)).detach().squeeze()
        self.coplayer_unconstrained_actor_optimizer.zero_grad() 
        coplayer_policy_probs = self.coplayer_unconstrained_actor(state)
        coplayer_policy_logits = torch.log(coplayer_policy_probs+self.perturb)
        coplayer_action_logit = coplayer_policy_logits[torch.arange(coplayer_policy_logits.shape[0]),coplayer_action.argmax(dim=1)]
        q_a_coplayer = self.coplayer_critic(torch.cat((state, coplayer_action, self_action),dim=1)).detach().squeeze()
        self_policy_probs = self.unconstrained_actor(state).detach()
        advantage_a_coplayer = q_a_coplayer - self.calculate_mega_unconstrained_value_coplayer(state, coplayer_policy_probs, self_policy_probs)
        # loss_unconstrained_actor_coplayer = (-q_a_coplayer * coplayer_action_logit * is_mutual_commitment).mean()
        loss_unconstrained_actor_coplayer = (-advantage_a_coplayer * coplayer_action_logit * is_mutual_commitment).mean()
        if self.is_entropy:
            entropy = -torch.mean(coplayer_policy_probs * torch.log(coplayer_policy_probs + self.perturb))
            # entropy = entropy/torch.log(torch.tensor(self.action_dim, dtype=self.dtype))
            loss_unconstrained_actor_coplayer -= entropy_coeff * entropy
        unconstrained_policy_grads_coplayer = torch.autograd.grad(loss_unconstrained_actor_coplayer, list(self.coplayer_unconstrained_actor.parameters()))
        unconstrained_policy_params_coplayer = list(self.coplayer_unconstrained_actor.parameters())
        for layer in range(len(unconstrained_policy_params_coplayer)):
            unconstrained_policy_params_coplayer[layer].grad = unconstrained_policy_grads_coplayer[layer]
            unconstrained_policy_params_coplayer[layer].grad.data.clamp_(-1, 1)
        self.coplayer_unconstrained_actor_optimizer.step()


    def update_commitment_policy(self, state, self_proposal, coplayer_proposal, self_commitment, coplayer_commitment, self_action, coplayer_action, entropy_coeff, epsilon):
        """
        Update policy for each agent
        """
        # We need Gumbel-softmax sample for commitment, because we need to take derivative \partial commitment / \partial parameters
        self_commitment_probs = self.commit_actor(torch.cat((state, self_proposal, coplayer_proposal),dim=1))
        self_commitment_probs = self.epsilon_greedy_probs(self_commitment_probs, epsilon)
        self_commitment_logits = torch.log(self_commitment_probs+self.perturb)
        self_commitment = F.gumbel_softmax(self_commitment_logits, hard=True, tau=self.temperature)
        self_commitment_logit = self_commitment_logits[torch.arange(self_commitment_logits.shape[0]), self_commitment.clone().detach().argmax(dim=1)]
        self_is_commitment = self.is_commit(self_commitment).squeeze()
        coplayer_is_commitment = self.is_commit(coplayer_commitment).squeeze()
        is_mutual_commitment = (self_is_commitment*coplayer_is_commitment).detach()

        q_sa = self.critic(torch.cat((state, self_action, coplayer_action),dim=1)).detach().squeeze()
        q_sm = self.critic(torch.cat((state, self_proposal, coplayer_proposal),dim=1)).detach().squeeze()
        advantage_sa = q_sa - self.calculate_mega_state_value(state, self_proposal, coplayer_proposal, self_commitment, coplayer_commitment, self_action, coplayer_action)
        advantage_sm = q_sm - self.calculate_mega_state_value(state, self_proposal, coplayer_proposal, self_commitment, coplayer_commitment, self_action, coplayer_action)

        # commitment_loss = (is_mutual_commitment * (-q_sm * self_commitment_logit + (q_sa-q_sm)*self_is_commitment)+(1-is_mutual_commitment) * (-q_sa * self_commitment_logit)).mean()
        commitment_loss = (is_mutual_commitment * (-advantage_sm * self_commitment_logit + (advantage_sa-advantage_sm)*self_is_commitment)+(1-is_mutual_commitment) * (-advantage_sa * self_commitment_logit)).mean()
        if self.is_entropy:
            entropy = -torch.mean(self_commitment_probs * torch.log(self_commitment_probs + self.perturb))
            # entropy = entropy/torch.log(torch.tensor(2, dtype=self.dtype))
            commitment_loss -= entropy_coeff * entropy

        self.commit_actor_optimizer.zero_grad() # Zero the gradients
        commitment_grads = torch.autograd.grad(commitment_loss, list(self.commit_actor.parameters()), retain_graph=True) # Compute the gradients
        commitment_params = list(self.commit_actor.parameters())
        for layer in range(len(commitment_params)):
            commitment_params[layer].grad = commitment_grads[layer]
            commitment_params[layer].grad.data.clamp_(-1, 1)
        self.commit_actor_optimizer.step() # Perform an optimization step
        return 
    
    def update_coplayer_commitment_policy(self, state, self_proposal, coplayer_proposal, self_commitment, coplayer_commitment, self_action, coplayer_action, entropy_coeff, epsilon):
        """
        update an estimate for coplayer
        """
        is_mutual_commitment = (self.is_commit(self_commitment)*self.is_commit(coplayer_commitment)).detach().squeeze()  
        coplayer_commitment_probs = self.coplayer_commit_actor(torch.cat((state, coplayer_proposal, self_proposal),dim=1))
        coplayer_commitment_probs = self.epsilon_greedy_probs(coplayer_commitment_probs, epsilon)
        coplayer_commitment_logits = torch.log(coplayer_commitment_probs+self.perturb)
        coplayer_commitment = F.gumbel_softmax(coplayer_commitment_logits, hard=True, tau=self.temperature)
        coplayer_commitment_logit = coplayer_commitment_logits[torch.arange(coplayer_commitment_logits.shape[0]), coplayer_commitment.clone().detach().argmax(dim=1)]
        coplayer_is_commitment = self.is_commit(coplayer_commitment).squeeze()
        
        q_sa_coplayer = self.coplayer_critic(torch.cat((state, coplayer_action, self_action),dim=1)).detach().squeeze()
        q_sm_coplayer = self.coplayer_critic(torch.cat((state, coplayer_proposal, self_proposal),dim=1)).detach().squeeze()
        advantage_sa_coplayer = q_sa_coplayer - self.calculate_mega_state_value_coplayer(state, coplayer_proposal, self_proposal, coplayer_commitment, self_commitment, coplayer_action, self_action)
        advantage_sm_coplayer = q_sm_coplayer - self.calculate_mega_state_value_coplayer(state, coplayer_proposal, self_proposal, coplayer_commitment, self_commitment, coplayer_action, self_action)
        
        # commitment_loss_coplayer = (is_mutual_commitment * (-q_sm_coplayer * coplayer_commitment_logit + (q_sa_coplayer-q_sm_coplayer)*coplayer_is_commitment)+(1-is_mutual_commitment) * (-q_sa_coplayer * coplayer_commitment_logit)).mean()
        commitment_loss_coplayer = (is_mutual_commitment * (-advantage_sm_coplayer * coplayer_commitment_logit + (advantage_sa_coplayer-advantage_sm_coplayer)*coplayer_is_commitment)+(1-is_mutual_commitment) * (-advantage_sa_coplayer * coplayer_commitment_logit)).mean()
        if self.is_entropy:
            entropy = -torch.mean(coplayer_commitment_probs * torch.log(coplayer_commitment_probs + self.perturb))
            # entropy = entropy/torch.log(torch.tensor(2, dtype=self.dtype))
            commitment_loss_coplayer -= entropy_coeff * entropy
        self.coplayer_commit_actor_optimizer.zero_grad()
        commitment_grads_coplayer = torch.autograd.grad(commitment_loss_coplayer, list(self.coplayer_commit_actor.parameters()), retain_graph=True)
        commitment_params_coplayer = list(self.coplayer_commit_actor.parameters())
        for layer in range(len(commitment_params_coplayer)):
            commitment_params_coplayer[layer].grad = commitment_grads_coplayer[layer]
            commitment_params_coplayer[layer].grad.data.clamp_(-1, 1)
        self.coplayer_commit_actor_optimizer.step()

    def update_proposal_policy(self, state, entropy_coeff, epsilon):
        # We need Gumbel-softmax sample for proposal and commitment, the derivatives should be retained in these samples.
        self_proposal_probs = self.proposing_actor(state)
        self_proposal_probs = self.epsilon_greedy_probs(self_proposal_probs, epsilon)
        self_proposal_logits = torch.log(self_proposal_probs+self.perturb)
        self_proposal = F.gumbel_softmax(self_proposal_logits, hard=True, tau=self.temperature)
        self_proposal_logit = self_proposal_logits[torch.arange(self_proposal_logits.shape[0]), self_proposal.argmax(dim=1)]
        coplayer_proposal_probs = self.coplayer_proposing_actor(state)
        coplayer_proposal_probs = self.epsilon_greedy_probs(coplayer_proposal_probs, epsilon)
        coplayer_proposal_logits = torch.log(coplayer_proposal_probs+self.perturb)
        coplayer_proposal = F.gumbel_softmax(coplayer_proposal_logits, hard=True, tau=self.temperature)

        self_commitment_probs = self.commit_actor(torch.cat((state, self_proposal, coplayer_proposal),dim=1))
        self_commitment_probs = self.epsilon_greedy_probs(self_commitment_probs)
        self_commitment_logits = torch.log(self_commitment_probs+self.perturb)
        self_commitment = F.gumbel_softmax(self_commitment_logits, hard=True, tau=self.temperature)
        self_is_commitment = self.is_commit(self_commitment).squeeze()
        self_commitment_logit = self_commitment_logits[torch.arange(self_commitment_logits.shape[0]), self_commitment.argmax(dim=1)]
        coplayer_commitment_probs = self.coplayer_commit_actor(torch.cat((state, coplayer_proposal, self_proposal),dim=1))
        coplayer_commitment_probs = self.epsilon_greedy_probs(coplayer_commitment_probs)
        coplayer_commitment_logits = torch.log(coplayer_commitment_probs+self.perturb)
        coplayer_commitment = F.gumbel_softmax(coplayer_commitment_logits, hard=True, tau=self.temperature)
        coplayer_commitment_logit = coplayer_commitment_logits[torch.arange(coplayer_commitment_logits.shape[0]), coplayer_commitment.argmax(dim=1)]
        coplayer_is_commitment = self.is_commit(coplayer_commitment).squeeze()
        is_mutual_commitment = (self_is_commitment*coplayer_is_commitment).detach()

        self_action_probs = self.unconstrained_actor(state).detach()
        self_action_probs = self.epsilon_greedy_probs(self_action_probs)
        self_action_logits = torch.log(self_action_probs+self.perturb)
        self_action = F.gumbel_softmax(self_action_logits, hard=True, tau=self.temperature)
        coplayer_action_probs = self.coplayer_unconstrained_actor(state).detach()
        coplayer_action_probs = self.epsilon_greedy_probs(coplayer_action_probs)
        coplayer_action_logits = torch.log(coplayer_action_probs+self.perturb)
        coplayer_action = F.gumbel_softmax(coplayer_action_logits, hard=True, tau=self.temperature).detach()

        q_sa = self.critic(torch.cat((state, self_action, coplayer_action),dim=1)).detach().squeeze()
        q_sm = self.critic(torch.cat((state, self_proposal, coplayer_proposal),dim=1)).detach().squeeze()
        q_sa_coplayer = self.coplayer_critic(torch.cat((state, coplayer_action, self_action),dim=1)).detach().squeeze()
        q_sm_coplayer = self.coplayer_critic(torch.cat((state, coplayer_proposal, self_proposal),dim=1)).detach().squeeze()
        advantage_sa = q_sa - self.calculate_mega_state_value(state, self_proposal, coplayer_proposal, self_commitment, coplayer_commitment, self_action, coplayer_action)
        advantage_sm = q_sm - self.calculate_mega_state_value(state, self_proposal, coplayer_proposal, self_commitment, coplayer_commitment, self_action, coplayer_action)
        advantage_sa_coplayer = q_sa_coplayer - self.calculate_mega_state_value_coplayer(state, coplayer_proposal, self_proposal, coplayer_commitment, self_commitment, coplayer_action, self_action)
        advantage_sm_coplayer = q_sm_coplayer - self.calculate_mega_state_value_coplayer(state, coplayer_proposal, self_proposal, coplayer_commitment, self_commitment, coplayer_action, self_action)

        self.proposing_actor_optimizer.zero_grad() # Zero the gradients
        # proposing gradients part
        # proposal_loss = is_mutual_commitment * (-q_sm * (self_proposal_logit + self_commitment_logit + coplayer_commitment_logit)) + (q_sa-q_sm)*(self_is_commitment+coplayer_is_commitment) + (1-is_mutual_commitment) * (-q_sa * (self_proposal_logit + self_commitment_logit+ coplayer_commitment_logit))
        proposal_loss = is_mutual_commitment * (-advantage_sm * (self_proposal_logit + self_commitment_logit + coplayer_commitment_logit)) + (advantage_sa-advantage_sm)*(self_is_commitment+coplayer_is_commitment) + (1-is_mutual_commitment) * (-advantage_sa * (self_proposal_logit + self_commitment_logit+ coplayer_commitment_logit))
        if self.with_constraints==True:
            # proposal_loss += torch.abs((q_sa-q_sm))*torch.maximum((q_sa-q_sm),torch.tensor(0.0))*self_proposal_logit
            # proposal_loss += torch.abs((q_sa_coplayer-q_sm_coplayer))*torch.maximum((q_sa_coplayer-q_sm_coplayer),torch.tensor(0.0))*self_proposal_logit
            proposal_loss += torch.abs((advantage_sa-advantage_sm))*torch.maximum((advantage_sa-advantage_sm),torch.tensor(0.0))*self_proposal_logit
            proposal_loss += torch.abs((advantage_sa_coplayer-advantage_sm_coplayer))*torch.maximum((advantage_sa_coplayer-advantage_sm_coplayer),torch.tensor(0.0))*self_proposal_logit
        if self.is_entropy: # Get entropy of the proposal network
            entropy = -torch.mean(self_proposal_probs * torch.log(self_proposal_probs + self.perturb))
            # entropy = entropy/torch.log(torch.tensor(self.action_dim, dtype=self.dtype))
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

    def update_coplayer_proposal_policy(self, state, entropy_coeff, epsilon):
        """
        update estimate for coplayer
        """
        self_proposal_probs = self.proposing_actor(state)
        self_proposal_probs = self.epsilon_greedy_probs(self_proposal_probs, epsilon)
        self_proposal_logits = torch.log(self_proposal_probs+self.perturb)
        self_proposal = F.gumbel_softmax(self_proposal_logits, hard=True, tau=self.temperature)
        coplayer_proposal_probs = self.coplayer_proposing_actor(state)
        coplayer_proposal_probs = self.epsilon_greedy_probs(coplayer_proposal_probs, epsilon)
        coplayer_proposal_logits = torch.log(coplayer_proposal_probs+self.perturb)
        coplayer_proposal = F.gumbel_softmax(coplayer_proposal_logits, hard=True, tau=self.temperature)
        coplayer_proposal_logit = coplayer_proposal_logits[torch.arange(coplayer_proposal_logits.shape[0]), coplayer_proposal.argmax(dim=1)]

        self_commitment_probs = self.commit_actor(torch.cat((state, self_proposal, coplayer_proposal),dim=1))
        self_commitment_probs = self.epsilon_greedy_probs(self_commitment_probs)
        self_commitment_logits = torch.log(self_commitment_probs+self.perturb)
        self_commitment = F.gumbel_softmax(self_commitment_logits, hard=True, tau=self.temperature)
        self_is_commitment = self.is_commit(self_commitment).squeeze()
        self_commitment_logit = self_commitment_logits[torch.arange(self_commitment_logits.shape[0]), self_commitment.argmax(dim=1)]
        coplayer_commitment_probs = self.coplayer_commit_actor(torch.cat((state, coplayer_proposal, self_proposal),dim=1))
        coplayer_commitment_probs = self.epsilon_greedy_probs(coplayer_commitment_probs)
        coplayer_commitment_logits = torch.log(coplayer_commitment_probs+self.perturb)
        coplayer_commitment = F.gumbel_softmax(coplayer_commitment_logits, hard=True, tau=self.temperature)
        coplayer_commitment_logit = coplayer_commitment_logits[torch.arange(coplayer_commitment_logits.shape[0]), coplayer_commitment.argmax(dim=1)]
        coplayer_is_commitment = self.is_commit(coplayer_commitment).squeeze()
        is_mutual_commitment = (self_is_commitment*coplayer_is_commitment).detach()

        self_action_probs = self.unconstrained_actor(state).detach()
        self_action_probs = self.epsilon_greedy_probs(self_action_probs)
        self_action_logits = torch.log(self_action_probs+self.perturb)
        self_action = F.gumbel_softmax(self_action_logits, hard=True, tau=self.temperature).detach()
        coplayer_action_probs = self.coplayer_unconstrained_actor(state).detach()
        coplayer_action_probs = self.epsilon_greedy_probs(coplayer_action_probs)
        coplayer_action_logits = torch.log(coplayer_action_probs+self.perturb)
        coplayer_action = F.gumbel_softmax(coplayer_action_logits, hard=True, tau=self.temperature).detach()
        q_sa = self.critic(torch.cat((state, self_action, coplayer_action),dim=1)).detach().squeeze()
        q_sm = self.critic(torch.cat((state, self_proposal, coplayer_proposal),dim=1)).detach().squeeze()
        q_sa_coplayer = self.coplayer_critic(torch.cat((state, coplayer_action, self_action),dim=1)).detach().squeeze()
        q_sm_coplayer = self.coplayer_critic(torch.cat((state, coplayer_proposal, self_proposal),dim=1)).detach().squeeze()
        advantage_sa = q_sa - self.calculate_mega_state_value(state, self_proposal, coplayer_proposal, self_commitment, coplayer_commitment, self_action, coplayer_action)
        advantage_sm = q_sm - self.calculate_mega_state_value(state, self_proposal, coplayer_proposal, self_commitment, coplayer_commitment, self_action, coplayer_action)
        advantage_sa_coplayer = q_sa_coplayer - self.calculate_mega_state_value_coplayer(state, coplayer_proposal, self_proposal, coplayer_commitment, self_commitment, coplayer_action, self_action)
        advantage_sm_coplayer = q_sm_coplayer - self.calculate_mega_state_value_coplayer(state, coplayer_proposal, self_proposal, coplayer_commitment, self_commitment, coplayer_action, self_action)

        self.coplayer_proposing_actor_optimizer.zero_grad()
        # proposal_loss_coplayer = is_mutual_commitment * (-q_sm_coplayer * (coplayer_proposal_logit + coplayer_commitment_logit + self_commitment_logit)) + (q_sa_coplayer-q_sm_coplayer)*(self_is_commitment+coplayer_is_commitment) + (1-is_mutual_commitment) * (-q_sa_coplayer * (coplayer_proposal_logit + coplayer_commitment_logit + self_commitment_logit))
        proposal_loss_coplayer = is_mutual_commitment * (-advantage_sm_coplayer * (coplayer_proposal_logit + coplayer_commitment_logit + self_commitment_logit)) + (advantage_sa_coplayer-advantage_sm_coplayer)*(self_is_commitment+coplayer_is_commitment) + (1-is_mutual_commitment) * (-advantage_sa_coplayer * (coplayer_proposal_logit + coplayer_commitment_logit + self_commitment_logit))
        if self.with_constraints==True:
            # proposal_loss_coplayer += torch.abs((q_sa_coplayer-q_sm_coplayer))*torch.maximum((q_sa_coplayer-q_sm_coplayer),torch.tensor(0.0))*coplayer_proposal_logit
            # proposal_loss_coplayer += torch.abs((q_sa-q_sm))*torch.maximum((q_sa-q_sm),torch.tensor(0.0))*coplayer_proposal_logit
            proposal_loss_coplayer += torch.abs((advantage_sa_coplayer-advantage_sm_coplayer))*torch.maximum((advantage_sa_coplayer-advantage_sm_coplayer),torch.tensor(0.0))*coplayer_proposal_logit
            proposal_loss_coplayer += torch.abs((advantage_sa-advantage_sm))*torch.maximum((advantage_sa-advantage_sm),torch.tensor(0.0))*coplayer_proposal_logit
        if self.is_entropy:
            entropy = -torch.mean(coplayer_proposal_probs * torch.log(coplayer_proposal_probs + self.perturb))
            # entropy = entropy/torch.log(torch.tensor(self.action_dim, dtype=self.dtype))
            proposal_loss_coplayer = proposal_loss_coplayer.mean()- entropy_coeff * entropy
        else:
            proposal_loss_coplayer = proposal_loss_coplayer.mean()
        proposal_grads_coplayer = torch.autograd.grad(proposal_loss_coplayer, list(self.coplayer_proposing_actor.parameters()),retain_graph=True)
        proposal_params_coplayer = list(self.coplayer_proposing_actor.parameters())
        for layer in range(len(proposal_params_coplayer)):
            proposal_params_coplayer[layer].grad = proposal_grads_coplayer[layer]
            proposal_params_coplayer[layer].grad.data.clamp_(-1, 1)
        self.coplayer_proposing_actor_optimizer.step()
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

    def update_coplayer_critic(self, state, self_proposal, coplayer_proposal, self_action, coplayer_action, is_mutual_commitment, coplayer_return):
        """
        Update critic Q^i(s,a^i,a^j)
        """
        # Compute the actual value
        is_mutual_commitment_expanded = is_mutual_commitment.expand(-1, self_proposal.shape[1])
        real_action = is_mutual_commitment_expanded * self_proposal + (1-is_mutual_commitment_expanded) * self_action
        real_coplayer_action = is_mutual_commitment_expanded * coplayer_proposal + (1-is_mutual_commitment_expanded) * coplayer_action
        actual_value_coplayer = self.coplayer_critic(torch.cat((state, real_coplayer_action, real_action),dim=1)).squeeze()
        target_value_coplayer = coplayer_return.squeeze()
        
        loss_func = torch.nn.MSELoss()
        critic_loss_coplayer = loss_func(actual_value_coplayer, target_value_coplayer)
        self.coplayer_critic_optimizer.zero_grad()
        critic_grads_coplayer = torch.autograd.grad(critic_loss_coplayer, list(self.coplayer_critic.parameters()))
        critic_params_coplayer = list(self.coplayer_critic.parameters())
        for layer in range(len(critic_params_coplayer)):
            critic_params_coplayer[layer].grad = critic_grads_coplayer[layer]
            critic_params_coplayer[layer].grad.data.clamp_(-1, 1)
        self.coplayer_critic_optimizer.step()