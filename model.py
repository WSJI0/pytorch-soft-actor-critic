import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6

LAYER_NORM = False

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.ln1 = nn.LayerNorm(hidden_dim, hidden_dim)
        self.linear11 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear12 = nn.Linear(hidden_dim, hidden_dim)
        self.linear13 = nn.Linear(hidden_dim, hidden_dim)
        self.linear14 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.ln2 = nn.LayerNorm(hidden_dim, hidden_dim)
        self.linear21 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear22 = nn.Linear(hidden_dim, hidden_dim)
        self.linear23 = nn.Linear(hidden_dim, hidden_dim)
        self.linear24 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        if LAYER_NORM: x1 = F.relu(self.ln1(self.linear11(xu)))
        else: x1 = F.relu(self.linear11(xu))
        x1 = F.relu(self.linear12(x1))
        # x1 = F.relu(self.linear13(x1))
        x1 = self.linear14(x1)

        if LAYER_NORM: x2 = F.relu(self.ln2(self.linear21(xu)))
        else: x2 = F.relu(self.linear21(xu))
        x2 = F.relu(self.linear22(x2))
        # x2 = F.relu(self.linear23(x2))
        x2 = self.linear24(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.ln = nn.LayerNorm(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        if LAYER_NORM: x = F.relu(self.ln(self.linear1(state)))
        else: x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
    

class GsdeGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None,
                 sde_sigma=0.5, use_layernorm=False):
        super().__init__()
        self.use_ln = use_layernorm
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=True)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)  # base std(s)

        # gSDE용: 상태특징(phi) -> action 차원으로 투영하는 가중치
        #   탐색 증폭은 |phi(s) @ (W_sde ⊙ Z)| 로 계산
        self.sde_proj = nn.Linear(hidden_dim, num_actions, bias=False)
        nn.init.orthogonal_(self.sde_proj.weight, gain=0.1)
        self.register_buffer("sde_noise", torch.zeros_like(self.sde_proj.weight))
        self.sde_sigma = float(sde_sigma)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.as_tensor((action_space.high - action_space.low) / 2.0)
            self.action_bias  = torch.as_tensor((action_space.high + action_space.low) / 2.0)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def reset_noise(self, std: float = 1.0):
        """
        gSDE 노이즈 행렬 Z를 재샘플. 보통 '롤아웃 시작' 시점에 한 번 호출.
        """
        with torch.no_grad():
            z = torch.randn_like(self.sde_proj.weight) * std
            self.sde_noise.copy_(z)

    def _features(self, state):
        x = self.linear1(state)
        x = self.ln(x) if self.use_ln else x
        x = F.relu(x)
        x = F.relu(self.linear2(x))
        return x

    def forward(self, state):
        # mean, base log_std
        h = self._features(state)
        mean = self.mean_linear(h)
        log_std = self.log_std_linear(h).clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, h

    def _std_eff(self, base_std, h):
        delta = F.linear(h, self.sde_proj.weight * self.sde_noise)  # => (B, num_actions)
        gain = 1.0 + self.sde_sigma * torch.tanh(delta).abs()
        std_eff = base_std * gain
        return std_eff

    def sample(self, state):
        mean, log_std, h = self.forward(state)
        base_std = log_std.exp()

        # gSDE: 상태 및 고정 Z에 의존하는 std 증폭
        std_eff = self._std_eff(base_std, h)

        normal = Normal(mean, std_eff)
        x_t = normal.rsample()              # reparameterization
        y_t = torch.tanh(x_t)               # squash
        action = y_t * self.action_scale + self.action_bias

        # log_prob (squash 보정 포함)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # deterministic mean (테스트 시)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias  = self.action_bias.to(device)
        return super().to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
    

class BetaPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(BetaPolicy, self).__init__()

        self.ln = nn.LayerNorm(hidden_dim) if LAYER_NORM else None
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.alpha_head = nn.Linear(hidden_dim, num_actions)
        self.beta_head = nn.Linear(hidden_dim, num_actions)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.linear1(state)
        if self.ln:
            x = F.relu(self.ln(x))
        else:
            x = F.relu(x)
        x = F.relu(self.linear2(x))

        alpha = F.softplus(self.alpha_head(x)) + 1.0  # Ensure alpha > 1
        beta = F.softplus(self.beta_head(x)) + 1.0    # Ensure beta > 1

        return alpha, beta

    def sample(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        raw_action = dist.rsample()  # ∈ (0, 1)
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)

        action = raw_action * self.action_scale.to(state.device) + self.action_bias.to(state.device)
        mean = (alpha / (alpha + beta)) * self.action_scale.to(state.device) + self.action_bias.to(state.device)
        return action, log_prob, mean # raw_action -> mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(BetaPolicy, self).to(device)
