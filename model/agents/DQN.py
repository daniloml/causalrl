# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_
from model import DQN
from model.layers.NoisyLinear import NoisyLinear

class DQN(nn.Module):
  def __init__(self, cfg, action_space):
    super(DQN, self).__init__()
    self.atoms = cfg["atoms"]
    self.action_space = action_space

    if cfg["architecture"] == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(cfg["history_length"], 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif cfg["architecture"] == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(cfg["history_length"], 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = NoisyLinear(self.conv_output_size, cfg["hidden_size"], std_init=cfg["noisy_std"])
    self.fc_h_a = NoisyLinear(self.conv_output_size, cfg["hidden_size"], std_init=cfg["noisy_std"])
    self.fc_z_v = NoisyLinear(cfg["hidden_size"], self.atoms, std_init=cfg["noisy_std"])
    self.fc_z_a = NoisyLinear(cfg["hidden_size"], action_space * self.atoms, std_init=cfg["noisy_std"])

  def forward(self, x, log=False):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

class Rainbow(nn.Module):
  def __init__(self, cfg, env):
    self.action_space = env.action_space()
    self.atoms = cfg['atoms']
    self.Vmin = cfg["V_min"]
    self.Vmax = cfg["V_max"]
    self.support = torch.linspace(cfg["V_min"], cfg["V_max"], self.atoms).to(device=cfg["device"])  # Support (range) of z
    self.delta_z = (cfg["V_max"] - cfg["V_min"]) / (self.atoms - 1)
    self.batch_size = cfg["batch_size"]
    self.n = cfg["multi_step"]
    self.discount = cfg["discount"]
    self.norm_clip = cfg["norm_clip"]
    self.online_net = DQN(cfg, self.action_space).to(device=cfg["device"])
    
    if "model" in cfg.keys():  # Load pretrained model if provided
      if os.path.isfile(cfg["model"]):
        state_dict = torch.load(cfg["model"], map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + cfg["model"])
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(cfg["model"])

    self.online_net.train() # Puts network on training mode (as opposed to evaluation mode).

    self.target_net = DQN(cfg, self.action_space).to(device=cfg["device"])
    self.update_target_net()
    self.target_net.train()  # Puts network on training mode (as opposed to evaluation mode).
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=cfg["learning_rate"], eps=cfg["adam_eps"])

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    self.eval()
    with torch.no_grad():
      action = (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
    self.train()
    return action

  # Acts with an ??-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ?? can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def update(self, mem, logger, step):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
    logger.log('train/batch_returns', returns.mean(), step)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ??; ??online)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; ??online)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ??; ??online)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ??; ??online))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; ??online))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ??; ??target)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; ??online))]; ??target)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (??^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / ??z
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    weighted_loss = (weights * loss).mean()
    logger.log('train/loss', loss.mean(), step)
    logger.log('train/weighted_loss', weighted_loss, step)

    self.online_net.zero_grad()
    weighted_loss.backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
