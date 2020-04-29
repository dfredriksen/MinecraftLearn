import random
from collections import namedtuple
import os
from threading import Thread
import torch
import torch.nn.functional as F
import math, numpy as np
from PIL import Image

MEMORY_FILE = 'memory.dat'

class ReplayMemory():

  BATCH_SIZE = 128
  GAMMA = 0.999
  EPS_START = 0.9
  EPS_END = 0.05
  EPS_DECAY = 200
  TARGET_UPDATE = 128

  Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'environment', 'reward'))
  def __init__(self, MEMORY_PATH, STATE_DICT_PATH, SCREENSHOT_HISTORY_ROOT_PATH, capacity, optimizer, device, policy_net, target_net, action_spaces):
      Thread.__init__(self)
      self.capacity = capacity
      self.memory = []
      self.position = 0
      self.optimizer = optimizer
      self.device = device
      self.MEMORY_PATH = MEMORY_PATH
      self.policy_net = policy_net
      self.target_net = target_net
      self.STATE_DICT_PATH = STATE_DICT_PATH
      self.action_spaces = action_spaces
      self.SCREENSHOT_HISTORY_ROOT_PATH = SCREENSHOT_HISTORY_ROOT_PATH
      self.last_brightness = 0
      self.last_health = 0
      self.last_food = 0


  def push(self, *args):
      """Saves a transition."""
      if len(self.memory) < self.capacity:
          self.memory.append(None)
      self.memory[self.position] = self.Transition(*args)
      self.position = (self.position + 1) % self.capacity


  def sample(self, batch_size):
      return random.sample(self.memory, batch_size)


  def read_memory(self, path):
      if(os.path.isfile(path)):
          f = open(os.path.join(path,MEMORY_FILE), 'r')
          memstring = f.read()
          f.close()
          self.memory = memstring.split('|')
          self.position = len(self.memory) % self.capacity


  def save_memory(self, path):
      f = open(os.path.join(path,MEMORY_FILE), 'w+')
      memstring = "|".join(self.memory)
      f.write(memstring)
      f.close()

  
  def calculate_reward(self, state, reward, image_data):
      hx1 = int(image_data.shape[1] * 0.312)
      hx2 = int(image_data.shape[1] * 0.458)
      hy1 = int(image_data.shape[0] * 0.87) +1
      hy2 = hy1+1
      
      fx1 = int(image_data.shape[1] * 0.52)
      fx2 = int(image_data.shape[1] * 0.67)
      fy1 = int(image_data.shape[0] * 0.87) + 1
      fy2 = fy1+1

      im = Image.fromarray(image_data)
      im2 = im.crop((hx1, hy1, hx2, hy2))
      im2 = im2.convert("L")
      im3 = im.crop((fx1, fy1, fx2, fy2))
      im3 = im3.convert("L")
      health = np.array(im2)
      food = np.array(im3)
      hmax = np.max(health[0])
      fmax = np.max(food[0])
      count_health = np.count_nonzero(health == hmax)
      count_food = np.count_nonzero(food == fmax)
      brightness = np.mean(image_data)
      health = count_health
      food = count_food

      new_reward = 0
      if health < 0:
        health = 0
      if food < 0:
        food = 0
      
      if(brightness > 150):
        new_reward = new_reward + 1
      elif(brightness < 50):
        new_reward = new_reward - 1

      if(self.last_health > 0):
        new_reward = new_reward + abs(self.last_health - health)
      
      if(self.last_food > 0):
        new_reward = new_reward + abs(self.last_food - food)

      self.last_health = health
      self.last_food = food

      return torch.tensor([[new_reward for i in range(len(self.action_spaces))]], device=self.device)


  def prepare_environment(self, environment_path, steps_done, episode):
      new_filename = os.path.join(os.path.join(self.SCREENSHOT_HISTORY_ROOT_PATH, str(episode)), str(steps_done) + '.png')
      os.rename(environment_path, new_filename)
      im = Image.open(new_filename)
      return [im, new_filename]

  def optimize_model(self):
      if len(self.memory) < self.BATCH_SIZE:
          return
    
      transitions = self.sample(self.BATCH_SIZE)
      # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
      # detailed explanation). This converts batch-array of Transitions
      # to Transition of batch-arrays.
      batch = self.Transition(*zip(*transitions))

      # Compute a mask of non-final states and concatenate the batch elements
      # (a final state would've been the one after which simulation ended)
      non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
      non_final_next_states = torch.cat([s for s in batch.next_state
                                                  if s is not None])
      state_batch = torch.cat(batch.state)
      action_batch = torch.cat(batch.action)
      reward_batch = torch.cat(batch.reward)

      # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
      # columns of actions taken. These are the actions which would've been taken
      # for each batch state according to policy_net
      state_action_values = self.policy_net(state_batch).gather(1, action_batch)

      # Compute V(s_{t+1}) for all next states.
      # Expected values of actions for non_final_next_states are computed based
      # on the "older" target_net; selecting their best reward with max(1)[0].
      # This is merged based on the mask, such that we'll have either the expected
      # state value or 0 in case the state was final.
      next_state_values = torch.zeros([self.BATCH_SIZE,5], device=self.device)
      action_tensor = self.target_net(non_final_next_states)
      next_state_values[non_final_mask] = self.process_state_actions(action_tensor.detach()).type(torch.FloatTensor)
      # Compute the expected Q values
      expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
      # Compute Huber loss
      loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

      # Optimize the model
      self.optimizer.zero_grad()
      loss.backward()
      for param in self.policy_net.parameters():
          param.grad.data.clamp_(-1, 1)
      
      self.optimizer.step()
      return [self.policy_net, self.target_net]

  def process_state_actions(self, action_tensor):
      placeholder = action_tensor.numpy()[0]
      action_probabilities = []
      count = 0
      for group, action_items in enumerate(self.action_spaces):
        action_probabilities.append([])
        for index in range(count, count + action_items):
          action_probabilities[group].append(placeholder[index])
          count = count + 1

      choices = []
      for discrete_action_space in action_probabilities:
        choices.append(np.argmax(discrete_action_space))
      choice_tensor = torch.tensor([choices])  
      return choice_tensor


  def select_action(self, state, steps_done):
      sample = random.random()
      eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
          math.exp(-1. * steps_done / self.EPS_DECAY)
      if sample > eps_threshold:
          with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #return policy_net(state).max(1)[1].view(1, 1)
            output = self.policy_net(state)
            return self.process_state_actions(output)
      else:
          random_actions = []
          for actions in self.action_spaces:
            random_actions.append(random.randrange(actions))
          return torch.tensor([random_actions], device=self.device, dtype=torch.long)


  def save_model(self):
      self.target_net.load_state_dict(self.policy_net.state_dict())
      torch.save(self.policy_net.state_dict(), self.STATE_DICT_PATH)


  def __len__(self):
      return len(self.memory) 