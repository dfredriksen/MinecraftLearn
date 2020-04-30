from threading import Thread, active_count
import numpy as np

class LearningThread(Thread):

  def __init__(self,replay_memory, state, action, next_state, reward, steps_done, environment, episode, inventory, learning_threads):
    self.memory = replay_memory
    self.state = state
    self.action = action
    self.next_state = next_state
    self.reward = reward
    self.steps_done = steps_done
    self.blocking_threads = learning_threads
    self.environment = environment
    self.episode = episode
    self.inventory = inventory
    Thread.__init__(self)

  def run(self):
    print(str(active_count()) + ' learning threads are alive')    
    im, environment = self.memory.prepare_environment(self.environment, self.steps_done, self.episode)
    reward = self.memory.calculate_reward(self.state, self.reward, np.array(im), self.inventory)
    print('Reward: ' + str(reward))
    self.memory.push(self.steps_done, self.state, self.action, self.next_state, environment, reward)
    for blocking_thread in self.blocking_threads:
      if(blocking_thread.is_alive() and blocking_thread.name != self.name):
        blocking_thread.join()
    self.memory.optimize_model()
