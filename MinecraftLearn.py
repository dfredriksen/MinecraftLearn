from gym_MinecraftLive.envs.MinecraftLive_env import MinecraftLiveEnv
from gym import logger
import math
import numpy as np
import os, time
import torch
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image
import torchvision.transforms as T
import random
from DQN import DQN
from ReplayMemory import ReplayMemory
from LearningThread import LearningThread
import torch.optim as optim

from MinecraftMind import Mind

TRIAL_LABEL = "Trial1"
MINECRAFT_SCREENSHOT_PATH = "C:\\Users\\dfred\\AppData\\Roaming\\.minecraft\\screenshots\\"
MINECRAFT_LAUNCHER_PATH = "C:\\Program Files (x86)\\Minecraft Launcher\\MinecraftLauncher.exe"
SCREENSHOT_HISTORY_ROOT_PATH = "C:\\Users\\dfred\\Desktop\\Projects\\RL\\MinecraftLearnData\\" + TRIAL_LABEL + "\\"
TESSERACT_PATH = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
MEMORY_PATH = "C:\\Users\\dfred\\Desktop\\Projects\\RL\\MinecraftLearnData\\" + TRIAL_LABEL + "\\"
STATE_DICT_PATH = os.path.join(MEMORY_PATH, 'state_dict.dat')
LOG_LEVEL = 5
agent = Mind(TESSERACT_PATH, 'none', logger)
env = MinecraftLiveEnv(agent, MINECRAFT_SCREENSHOT_PATH)
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym.
screen_width = int(agent.resolution[0] * .05)
screen_height = int(agent.resolution[1] * .05)

resize = T.Compose([T.ToPILImage(),
                    T.Resize([screen_width, screen_height], interpolation=Image.CUBIC),
                    T.ToTensor()])

#button_coordinates = agent.start_launcher(MINECRAFT_LAUNCHER_PATH)
#agent.play_minecraft_multiplayer(button_coordinates, 60)
time.sleep(5)


def print_output(message, level):
  if LOG_LEVEL >= level:
    print(message)


def get_screen():
    # Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    result = resize(screen).unsqueeze(0).to(device)
    return result

# Get number of actions from gym action space
n_actions = 0

for actions in agent.action_spaces:
  n_actions = n_actions + actions

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)

if(os.path.exists(STATE_DICT_PATH)):
  policy_net.load_state_dict(torch.load(STATE_DICT_PATH))
  policy_net.eval()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(MEMORY_PATH, STATE_DICT_PATH, SCREENSHOT_HISTORY_ROOT_PATH, 10000, optimizer, device, policy_net, target_net, agent.action_spaces)
steps_done = 0
episode_durations = []
learning_threads = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


i_episode = -1
memory.read_memory(MEMORY_PATH)
while True:
    i_episode = i_episode + 1
    print_output('Starting episode ' + str(i_episode) + '...', 3)
    try:
      os.mkdir(SCREENSHOT_HISTORY_ROOT_PATH + str(i_episode))
    except: 
      print_output('Unable to create history directory', 5)
    env.set_screenshot_history_path(SCREENSHOT_HISTORY_ROOT_PATH + str(i_episode))
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = memory.select_action(state, steps_done)
        action_list = action.numpy()[0]
        print_output(action_list, 5)
        environment, reward, done, _ = env.step(action_list)
        steps_done = steps_done + 1
        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        learning_thread = LearningThread(memory, state, action, next_state, reward, steps_done, environment, i_episode, learning_threads)
        learning_thread.start()
        learning_threads.append(learning_thread)        
        # Move to the next state
        state = next_state
            
        if done:
            print_output('Agent has died...', 3)
            episode_durations.append(t + 1)
            break


print_output('Complete', 1)
env.render()
env.close()
plot_durations()
plt.ioff()
plt.show()

