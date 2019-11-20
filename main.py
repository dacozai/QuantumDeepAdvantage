from Agent.dqn import *
from Environment.Environment import *
import matplotlib.pyplot as plt

NUM_QUBITS = 2
NUM_SIM = 10

env = Environment(NUM_QUBITS) 
# bot = dqn(num_qubits=NUM_QUBITS)
bot = drqn(num_qubits=NUM_QUBITS)

def learning(bot, env):
  #for sim_times in range(NUM_SIM):
  reward = 0
  state  = env.state
  ct = 0
  while not env.is_terminated():
    bot.learn_from_transition(state, reward, env.is_terminated())
    move = bot.get_action(state)
    env.step(move)
    state  = env.state
    reward = env.reward()
    # bot.total_reward += reward
    ct+=1

  bot.learn_from_transition(state, reward, env.is_terminated())
  bot.total_reward += env.reward()
  if reward is not -1:
    bot.win_times += 1

  env.reset()
  # bot.reset()
  # return bot.total_reward

reward_array = []
episodes = 1000
gap = 50
for num in range(1, episodes+1):
  print("Times is "+str(num))
  learning(bot, env)
  if num%gap == 0:
    reward_array.append(bot.win_times/gap*100)
    bot.total_reward = 0
    bot.win_times = 0

xAxis = [ num for num in range(1, int(episodes/gap)+1 )]
_, ag = plt.subplots()
ag.plot(xAxis, reward_array, color = 'blue', label="DQN")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ag.set(xlabel='episodes', ylabel='average return(agents)',
      title='Learning curve')
ag.grid()
plt.show()





