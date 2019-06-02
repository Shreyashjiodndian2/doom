import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from skimage import transform
import keras
import random
import time
from vizdoom import *

def create_environment():
    game = DoomGame() # call function for furthur use in program
    game.load_config("basic.cfg")  # load configuration file in game so that we don't have to do so much prprocessing
    game.set_doom_scenario_path("basic.wad")
    game.init() # initialize the game
    
    
    left = [1,0,0]
    right = [0,0,1]
    shoot = [0,1,0]
    possible_actions = [left, right, shoot]
    
    return game, possible_actions
    
    
    game, possible_actions = create_environment()
    
    def test_env():
      game = DoomGame()
      game.load_config("basic.cfg")
      game.set_doom_scenario_path("basic.wad")
      game.init()


      left = [1,0,0]
      right = [0,0,1]
      shoot = [0,1,0]
      actions = [left, right, shoot]

      episodes = 10
      for _ in range(episodes):
          game.new_episode()
          if not game.is_episode_finished():
              state = game.get_state()
              img  = state.screen_buffer
              action = random.choice(actions)
              misc = state.game_variables
              rewards = game.make_action(action)
              print("reward= {}".format(rewards))
              time.sleep(0.002)
          print("total_reward = ",game.get_total_reward)
          time.sleep(1)
      game.close()
    
    def preprocess_frame(frame):
      img = frame[30:-10, 30:-30] # remove roof as it does not contain any info
      img = img/255  # scale down image array for fast processing
      img = transform.resize(img, [84,84])

      return img
    
    stack_size = 4 # number of frames to be stacked

# initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

def stack_frames(stacked_frames, state, is_new_episode):
    # preprocess frame for furthur use
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # copy same frame 4x because it is beginning
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # stacking the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # commented was done to try another apprach to decrease processing time
        """stacked_frames[:][:][1:] = stacked_frames[:][:][:-1]
        stacked_frames[:,:,0] = frame"""
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

### MODEL HYPERPARAMETERS
state_size = [84,84,4]      # input is stack of 4 frames so dimension is (84,84,4) 
action_size = game.get_available_buttons_size()              # this option gives number of possible actions in game i.e. [left, right, shoot]
learning_rate =  0.0002      # learning rate

### TRAINING HYPERPARAMETERS
total_episodes = 500        # number of episodes for training
max_steps = 100              # maximum possible steps in an episode
batch_size = 64             

# parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True

### next class is like heart of code it is the thing which will train it self

class DQN():
    def __init__(self, state_size, action_size, learning_rate, name = "DQN"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")
            
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")
        
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            self.flatten = tf.layers.flatten(self.conv3_out)
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")
            
            
            self.output = tf.layers.dense(inputs = self.fc, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = 3, 
                                        activation=None)
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            
tf.reset_default_graph()
DQNet = DQN(state_size, action_size, learning_rate)


### it will store experience = (state,action, reward, next_state, done) for small interval
class Memory():
    def __init__(self,max_size):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, exp):
        self.buffer.append(exp)  # it will add experience to memory
        
    def sample(self, batch_size):
        buffer_size = len(self.buffer) # this function will give return a certain sample out of memory
        index = np.random.choice(np.arange(buffer_size), 
                                       size=batch_size,
                                       replace=False)
        return [self.buffer[i] for i in index]
        
memory = Memory(max_size=memory_size)

game.new_episode()

### it will 

for i in range(pretrain_length):
    if i==0:
        state = game.get_state().screen_buffer
        state,stacked_frames = stack_frames(stacked_frames, state, True)
    action = random.choice(possible_actions)
    reward = game.make_action(action)
    done = game.is_episode_finished()
    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))
        game.new_episode()
        state = game.get_state().screen_buffer
        state,stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        next_state = game.get_state().screen_buffer
        next_state,stacked_frames = stack_frames(stacked_frames, next_state, False)
        memory.add((state, action, reward, next_state, done))
        state = next_state
        
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

tf.summary.scalar("Loss", DQNet.loss)

write_op = tf.summary.merge_all()

def predict_action(explore_starting, explore_stop, decay_step, decay_rate, state, action):
    exp_tradeoff = np.random.rand()
    explore_prob = explore_stop+(explore_start-explore_stop)*np.exp(-decay_rate*decay_step)
    if explore_prob>exp_tradeoff:
        action = random.choice(possible_actions)
    else:
        qs = sess.run(DQNet.output, feed_dict = {DQNet.inputs_: state.reshape((1, *state.shape))})
        action = possible_actions[int(np.argmax(qs))]
    return action, explore_prob
    
saver = tf.train.Saver()
if training==True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        decay_step = 0
        game.init()
        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            while step<max_steps:
                step += 1
                decay_step += 1
                action, explore_probab = predict_action(explore_start, explore_stop, decay_step, decay_rate, state, action)
                reward = game.make_action(action)
                done = game.is_episode_finished()
                episode_rewards.append(reward)
                if done:
                    next_state = np.zeros((84,84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    step=max_steps
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probab))
                    memory.add((state, action, reward, next_state, done))

                else:
                    next_state = game.get_state().screen_buffer
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    memory.add((state, action, reward, next_state, done))
                    state = next_state


                batch = memory.sample(batch_size)
                state_sm = np.array([each[0] for each in batch], ndmin=3)
                action_sm = np.array([each[1] for each in batch])
                reward_sm = np.array([each[2] for each in batch])
                next_state_sm = np.array([each[3].reshape((84,84,4)) for each in batch], ndmin=3)
                done_sm = np.array([each[4] for each in batch])
                target_qs_sm = []
                qs_next_state = sess.run(DQNet.output, feed_dict={DQNet.inputs_:next_state_sm})

                for i in range(0, len(batch)):
                    if done_sm[i]:
                        target_qs_sm.append(reward_sm[i])
                    else:
                        target = reward_sm[i] + gamma*np.max(qs_next_state[i])
                        target_qs_sm.append(target)
                target_qs_batch = np.array([each  for each in target_qs_sm])

                loss, _ = sess.run([DQNet.loss, DQNet.optimizer],
                                    feed_dict={DQNet.inputs_: state_sm,
                                               DQNet.target_Q: target_qs_batch,
                                               DQNet.actions_: action_sm})
                summary = sess.run(write_op, feed_dict={DQNet.inputs_:state_sm, DQNet.actions_:action_sm, DQNet.target_Q:target_qs_batch})
                writer.add_summary(summary, episode)
                writer.flush()

            if episode%5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")


saver = tf.train.Saver()
with tf.Session() as sess:
    
    game, possible_actions = create_environment()
    
    totalScore = 0
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(10):
        
        done = False
        
        game.new_episode()
        
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
            
        while not game.is_episode_finished():
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQNet.output, feed_dict = {DQNet.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]
            
            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()
            
            if done:
                break  
                
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
                
        score = game.get_total_reward()
        print("Score: ", score)
    game.close()
