
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import vizdoom as vzd
from argparse import ArgumentParser
import time as time_

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 4

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

# TODO move to argparser
save_model = True
load_model = False
skip_learning = False

# Configuration file path
DEFAULT_MODEL_SAVEFILE = "/tmp/model"
DEFAULT_CONFIG = "../../scenarios/simpler_basic.cfg"


# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        # sample(population, k): Chooses k unique random elements from a population sequence or set.
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

class PER_Memory(object):

    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4

    PER_b_increment_per_sampling = 0.001
    size = 0
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):

        self.capacity = capacity
        self.tree = SumTree(capacity)

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.size += 1
        experience = [s1, action, s2, isterminal, reward]
        max_priority = self.tree.get_current_max_priority()

        if max_priority == 0:
            max_priority = PER_Memory.absolute_error_upper

        self.tree.add(max_priority, experience)

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += PER_Memory.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, PER_Memory.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def get_sample(self, batch_size):

        memory_minibatch = []

        b_idx, b_ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into batch_size ranges
        priority_segment = self.tree.total_priority / batch_size  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * batch_size) ** (-self.PER_b)

        for i in range(batch_size):

            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(batch_size * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            # print("experience shape:",np.array(experience).shape)
            # print("type(data)",type(data))
            # print("shape(data)",np.array(data).shape)
            # for xxx in data:
            #     print("\n   ",xxx)
            # print("--------------------")
            memory_minibatch.append(experience)

        # return memory_minibatch
        return b_idx, memory_minibatch, b_ISWeights
        # return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def get_current_max_priority(self):
        # return np.max(self.tree[:])
        # TODO: Can this be replaced by self.tree.max() ?
        return np.max(self.tree[-self.capacity:])

    """Here we add our priority score in the sumtree leaf and add the experience in data"""
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """
        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            # If we each bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node

def create_network(session, available_actions_count):
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")
    ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2_flat = tf.contrib.layers.flatten(conv2)
    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)

    # loss = tf.losses.mean_squared_error(q, target_q_)

    loss = tf.reduce_mean(ISWeights_ * tf.squared_difference(target_q_, q))

    abs_err = tf.abs(target_q_ - q)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q, ISWeights_mb):
        # feed_dict = { s1_: s1,
        #               target_q_: target_q}
        feed_dict = {s1_: s1,
                     target_q_: target_q,
                     # DQNetwork.actions_: actions_mb,
                     ISWeights_: ISWeights_mb}
        l, _, absolute_error = session.run([loss, train_step, abs_err], feed_dict=feed_dict)
        return l, _, absolute_error

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state})

    def function_get_best_action(state):
        feed_dict = {s1_: s1,
                     target_q_: target_q,
                     # DQNetwork.actions_: actions_mb,
                     ISWeights_: ISWeights_mb}
        return session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]

    return function_learn, function_get_q_values, function_simple_get_best_action

def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """
    # [s1, action, s2, isterminal, reward]

    # print("in learnfrommemory(). \tmemory.size:",memory.size," \tbatch size:",batch_size)

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        # s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        tree_idx, batch, ISWeights_mb = memory.get_sample(batch_size)
        batch = np.array(batch)
        # batch = [ [exp_1], [ exp_2], ... ]
        # batch.shape = (64, 1, 5)
        # batch[exp. number, 0, {1,...,5} ]

        s1 = np.zeros((batch_size,resolution[0],resolution[1],1))
        a = np.zeros((batch_size,))
        s2 = np.zeros((batch_size,resolution[0],resolution[1],1))
        isterminal = np.zeros((batch_size,))
        r = np.zeros((batch_size,))

        for b in range(batch_size):
            s1[b,:,:,0] = batch[b][0][0]
            a[b] = batch[b][0][1]
            s2[b,:,:,0] = batch[b][0][2]
            isterminal[b,] = batch[b][0][3]
            r[b,] = batch[b][0][4]


        # print(s1.shape)
        # print(a.shape)
        # print(s2.shape)
        # print(isterminal.shape)
        # print(r.shape)
        #
        # # (4, 30, 45, 1)
        # # (4,)
        # # (4, 30, 45, 1)
        # # (4,)
        # # (4,)

        # s1.shape          = (batch_size, 30, 45, 1)
        # a.shape           = (batch_size,)
        # s2.shape          = (batch_size, 30, 45, 1)
        # isterminal.shape  = (batch_size,)

        q2 = np.max(get_q_values(s2), axis=1)
        target_q = get_q_values(s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r

        a = a.astype(int)

        # np.arange(target_q.shape[0]): [0 1 2 3]
        # a: [0. 4. 3. 4.]
        # a = np.int
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        l, _, absolute_error_matrix = learn(s1, target_q, ISWeights_mb)
        absolute_error = np.zeros((batch_size,))

        # print("absolute_error_matrix",absolute_error_matrix)

        eps = .000001
        absolute_error_matrix[np.isnan(absolute_error_matrix)] = -eps
        average = np.sum(absolute_error_matrix) / batch_size

        for row in range(batch_size):
            if np.min(absolute_error_matrix[row]) < 0:
                absolute_error[row] = average
            else:
                absolute_error[row] = np.max(absolute_error_matrix[row])

        #  doesn't work
        # SPECIALNUM = .0001
        # absolute_error_matrix[np.isnan(absolute_error_matrix)] = SPECIALNUM
        # for row in range(batch_size):
        #
        #     absolute_error[row] = np.max(absolute_error_matrix[row])
        #
        #     if absolute_error[row] == SPECIALNUM:
        #
        #         if row >= 1:
        #             if not absolute_error[0] == SPECIALNUM:
        #                 absolute_error[row] = absolute_error[0]
        #         else:
        #             if not np.max(absolute_error_matrix[1]) == SPECIALNUM:
        #                 absolute_error[0] = absolute_error[1]
        #             elif not np.max(absolute_error_matrix[2]) == SPECIALNUM:
        #                 absolute_error[0] = absolute_error[2]
        #
        # print("absolute_error",absolute_error)

        # time_.sleep(.5)

        return tree_idx, absolute_error, True
    return None, None, False

# Called once per time step
def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    tree_idx, absolute_errors, update = learn_from_memory()

    if update:
        memory.batch_update(tree_idx, absolute_errors)


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':
    parser = ArgumentParser("ViZDoom example showing how to train a simple agent using simplified DQN.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    # Create Doom instance
    game = initialize_vizdoom(args.config)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    # memory = ReplayMemory(capacity=replay_memory_size)
    memory = PER_Memory(capacity=replay_memory_size)

    session = tf.Session()
    learn, get_q_values, get_best_action = create_network(session, len(actions))
    saver = tf.train.Saver()
    if load_model:
        print("Loading model from: ", DEFAULT_MODEL_SAVEFILE)
        saver.restore(session, DEFAULT_MODEL_SAVEFILE)
    else:
        init = tf.global_variables_initializer()
        session.run(init)
    print("Starting the training!")

    time_start = time()
    if not skip_learning:

        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
            # for learning_step in range(learning_steps_per_epoch):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)
            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()),
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = get_best_action(state)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", DEFAULT_MODEL_SAVEFILE)
            saver.save(session, DEFAULT_MODEL_SAVEFILE)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
