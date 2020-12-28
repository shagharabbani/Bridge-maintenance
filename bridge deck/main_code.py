import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import os, time, copy
from bridgedeterioration import BridgeEnv
from policynet import CNNPolicy
from utils import getreturn, experience_buffer, processsa
# from pretrain import  pretrain
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
env = BridgeEnv()
start = time.time()

batch_size = 1000; update_freq = 1
startE = 1.0; endE = 0.01; anneling_steps = int(1e2); e_step = (startE-endE)/anneling_steps; e = startE
num_episodes = int(1e4); pre_train_steps = int(1e4)
loadmodel = False


tf.reset_default_graph()
tf.disable_eager_execution()

#mH =Fully connected, gamma= discount factor, alpha= learning rate
mH = 128; gamma = 1.0; alpha = 1.0
mainQN = CNNPolicy(); mainQN.create_network(mH)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1000)
myBuffer = experience_buffer(pre_train_steps)

costs = [0]; DQNloss = [0]; Q_dict = {}
if not os.path.exists('./result'):
    os.mkdir('./result')
networkmodel = './result/training result/'

# guidebuffer = pretrain(gamma)
sess = tf.Session()
sess.run(init)

if loadmodel:
    ckpt = tf.train.get_checkpoint_state(networkmodel)
    saver.restore(sess, ckpt.model_checkpoint_path)

# MC simulation
for i_episode in range(num_episodes+1):
    episodeBuffer = experience_buffer(pre_train_steps)

    state = env.reset()
    s = np.reshape(state,[49])
    done = False
    rAll = 0; t = 0
    states = np.zeros([100,49])
    actions = np.zeros([100,7])
    codeactions = np.zeros([100,28])
    states1 = np.zeros([100,49])
    rewards = np.zeros([100,7])
    dones = np.zeros([100,1])
    rnd = np.random.rand()

    while not done:
        states[t] = copy.deepcopy(s)

        # Preallocating the action vector (1x7).
        a = np.zeros(7,dtype=np.int32)
        if rnd < e:
            # Generating a random action for each 7 component (1x7 vector, each entry is 0, 1, 2, or 3).
            a = np.random.randint(0,4,7)
        else:
            a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[s]})[0]

        # Getting the new state and the corresponding reward of action 'a' and determining if a terminal node is reached.
        state1, reward, done = env.step(a)
        # Normalizing the reward vector (1x7)
        r = reward/600
        # Converting the state matrix (7x7) to a row vector (1x49)
        s1 = np.reshape(state1,[49])
        # Converting the binary action matrix (7x4) to a row vector (1x28)
        code_a = np.reshape(processsa(a,4),[28])

        # Storing the data at each time in a variable
        states1[t] = copy.deepcopy(s1); rewards[t] = copy.deepcopy(r); actions[t] = copy.deepcopy(a)
        codeactions[t] = copy.deepcopy(code_a); dones[t] = copy.deepcopy(done)
        # Updating the state and time for next year after applying the actions
        s = s1; t+=1
    # Getting the return G (a parameter that balances the short- and long-term rewards). Refer to Eq. 1 in the paper.
    G = getreturn(rewards,gamma=gamma)
    # Storing the reward vectors of each year in a single numpy array.
    costs.append(np.sum(rewards))
    if i_episode%100==0:
        print(str(i_episode), '***********************', costs[-1],  DQNloss[-1], '***********************', time.time() - start)

    for i in range(100):
        # Getting the state, action (decimal and binary), value function (G), terminal node condition, and reward at time i. AND...
        # Getting the state at the next time step
        s = states[i]; a = actions[i]; Q = G[i]; s1 = states1[i]; d = dones[i]; code_a = codeactions[i]; r = rewards[i]
        # Converting back the state from a row vector (1x49) to a square matirx (7x7)
        state = np.int32(np.reshape(s,[7,7]))
        # state_num is a (1x7) vector which contains the condition of each component at time i.
        _, state_num = np.where(state[:,0:6]==1)
        # Converting action vector 'a' to int32 data type.
        a = np.int32(a)
        
        for component in range(7):
            # Creating a tuple for tracking the number of times a state (i.e. node) is visited.
            tuple_idx = (component, state_num[component], a[component], i)
            # Have we seen this situation before?
            # If yes:
            if Q_dict.get(tuple_idx):
                # Getting n_times and Q_value from the previous time that this exact combination is visited.
                n_times,Q_value = Q_dict[tuple_idx]
                # Incrementing n_times
                n_times += 1
                #############################################################################
                # Updating the Q_value based on the difference between the new and old values beased on Eq. 2 in the paper.
                Q_value += (Q[component]-Q_value)*alpha
                #############################################################################
                # Storing n_times and Q_value for each tuple in a dictionary 'Q_dict'
                Q_dict[tuple_idx] = (n_times, Q_value)
            # If no:
            else:
                # This is the first time we face this situation. So the number of times should be zero.
                n_times = 0
                # The reward corresponding to 'component' at state 'state_num[component]' with action 'a[component]' at time i
                Q_value = Q[component]
                # Storing n_times and Q_value for each tuple in a dictionary 'Q_dict'
                Q_dict[tuple_idx] = (n_times, Q_value)
            # Updating the new Q_value (if it has not been visited, nothing will be updated.)
            Q[component] = Q_value
        episodeBuffer.add(np.reshape(np.array([s,a,Q,r,s1,d,code_a]),[1,7]))
    myBuffer.add(episodeBuffer.buffer)

    if i_episode >100 and (i_episode%update_freq == 0):
        if e>endE:
            e -= e_step
        trainBatch = myBuffer.sample(batch_size)
        input_s = np.vstack(trainBatch[:,0])
        input_a = np.vstack(trainBatch[:,1])
        target_Q = np.vstack(trainBatch[:,2])

        _, qloss = sess.run([mainQN.updateModel,mainQN.loss],
                 feed_dict={mainQN.scalarInput:input_s, mainQN.actions:input_a, mainQN.targetQ:target_Q})
        DQNloss.append(qloss)

    if i_episode%5000 == 0:
        print(rnd>e)
        filepath = 'bridge deck/result/training results/step' + str(i_episode)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        saver.save(sess, filepath+'/training-'+str(i_episode)+'.cpkt')
        np.save(filepath+'/costs.npy', costs)
        np.save(filepath+'/DQNloss.npy', DQNloss)
        np.save(filepath+'/Q_dict.npt',Q_dict)
        print("Save Model")
        elapsed = time.time()-start
        print(i_episode,e,elapsed,costs[-1])
        state,_ = env.randomint()
        print(state)
        s = np.reshape(state,[49])
        print(sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0])
        start = time.time()
sess.close()

