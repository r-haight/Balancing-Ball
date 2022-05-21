# Libraries and Classes
import numpy as np
from Agent import Agent
from FuzzyQVD import FuzzyQVD
import time
import matplotlib.pyplot as plt


def plot_both_velocities():
    fig, ax = plt.subplots()
    plt.title('sharon and diana final epoch velocity')
    ax.plot(sharon.controller.velocity_path,label='sharon')
    ax.plot(diana.controller.velocity_path,label='diana')
    plt.xlabel('time (10ms)')
    plt.ylabel('velocity')
    plt.legend()
    plt.show()


# General Fuzzy Parameters
state = [0.5,0.1]# start position on the grid. make random later
state_max = [1.5, 5] # max values of the grid [x,y]
state_min = [-1.5, -5] # smallest value of the grid [x,y]
num_of_mf = [9, 9] # breaking up the state space (grid in this case) into 29 membership functions
action_space = [-1, -0.866, -0.733, -0.600, -0.466, -0.3, -0.2, 0, 0.06, 0.2, 0.3, 0.466, 0.600, 0.733, 0.866, 1]

failiure = 0
success = 0
# system dynamics
m = 0.5
l = 2
g = 9.81
c = 0.01
dt = 0.03
########## TRAINING SECTION ###############
# two agents: sharon and diane

start = time.time() # used to see how long the training time took
Sharon_FQLcontroller = FuzzyQVD(state, state_max, state_min, num_of_mf, action_space) #create the FACL controller
Diana_FQLcontroller = FuzzyQVD(state,state_max,state_min,num_of_mf,action_space)
sharon = Agent(Sharon_FQLcontroller) # create the agent with the above controller
diana = Agent(Diana_FQLcontroller)

#print out all the rule sets
print("rules:")
print(sharon.controller.rules)

for i in range(300):
    sharon.controller.reset()
    diana.controller.reset()
    for j in range(sharon.training_iterations_max):
        # sharon.controller.iterate_train()
        # diana.controller.iterate_train()
        if (sharon.controller.state[0] < l/2  and sharon.controller.state[0]> -l/2):  ##if both havent crossed the finish line, train
            ##################
            # STEP 1: select the action of each rule
            sharon.controller.select_action()
            diana.controller.select_action()

            # STEP 2: calculate phi
            sharon.controller.phi = sharon.controller.update_phi()
            diana.controller.phi = diana.controller.update_phi()

            # STEP 3: calulate the output of the fuzzy system, U_t
            sharon.controller.calculate_ut(sharon.controller.phi)
            diana.controller.calculate_ut(diana.controller.phi)

            # STEP 4: calculate the Q function
            sharon.controller.calculate_Q()
            diana.controller.calculate_Q()

            # STEP 5 : update the state
            a = -c / m * sharon.controller.state[1] + g * (sharon.controller.u_t - diana.controller.u_t) / l
            sharon.controller.state[1] = sharon.controller.state[1] + a * dt
            sharon.controller.state[0] = sharon.controller.state[0] + sharon.controller.state[1] * dt
            diana.controller.state[0] = sharon.controller.state[0]
            diana.controller.state[1] = sharon.controller.state[1]
            sharon.controller.phi_next = sharon.controller.update_phi()
            diana.controller.phi_next = diana.controller.update_phi()

            # Some variables that get updated so we can see some nice graphs
            diana.controller.update_path([diana.controller.state[0]])
            sharon.controller.update_path([diana.controller.state[0]])
            diana.controller.update_v_path([diana.controller.state[1]])
            sharon.controller.update_v_path([sharon.controller.state[1]])
            diana.controller.update_input_array(diana.controller.u_t)
            sharon.controller.update_input_array(sharon.controller.u_t)


            # STEP 6: Get reward
            sharon.controller.reward = sharon.controller.get_reward()
            diana.controller.reward = diana.controller.get_reward()

            # STEP 7 : Calculate global Q max
            sharon.controller.calculate_Q_star()
            diana.controller.calculate_Q_star()

            # STEP 8: calculate the temporal difference
            #Regular TD
            sharon.controller.calculate_temporal_difference()
            diana.controller.calculate_temporal_difference()

            #Value decomp TD
            # sharon.controller.temporal_difference = (sharon.controller.reward+diana.controller.reward) + sharon.controller.gamma * (sharon.controller.Q_star +diana.controller.Q_star) - (sharon.controller.Q_function+ diana.controller.Q_function)
            # sharon.controller.E = sharon.controller.temporal_difference
            # diana.controller.temporal_difference = (diana.controller.reward+sharon.controller.reward) + sharon.controller.gamma * (
            #             sharon.controller.Q_star + diana.controller.Q_star) - (
            #                                                     sharon.controller.Q_function + diana.controller.Q_function)
            # diana.controller.E = diana.controller.temporal_difference

            #Weight TD
            # w = 0.7
            # sharon.controller.temporal_difference = (w*sharon.controller.reward + (1-w)*diana.controller.reward) + sharon.controller.gamma * (
            #                                                     w*sharon.controller.Q_star + (1-w)*diana.controller.Q_star) - (
            #                                                     w*sharon.controller.Q_function + (1-w)*diana.controller.Q_function)
            # sharon.controller.E = sharon.controller.temporal_difference
            # diana.controller.temporal_difference = (
            #                                                    w*diana.controller.reward + (1-w)*sharon.controller.reward) + sharon.controller.gamma * (
            #                                                (1-w)*sharon.controller.Q_star + w*diana.controller.Q_star) - (
            #                                                (1-w)*sharon.controller.Q_function + w*diana.controller.Q_function)
            # diana.controller.E = diana.controller.temporal_difference

            # STEP 9: update the q table
            sharon.controller.update_q_table()
            diana.controller.update_q_table()
            ##################

            if(j>=666): # it stayed on the table for the entire game
                success+=1
                break

        else: #if the balls rolls off the table
            failiure+=1
            break


    sharon.controller.updates_after_an_epoch()
    sharon.reward_total.append(sharon.reward_sum_for_a_single_epoch())
    diana.controller.updates_after_an_epoch()
    diana.reward_total.append(diana.reward_sum_for_a_single_epoch())

    # print out some stats as it trains every so often
    if (i % 100 == 0):
        print(i)
        print("time:", time.time()-start)
        # print("xy path of sharon",sharon.controller.path) #numerical values of path
        #print("xy path of diana", diana.controller.path)  # numerical values of path
        print('length of game', len(diana.controller.path))

        print('cummulative fails ', failiure)
        print('cummulative success ', success)


        #print("input, ut:", sharon.controller.input)

end = time.time()
print('total train time : ', end-start)
print('total number of fails', failiure)
# Print the path that our agent sharon took in her last epoch
#print("xy path",sharon.controller.path) #numerical values of path
print("input, ut:" , sharon.controller.input)




#Print out the reward plots combined
fig, ax = plt.subplots()
plt.title('sharon and diana rewards per epoch')
ax.plot(sharon.reward_total,label='sharon')
ax.plot(diana.reward_total,label='diana')
plt.xlabel('epoch')
plt.ylabel('total rewards per epoch')
plt.legend()
plt.show()

plot_both_velocities()

fig, ax = plt.subplots()
plt.title('path of ball')
ax.plot(sharon.controller.path,label='path of ball')
plt.xlabel('iteration')
plt.ylabel('position')
plt.legend()
plt.show()

fig, ax = plt.subplots()
plt.title('rewards in 1 training epoch')
ax.plot(sharon.controller.reward_track, label='reward at each time step')
plt.xlabel('iteration')
plt.ylabel('reward')
plt.legend()
plt.show()