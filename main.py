# Libraries and Classes
import numpy as np
import FACL
from Agent import Agent
from VDControl import VDControl
import time
import matplotlib.pyplot as plt
#
# This driver program is used for training an agent and then playing a series of games
# First the fuzzy inference (FIS) system is setup (number of membership functions etc)
# Then the type of training is selected: either fuzzy Q learning or fuzzy actor critic
# The controller is made and then plugged into an agent object. It is trained for a # of epochs
# Infomation like the time it took to train and the reward plot are shown after. Finally a new
# agent and controller object a made using the saved data and then they play a game 100 times, recording
# the outcome of each game (success of fail)
def plot_both_velocities():
    fig, ax = plt.subplots()
    plt.title('sharon and diana final epoch velocity')
    ax.plot(sharon.controller.velocity_path,label='sharon')
    ax.plot(diana.controller.velocity_path,label='diana')
    plt.xlabel('time (10ms)')
    plt.ylabel('velocity')
    plt.legend()
    plt.show()
# This driver program looks at

# General Fuzzy Parameters
state = [0.5,0.1]# start position on the grid. make random later
state_max = [1.5, 5] # max values of the grid [x,y]
state_min = [-1.5, -5] # smallest value of the grid [x,y]
num_of_mf = [9, 9] # breaking up the state space (grid in this case) into 29 membership functions
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
Sharon_FACLcontroller = VDControl(state, state_max, state_min, num_of_mf) #create the FACL controller
Diana_FACLcontroller = VDControl(state,state_max,state_min,num_of_mf)
sharon = Agent(Sharon_FACLcontroller) # create the agent with the above controller
diana = Agent(Diana_FACLcontroller)

#print out all the rule sets
print("rules:")
print(sharon.controller.rules)

for i in range(5000):
    sharon.controller.reset()
    diana.controller.reset()
    for j in range(sharon.training_iterations_max):
        # sharon.controller.iterate_train()
        # diana.controller.iterate_train()
        if (sharon.controller.state[0] < l/2  and sharon.controller.state[0]> -l/2):  ##if both havent crossed the finish line, train
            sharon.controller.generate_noise()
            diana.controller.generate_noise()

            # Step 3 :  calculate the necessary action
            sharon.controller.calculate_ut()
            diana.controller.calculate_ut()

            # Step 4: calculate the value function at current iterate/time step
            sharon.controller.v_t = sharon.controller.calculate_vt(sharon.controller.phi) #  v_t = sum of self.phi[l] * self.zeta[l]
            diana.controller.v_t = diana.controller.calculate_vt(diana.controller.phi)

            # Step 5: update the state of the system
            if(sharon.controller.u_t>1):
                sharon.controller.u_t = 1
            elif(sharon.controller.u_t<-1):
                sharon.controller.u_t = -1

            if (diana.controller.u_t > 1):
                diana.controller.u_t = 1
            elif (diana.controller.u_t < -1):
                diana.controller.u_t = -1
            a = -c/m * sharon.controller.state[1] + g*(sharon.controller.u_t - diana.controller.u_t)/l
            sharon.controller.state[1] = sharon.controller.state[1] + a*dt
            sharon.controller.state[0] = sharon.controller.state[0] + sharon.controller.state[1]*dt
            diana.controller.state[0] = sharon.controller.state[0]
            diana.controller.state[1] = sharon.controller.state[1]

            diana.controller.update_path([diana.controller.state[0]])
            sharon.controller.update_path([diana.controller.state[0]])
            diana.controller.update_v_path([diana.controller.state[1]])
            sharon.controller.update_v_path([sharon.controller.state[1]])
            diana.controller.update_input_array(diana.controller.u_t)
            sharon.controller.update_input_array(sharon.controller.u_t)

            sharon.controller.phi_next = sharon.controller.update_phi()

            diana.controller.phi_next = diana.controller.update_phi()

            # Step 6: get reward, this will be replaced for value decomp?
            sharon.controller.reward = sharon.controller.get_reward()
            diana.controller.reward = diana.controller.get_reward()


            # Step 7: Calculate the expected value for the next step
            sharon.controller.v_t_1 = sharon.controller.calculate_vt(sharon.controller.phi_next) # self.phi[l] * self.zeta[l]
            diana.controller.v_t_1 = diana.controller.calculate_vt(diana.controller.phi_next)

            # Step 8: calculate the temporal difference
            #No VD
            # sharon.controller.calculate_prediction_error()
            # diana.controller.calculate_prediction_error()

            # Regular VD
            sharon.controller.temporal_difference = (sharon.controller.reward+diana.controller.reward) + sharon.controller.gamma * (sharon.controller.v_t_1 + diana.controller.v_t_1) - (sharon.controller.v_t+diana.controller.v_t)
            diana.controller.temporal_difference = (diana.controller.reward+sharon.controller.reward) + diana.controller.gamma * (diana.controller.v_t_1+sharon.controller.v_t_1) - (diana.controller.v_t+sharon.controller.v_t)

            # A weighted version of VD
            # w=0.7
            # sharon.controller.temporal_difference = (
            #                                                     w*sharon.controller.reward + (1-w)*diana.controller.reward) + sharon.controller.gamma * (
            #                                                     w*sharon.controller.v_t_1 + (1-w)*diana.controller.v_t_1) - (
            #                                                     w*sharon.controller.v_t + (1-w)*diana.controller.v_t)
            # diana.controller.temporal_difference = (
            #                                                    w*diana.controller.reward + (1-w)*sharon.controller.reward) + diana.controller.gamma * (
            #                                                    w*diana.controller.v_t_1 + (1-w)*sharon.controller.v_t_1) - (
            #                                                    w*diana.controller.v_t + (1-w)*sharon.controller.v_t)

            # Step 9: update the actor and critic functions
            sharon.controller.update_zeta() # update the critic
            sharon.controller.update_omega() # update the actor
            diana.controller.update_zeta()  # update the critic
            diana.controller.update_omega()  # update the actor

            sharon.controller.phi = sharon.controller.phi_next
            diana.controller.phi = diana.controller.phi_next

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
        print('sigma ', diana.controller.sigma)
        print('cummulative fails ', failiure)
        print('cummulative success ', success)


        #print("input, ut:", sharon.controller.input)

end = time.time()
print('total train time : ', end-start)
print('total number of fails', failiure)
# Print the path that our agent sharon took in her last epoch
#print("xy path",sharon.controller.path) #numerical values of path
print("input, ut:" , sharon.controller.input)


#
# sharon.success = 0
# diana.success = 0
# number_of_ties=0
# sharon.controller.sigma = 0.15
# diana.controller.sigma = 0.15
# #Run a series of games
# for i in range(1000):
#     sharon.controller.reset()
#     diana.controller.reset()
#     for j in range(sharon.training_iterations_max):
#         # sharon.controller.iterate_train()
#         # diana.controller.iterate_train()
#         if (sharon.controller.state[0] < sharon.controller.finish_line and diana.controller.state[0] <  diana.controller.finish_line):  ##if both havent crossed the finish line, train
#             sharon.controller.generate_noise()
#             diana.controller.generate_noise()
#             # Step 3 :  calculate the necessary action
#             sharon.controller.calculate_ut()
#             diana.controller.calculate_ut()
#
#             # Step 5: update the state of the system
#             sharon.controller.update_state()
#             diana.controller.update_state()
#             sharon.controller.phi_next = sharon.controller.update_phi()
#             sharon.controller.phi = sharon.controller.phi_next
#             diana.controller.phi_next = diana.controller.update_phi()
#             diana.controller.phi = diana.controller.phi_next
#         else:  # if an agent has crossed the line
#             if (sharon.controller.state[0] >= sharon.controller.finish_line and diana.controller.state[
#                 0] >= diana.controller.finish_line):
#                 number_of_ties += 1
#                 # plot_both_velocities()
#                 break
#             if (sharon.controller.state[0] >= sharon.controller.finish_line):
#                 sharon.success += 1
#                 if (diana.controller.state[0] >= diana.controller.finish_line - 1):
#                     sharon.success -= 1
#                     number_of_ties += 1
#                     # plot_both_velocities()
#                 break
#             if (diana.controller.state[0] >= diana.controller.finish_line):
#                 diana.success += 1
#                 if (sharon.controller.state[0] >= sharon.controller.finish_line - 1):
#                     diana.success -= 1
#                     number_of_ties += 1
#                     # plot_both_velocities()
#                 break


sharon.save_epoch_training_info() #save all the important info from our training sesh



#Print out the reward plots combined
fig, ax = plt.subplots()
plt.title('sharon and diana rewards per epoch')
ax.plot(sharon.reward_total,label='sharon')
ax.plot(diana.reward_total,label='diana')
plt.xlabel('epoch')
plt.ylabel('total rewards per epoch')
plt.legend()
plt.show()
#
# sharon.print_reward_graph()
# diana.print_reward_graph()

plot_both_velocities()
# sharon.print_velocity_graph()
# diana.print_velocity_graph()

fig, ax = plt.subplots()
plt.title('path of ball')
ax.plot(sharon.controller.path,label='path of ball')
plt.xlabel('iteration')
plt.ylabel('position')
plt.legend()
plt.show()
