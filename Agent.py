import VDControl
import FACL
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# This is an agent class that is used to interact with the controller that gets created
# This class takes care of things like:
# - running training epochs and games
# - And plotting things like reward graphs and agent paths

class Agent:
    def __init__(self, controller):
        # default values
        self.training_iterations_max = 667 # number of iteration in 1 epoch
        self.game_iterations_max = 45
        self.controller = controller # this is the controller (FACL or FQL) that gets passed into the actor object
        self.success = 0 # this will count the number of sucesses (to be taken out later)
        self.record_success_flag = 0
        self.figure_number =1
        self.reward_total = []

    def run_one_epoch(self): # runs a single epoch of training
        # Reset the game
        self.controller.reset()
        # This function calls the controller iterator
        for i in range(self.training_iterations_max):
            self.controller.iterate_train()
            if (self.controller.state[0] >= 10): ##change to a check capture / completion function later
                self.success +=1
                break
        self.controller.updates_after_an_epoch()
        self.reward_total.append(self.reward_sum_for_a_single_epoch())

    def run_one_game(self):  # runs a single epoch of training
        # Reset the game
        self.controller.reset()
        # This function calls the controller iterator
        for i in range(self.game_iterations_max):
            self.controller.iterate_run()
            if (
                    self.controller.distance_from_target() < 1):  ##change to a check capture / completion function later
                self.success += 1
                # print('success -- end')
                break


        #print(self.controller.path)

    def save_epoch_training_info(self):
        # gonna call a controller function so that we know what to save
        self.controller.save()
        pass

    # plots the path of the agent taken
    def print_path(self):  # this function prints the path of the agent taken
        x = [0] * (len(self.controller.path) - 1)
        y = [0] * (len(self.controller.path) - 1)
        print('number of steps taken: ', len(self.controller.path))
        for i in range(len(self.controller.path) - 1):
            x[i] = self.controller.path[i][0]
            y[i] = self.controller.path[i][1]
        # plt.clf()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        # add circle
        circle = plt.Circle((self.controller.territory_coordinates[0], self.controller.territory_coordinates[1]),
                            self.controller.r, color='g', fill=False)
        plt.plot(self.controller.territory_coordinates[0], self.controller.territory_coordinates[1], 'ro')
        ax.add_patch(circle)
        plt.show()
        pass

    # called during training only
    # sums all the rewards obtained in 1 epoch of training
    def reward_sum_for_a_single_epoch(self):
        total_rewards = sum(self.controller.reward_track)
        #print(total_rewards)
        return total_rewards

    # prints a plot of the rewards obtained during each epoch
    def print_reward_graph(self):
        fig, ax = plt.subplots()
        plt.title('rewards')
        plt.xlabel('epoch')
        plt.ylabel('sum of rewards per epoch')
        ax.plot(self.reward_total)
        plt.show()
        pass
    def print_velocity_graph(self):
        fig, ax = plt.subplots()
        plt.title('velocity of agent')
        plt.xlabel('time')
        plt.ylabel('velocity')
        ax.plot(self.controller.velocity_path)
        plt.show()