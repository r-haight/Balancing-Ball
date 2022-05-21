import numpy as np
import matplotlib as plt
import abc
import random

class FQL :
    def __init__(self, actions : list, statemax : list, statemin : list, numMF : list):

        self.gamma = 0.9  # discount factor
        self.L = np.prod(numMF)  # number of rules
        self.rules = np.zeros(self.L)
        self.u_t = float(0) # action
        self.reward = float(0) # gets overridden by the child class
        self.alpha = 0.05
        self.E = float(0)
        self.num_of_actions = len(actions)
        self.action = np.zeros(self.L)
        self.index_of_selected_action = np.zeros(self.L)
        self.greedy_factor = 0.5
        self.action_space = actions
        self.Q_function = float(0)  #
        self.q_table = np.zeros((self.L, self.num_of_actions)) #row = rules, column action
        self.indices_of_firing_rules = [] # record the index of the rules that are non zero
        # create the fuzzy rules
        self.rule_creation(statemax, statemin, numMF)
        self.phi = self.update_phi()  # set phi
        self.phi_next = np.zeros(self.L)  # gets calculated in the loop

    def rule_creation(self, state_max: list, state_min: list, number_of_mf: list) -> list:
        """
        :param state_max: f
        :param state_min:f
        :param number_of_mf:f
        :return:e
        """
        # get triangle mf points for all the states
        triangle_matrix = []
        for sMax, sMin, nMf in zip(state_max, state_min, number_of_mf):
            boundary_array = self.calculate_boundary_values(sMax, sMin, nMf)
            triangle_matrix.append(self.create_triangular_sets(nMf, boundary_array))
            pass

        combinations = []
        iterator = [0] * (len(triangle_matrix) + 1)  # set all iterators to 0

        while iterator[-1] == 0 and len(triangle_matrix) != 0:  # loop through each combination iterator
            triangles = [triangle_row[it] for it, triangle_row in zip(iterator, triangle_matrix)]
            combinations.append(triangles)
            # increment iterators
            iterator[0] += 1
            for index in range(len(iterator) - 1):
                if iterator[index] >= len(triangle_matrix[index]):
                    iterator[index] = 0
                    iterator[index + 1] += 1  # ripple iterator
                else:
                    break
                pass
            pass
        self.rules = combinations

        pass


    def calculate_boundary_values(self, state_max: float, state_min: float, num_of_mf: int) -> list:
        """
        :param state_max: value that defines the max state we go to for the fuzzy system
        :param state_min: value that defines the min state we go to for the fuzzy system
        :param num_of_mf: number of membership functions we want to divide the state space into
        :return:
        """
        # example: state_max = 100, state_min = 0, num_of_mf = 4 -> the output of values would numbers
        # [ 0 20 40 60 80 100] so that we can make up the triangle membership functions out of it

        gap_size = (state_max - state_min) / (num_of_mf + 1)
        b = [0] * int(((abs(state_min) + abs(state_max)) / gap_size) + 1)
        for i in range(int(((abs(state_min) + abs(state_max)) / gap_size) + 1)):
            b[i] = state_min + gap_size * i
        return b

    def create_triangular_sets(self, num_of_mf: int, boundary_values: list) -> list:
        # example : if we have a boundary_value array, than we can create sets of 3 points for our triangular MFs
        # using output from the previous boundary_values would be:
        # [[0 20 40], [20 40 60], [40 60 80], [60 80 100]]
        state_rules = np.zeros((num_of_mf, 3))
        for i in range(num_of_mf):
            state_rules[i][0] = boundary_values[i]
            state_rules[i][1] = boundary_values[i+1]
            state_rules[i][2] = boundary_values[i+2]
        return state_rules
    def mu(self, state: float, rule: list): # This is the triangular membership function
        #print(state)
        #print(rule)
        if state <= rule[0]:
            f = 0
        elif state > rule[0] and state <= rule[1]:
            f = (state - rule[0]) / (rule[1] - rule[0])
        elif rule[1] < state and state < rule[2]:
            f = (rule[2] - state) / (rule[2] - rule[1])
        elif state >= rule[2]:
            f = 0
        return f

    def select_action(self):

        # roll a number between 0 and 1 to figure out what kind of action to select
        # for l in range(self.L):
        #     n = random.random()
        #     if(n>self.greedy_factor):
        #     # if greater than the greedy factor, then choose a random action for all the rules?
        #         self.action[l] = random.choice(self.action_space)
        #         self.index_of_selected_action[l] = self.action_space.index(self.action[l])
        #     else:
        #         q_values_for_rule = self.q_table[l][:]
        #         max_val_index = np.argmax(q_values_for_rule)
        #         self.index_of_selected_action[l] = max_val_index
        #         self.action[l] = self.action_space[max_val_index]
        n = random.random()
        if (n > self.greedy_factor):
            self.action = random.choices(self.action_space, k = self.L)
            for l in range(self.L):
                self.index_of_selected_action[l] = self.action_space.index(self.action[l])
        else:
            self.index_of_selected_action = np.argmax(self.q_table, axis=1)
            for l in range(self.L):
                 self.action[l] = self.action_space[self.index_of_selected_action[l]]




    def calculate_Q(self):
        Q = float(0)
        for l in range(int(self.L)):
            Q = Q + self.phi[l] * self.q_table[l][int(self.index_of_selected_action[l])]
        self.Q_function = Q


    def calculate_Q_star(self):
        Q = float(0)
        for l in range(self.L):
            Q = Q + self.phi_next[l] * max(self.q_table[l][:])
        self.Q_star = Q
        # max_q_vals = np.argmax(self.q_table,axis=1)
        # self.Q_star=np.sum(np.multiply(max_q_vals,self.phi_next))

    def update_q_table(self):
        #print("learn rate",self.alpha)
        #print("temp diff ", self.temporal_difference)
        #print("phi : ", self.phi)

        for l in range(self.L):
            self.q_table[l][int(self.index_of_selected_action[l])] = self.q_table[l][int(self.index_of_selected_action[l])] + self.alpha * self.E * self.phi[l]


    def update_phi(self) -> None:
        # rules_firing = [[0] * len(self.state) for _ in range(self.L)]  # np.zeros((self.L, len(state)))
        # product_of_rule = [0] * self.L
        # for l in range(self.L):
        #     product_of_rule[l] = 1
        #     for i in range(0, len(self.state)):
        #         rules_firing[l][i] = self.mu(self.state[i], [self.rules[l][i][0], self.rules[l][i][1], self.rules[l][i][2]])
        #         product_of_rule[l] = product_of_rule[l] * rules_firing[l][i]
        #
        # # Sum all the array values of the products for all rules
        # sum_of_rules_fired = sum(product_of_rule)
        rules_firing = [[0] * len(self.state) for _ in range(self.L)]  # np.zeros((self.L, len(state)))
        product_of_rule = [1] * self.L
        for l in range(self.L):
            for i in range(0, len(self.state)):
                rules_firing[l][i] = self.mu(self.state[i],
                                             [self.rules[l][i][0], self.rules[l][i][1], self.rules[l][i][2]])
                if (rules_firing[l][i] != 0):
                    product_of_rule[l] = product_of_rule[l] * rules_firing[l][
                        i]  # gets the product of all the states that went thru the fuzzy MF for a specific rule
                else:
                    product_of_rule[l] = 0
        # Calculate phi^l
        # phi = np.zeros(self.L)
        # for l in range(self.L):
        #     phi[l] = product_of_rule[l] / sum_of_rules_fired
        phi = product_of_rule.copy()
        return phi
        pass

    def calculate_ut(self, phi):
        # u_t = float(0)
        # for l in range(self.L):
        #     u_t = u_t + phi[l]*self.action[l]
        # self.u_t = u_t
        self.u_t = np.sum(np.multiply(phi,self.action))
        pass

    def calculate_temporal_difference(self):
        self.temporal_difference = self.reward + self.gamma * self.Q_star - self.Q_function
        self.E = self.temporal_difference
        #print("temp diff = ", self.reward, '+', self.gamma, "*", self.Q_star, '-', self.Q_function, '=', self.temporal_difference)
        pass

    @abc.abstractmethod
    def update_state(self):
        pass

    @abc.abstractmethod
    def get_reward(self):
        pass

    def iterate_train(self):
        # STEP 1: select the action of each rule
        self.select_action()

        # STEP 2: calculate phi
        self.phi = self.update_phi()

        # STEP 3: calulate the output of the fuzzy system, U_t
        self.calculate_ut(self.phi)

        #STEP 4: calculate the Q function
        self.calculate_Q()

        #STEP 5 : update the state
        self.update_state()
        self.phi_next = self.update_phi()

        #STEP 6: Get reward
        self.reward = self.get_reward()

        # STEP 7 : Calculate global Q max
        self.calculate_Q_star()

        # STEP 8: calculate the temporal difference
        self.calculate_temporal_difference()

        #STEP 9: update the q table
        self.update_q_table()


    def updates_after_an_epoch(self):
        self.alpha = 0.9999 * self.alpha
        if(self.greedy_factor > 0.99):
            pass
        else:
            self.greedy_factor = 1.001*self.greedy_factor

    def iterate_run(self):
        # STEP 1: select the action of each rule
        self.select_action()

        # STEP 2: calculate phi
        self.phi = self.update_phi()

        # STEP 3: calulate the output of the fuzzy system, U_t
        self.calculate_ut(self.phi)

        # STEP 4 : update the state
        self.update_state()
        pass