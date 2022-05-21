import numpy as np
import matplotlib as plt
import abc
# This class is an abstract base class that implements the Fuzzy Actor Critic Algorithm
# Actor controllers are meant to inherit all of it's properties and use polymorphism for the following methods:
# get_reward, update_state
# At the bottom is a function called "iterate" which does 1 iteration of the FACL algorithm (train or run)


# Constructor
class FACL:
    def __init__(self, stateMax : list, stateMin : list, numMF : list):
        self.alpha = 0.1  # critic learning rate
        self.beta = 0.05  # actor learning rate
        self.gamma = 0.9  # discount factor
        self.L = int(np.prod(numMF) ) # total number of rules
        self.zeta = np.zeros(self.L)  # critic
        self.omega = np.zeros(self.L)  # actor
        self.rules = np.zeros(self.L) #saving a space for rules
        self.u_t = float(0) # action / heading angle
        self.reward = float(0) # gets overridden by the child class
        self.v_t = float(0)  # value function
        self.v_t_1 = float(0)  # value function at next time step
        self.temporal_difference = float(0)
        self.indices_of_firing_rules = [] # record the index of the rules that are non zero, implement later
        self.noise = float(0)
        self.sigma = 0.50 # standard dev of the noise, shrinks with each iteration
        # create the fuzzy rules
        self.rule_creation(stateMax, stateMin, numMF)
        self.phi = self.update_phi() #set phi (phi = rules that are firing)
        self.phi_next = np.zeros(self.L) #gets calculated in the loop
        pass

    # This function creates all the fuzzy rules given the state minimums, state maximums and number of MF
    # For example, max may look like [50,50], min is [-50,-50] and the number of MF/divisions is [9,9]
    # This means 9*9 rules are created, and -50 to 50 is divided into those 9 segements
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
            boundary_array = self.calculate_boundary_values(sMax, sMin, nMf) # divides up the state space and records the values
            triangle_matrix.append(self.create_triangular_sets(nMf, boundary_array)) # creates the triangular values for the Membership Function
            pass

        combinations = []
        # This section simply creates the rule sets from all the differet possible state spaces that are passed in
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

    # A function used in "rule_creation", here we are creating all the different sets of rules that exists in terms
    # of the 3 points on the triangle, read below
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

    # Update the critic weights
    def update_zeta(self) -> None: #Critic
        # for l in range(self.L):
        #     self.zeta[l] += self.alpha * self.temporal_difference * self.phi[l]
        #     pass
        self.zeta = self.zeta + self.alpha * np.multiply(self.temporal_difference,self.phi)

    # Update the actor weights
    def update_omega(self) -> None: #Actor
        # for l in range(self.L):
        #     self.omega[l] = self.omega[l] + self.beta * self.phi[l] * self.noise*self.temporal_difference
        #     pass
        self.omega = self.omega + self.beta * self.noise * np.multiply(self.temporal_difference,self.phi)
        pass

    # Calculate phi, which is the firing strength for the rules at a given iteration
    def update_phi(self) -> None: # finds the rules that get fired
        rules_firing = [[0] * len(self.state) for _ in range(self.L)]  # np.zeros((self.L, len(state)))
        product_of_rule = [1] * self.L
        for l in range(self.L):
            for i in range(0, len(self.state)):
                rules_firing[l][i] = self.mu(self.state[i], [self.rules[l][i][0], self.rules[l][i][1], self.rules[l][i][2]])
                if(rules_firing[l][i] != 0 ):
                    product_of_rule[l] = product_of_rule[l] * rules_firing[l][i] # gets the product of all the states that went thru the fuzzy MF for a specific rule
                else:
                    product_of_rule[l] = 0
        # Sum all the array values of the products for all rules
        #sum_of_rules_fired = sum(product_of_rule)

        # Calculate phi^l
        # phi = np.zeros(self.L)
        # for l in range(self.L):
        #     phi[l] = product_of_rule[l] #/ sum_of_rules_fired

        phi = product_of_rule.copy() # divided by the sum of rules fired
                                     # but since the mf are triangular
                                     # the sum is always 1
        product_of_rule = None # delete the array just in case
        return phi
        pass

    # Calculate the value function
    def calculate_vt(self, phi): #Value Function
        # v_t = 0.0
        # for l in range(self.L):
        #     v_t = v_t + phi[l] * self.zeta[l]
        v_t = np.sum(np.multiply(phi,self.zeta))
        return v_t

    # Calculate the action
    def calculate_ut(self) -> None: # Calculates u_t, the action
        self.u_t = np.sum(np.multiply(self.phi,self.omega))+self.noise

    # calculate the temporal difference / prediction error
    def calculate_prediction_error(self):
        self.temporal_difference = self.reward + self.gamma * self.v_t_1 - self.v_t

    # Method is abstract for a controller class to overwrite it
    @abc.abstractmethod
    def update_state(self):
        pass

    # Method is abstract for a controller class to overwrite it
    @abc.abstractmethod
    def get_reward(self):
        pass

    # Triangular membership function mu
    # Used to determine which rules fire
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
        #print(f)
        return f

    # Noise generation for exploration
    def generate_noise(self):
        self.noise = np.random.normal(0, self.sigma)

    # This function is used for a single full iteration during TRAINING
    # Print statements commented out for debugging
    def iterate_train(self) :
        #print statements were added for debugging
        self.generate_noise()
        #print('noise : ', self.noise)
        # Step 3 :  calculate the necessary action
        #print('action')
        self.calculate_ut()
        #print(self.u_t)

        # Step 4: calculate the value function at current iterate/time step
        self.v_t = self.calculate_vt(self.phi) #  v_t = sum of self.phi[l] * self.zeta[l]
        #print('v_t')
        #print(self.v_t)

        # Step 5: update the state of the system
        self.update_state()
        self.phi_next = self.update_phi()
        #print('new state')
        #print(self.state)
        #print('current phi')
        #print(self.phi)
        #print('next phi')
        #print(self.phi_next)

        # Step 6: get the reward for this iteration
        self.reward = self.get_reward()
        #print('reward')
        #print(self.reward)

        # Step 7: Calculate the expected value for the next step
        self.v_t_1 = self.calculate_vt(self.phi_next) # self.phi[l] * self.zeta[l]
        #print('v_t_1')
        #print(self.v_t_1)
        # Step 8: calculate the temporal difference
        self.calculate_prediction_error()
        #print('temporal_difference')
        #print(self.temporal_difference)

        # Step 9: update the actor and critic functions
        self.update_zeta() # update the critic
        self.update_omega() # update the actor
        #print('weights w')
        #print(self.omega)
        #print('critic zeta')
        #print(self.zeta)
        self.phi = self.phi_next # the rules that fire for the next iteration
    pass

    # After 1 epoch of training, the standard deviation of the noise, actor and critic learning rates
    # are decreased a little
    def updates_after_an_epoch(self):
        self.sigma = 0.9999 * self.sigma
        # self.alpha = 0.99999 * self.alpha
        # self.beta = 0.999999 * self.beta
        pass
        # self.sigma = self.sigma*10**(np.log10(0.1)/1000)
        # self.alpha = self.alpha * 10 ** (np.log10(0.1) / 1000)
        # self.beta = self.beta * 10 ** (np.log10(0.1) / 1000)

    # This function is used for a single full iteration during a formal run
    # No actor critic updates are done since it is not training
    def iterate_run(self):
        #generate noise
        #select action
        self.calculate_ut()
        #update states
        self.update_state()
        self.phi_next = self.update_phi()
        self.phi = self.phi_next

        pass

