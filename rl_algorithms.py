import numpy as np
import matplotlib.pyplot as plt

#p = 0.75
#gamma = 0.3

class GridWorld(object):
    
    def __init__(self, p):
        
        ### Attributes defining the Gridworld #######

        # Shape of the gridworld
        self.shape = (6, 6)
        
        # Locations of the obstacles
        self.obstacle_locs = [(1,1), (2,3), (2,5), (3,1), (4,1), (4,2), (4, 4)]
        
        # necessary as the order is quite chaotic
        self.state_list = [12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 19, 20, 5, 6, 7, 21, 8, 9, 10, 22, 23, 11, 24, 25, 26, 27, 28, 29, 30]
        
        # Locations for the absorbing states
        self.absorbing_locs = [(1,2), (4,3)]
        
        # Rewards for each of the absorbing states 
        self.special_rewards = [10, -100] # Corresponds to each of the absorbing_locs
        
        self.risky_states = [17, 26]

        # Reward for all the other states
        self.default_reward = 0
        
        # Starting location
        self.starting_loc = (0, 0)
        
        # Action names
        self.action_names = ['N','E','S','W'] # Action 0 is 'N', 1 is 'E' and so on
        
        # Number of actions
        self.action_size = len(self.action_names)

        self.action_space = [i for i in range(0, self.action_size)]
        
        # Randomizing action results: [1 0 0 0] to no Noise in the action results.
        self.action_randomizing_array = [p, (1-p)/3, (1-p)/3, (1-p)/3]
        
        ############################################
    

        #### Internal State  ####
        
        # Get attributes defining the world
        state_size, T, R, absorbing, locs = self.build_grid_world()
        
        # Number of valid states in the gridworld (there are 29 of them - 6x6 grid minus obstacles)
        self.state_size = state_size
        
        # Transition operator (3D tensor)
        self.T = T # T[st+1, st, a] gives the probability that action a will 
                   # transition state st to state st+1
        
        # Reward function (3D tensor)
        self.R = R # R[st+1, st, a ] gives the reward for transitioning to state
                   # st+1 from state st with action a
        
        # Absorbing states
        self.absorbing = absorbing
        
        # The locations of the valid states 
        self.locs = locs # State 0 is at the location self.locs[0] and so on
        
        # Number of the starting state
        self.starting_state = self.loc_to_state(self.starting_loc, locs)
        
        # Locating the initial state
        self.initial = np.zeros((1,len(locs)))
        self.initial[0, self.starting_state] = 1
        
        # Placing the walls on a bitmap
        self.walls = np.zeros(self.shape)
        for ob in self.obstacle_locs:
            self.walls[ob]=1
            
        # Placing the absorbers on a grid for illustration
        self.absorbers = np.zeros(self.shape)
        for ab in self.absorbing_locs:
            self.absorbers[ab] = -1
        
        # Placing the rewarders on a grid for illustration
        self.rewarders = np.zeros(self.shape)
        for i, rew in enumerate(self.absorbing_locs):
            self.rewarders[rew] = self.special_rewards[i]
        
        #Illustrating the grid world
        self.paint_maps()

        ################################
    
    

    ####### Getters ###########
    
    def get_transition_matrix(self):
        return self.T
    
    def get_reward_matrix(self):
        return self.R
    
    ########################
    

    ##########################
    
    ########### Internal Drawing Functions #####################

    ## You do not need to understand these functions in detail in order to complete the lab ##


    def draw_deterministic_policy(self, Policy, save_link=None):
        # Draw a deterministic policy
        # The policy needs to be a np array of 22 values between 0 and 3 with
        # 0 -> N, 1->E, 2->S, 3->W
        plt.figure()
        
        plt.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
        #plt.hold('on')
        for state, action in enumerate(Policy):
            if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
                continue
            arrows = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
            action_arrow = arrows[action] # Take the corresponding action
            location = self.locs[state] # Compute its location on graph
            plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph

        if save_link != None:
            plt.savefig(save_link)

        plt.show()

    
    def draw_value(self, Value, save_link=None):
        # Draw a policy value function
        # The value need to be a np array of 22 values 
        plt.figure()
        
        plt.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
        for state, value in enumerate(Value):
            if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
                continue
            location = self.locs[state] # Compute the value location on graph
            plt.text(location[1], location[0], round(value, 2), ha='center', va='center') # Place it on graph

        if save_link != None:
            plt.savefig(save_link)

        plt.show()


    def draw_deterministic_policy_grid(self, Policy, title, n_columns, n_lines):
        # Draw a grid of deterministic policy
        # The policy needs to be an arrya of np array of 22 values between 0 and 3 with
        # 0 -> N, 1->E, 2->S, 3->W
        plt.figure(figsize=(20,8))
        for subplot in range (len(Policy)): # Go through all policies
          ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
          ax.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
          for state, action in enumerate(Policy[subplot]):
              if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
                  continue
              arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
              action_arrow = arrows[action] # Take the corresponding action
              location = self.locs[state] # Compute its location on graph
              plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
          ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
        plt.show()

    def draw_value_grid(self, Value, title, n_columns, n_lines):
        # Draw a grid of value function
        # The value need to be an array of np array of 22 values 
        plt.figure(figsize=(20,8))
        for subplot in range (len(Value)): # Go through all values
          ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
          ax.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
          for state, value in enumerate(Value[subplot]):
              if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
                  continue
              location = self.locs[state] # Compute the value location on graph
              plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
          ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
        plt.show()

    ##########################
    
    
    ########### Internal Helper Functions #####################

    ## You do not need to understand these functions in detail in order to complete the lab ##

    def paint_maps(self):
        # Helper function to print the grid word used in __init__
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(self.walls)
        plt.title('Obstacles')
        plt.subplot(1,3,2)
        plt.imshow(self.absorbers)
        plt.title('Absorbing states')
        plt.subplot(1,3,3)
        plt.imshow(self.rewarders)
        plt.title('Reward states')
        plt.show()
        

    def build_grid_world(self):
        # Get the locations of all the valid states, the neighbours of each state (by state number),
        # and the absorbing states (array of 0's with ones in the absorbing states)
        locations, neighbours, absorbing = self.get_topology()
        
        # Get the number of states
        S = len(locations)
        
        # Initialise the transition matrix
        T = np.zeros((S, S, 4))
        
        for action in range(4):
            for effect in range(4):
                # Randomize the outcome of taking an action
                outcome = (action + effect + 1) % 4
                if outcome == 0:
                    outcome = 3
                else:
                    outcome -= 1

                # Fill the transition matrix:
                # A good way to understand the code, is to first ask ourselves what the structure 
                # of the transition probability ‘matrix’ should be, given that we have state, successor state and action. 
                # Thus, a simple row x column matrix of successor state and will not suffice, as we also have to condition 
                #  on the action. So we can therefore choose to implement this to  have a structure that is 3 dimensional
                # (technically a tensor, hence the variable name T). I would not worry too much about what a tensor is, 
                # it is simply an array that takes 3 arguments to get a value, just like conventional matrix is an array that
                # takes 2 arguments (row and column), to get a value. To touch all the elements in this structure we
                # need therefore to loop over states and actions.

                prob = self.action_randomizing_array[effect]
                for prior_state in range(S):
                    post_state = neighbours[prior_state, outcome]
                    post_state = int(post_state)
                    T[post_state, prior_state, action] = T[post_state, prior_state, action] + prob
                    
        # Build the reward matrix
        R = self.default_reward * np.ones((S, S, 4))
        for i, sr in enumerate(self.special_rewards):
            post_state = self.loc_to_state(self.absorbing_locs[i], locations)
            R[post_state, :, :]= sr
        
        return S, T, R, absorbing,locations
    

    def get_topology(self):
        height = self.shape[0]
        width = self.shape[1]
        
        locs = []
        neighbour_locs = []
        
        # get into all locs of the grid
        for i in range(height):
            for j in range(width):
                # Get the location of each state
                loc = (i, j)
                
                # And append it to the valid state locations if it is a valid state (ie not absorbing)
                if(self.is_location(loc)):
                    locs.append(loc)
                    
                    # Get an array with the neighbours of each state, in terms of locations
                    local_neighbours = [self.get_neighbour(loc, direction) for direction in ['nr','ea','so', 'we']]
                    neighbour_locs.append(local_neighbours)
                
        # translate neighbour lists from locations to states
        num_states = len(locs)
        state_neighbours = np.zeros((num_states, 4))
        
        for state in range(num_states):
            for direction in range(4):
                # Find neighbour location
                nloc = neighbour_locs[state][direction]
                
                # Turn location into a state number
                nstate = self.loc_to_state(nloc, locs)
      
                # Insert into neighbour matrix
                state_neighbours[state, direction] = nstate
                
    
        # Translate absorbing locations into absorbing state indices
        absorbing = np.zeros((1, num_states))
        for a in self.absorbing_locs:
            absorbing_state = self.loc_to_state(a, locs)
            absorbing[0, absorbing_state] = 1
        
        return locs, state_neighbours, absorbing 


    def loc_to_state(self, loc, locs):
        # takes list of locations and gives index corresponding to input loc
        #ind = locs.index(tuple(loc))
        #return self.state_list[ind]
        return locs.index(tuple(loc))
    
    def is_location(self, loc):
        # It is a valid location if it is in grid and not obstacle
        if(loc[0] < 0 or loc[1] < 0 or loc[0] > self.shape[0] - 1 or loc[1] > self.shape[1] - 1):
            return False
        elif(loc in self.obstacle_locs):
            return False
        else:
             return True
            
    def get_neighbour(self, loc, direction):
        # Find the valid neighbours (ie that are in the grid and not obstacle)
        i = loc[0]
        j = loc[1]
        
        nr = (i - 1, j)
        ea = (i, j + 1)
        so = (i + 1, j)
        we = (i, j - 1)
        
        # If the neighbour is a valid location, accept it, otherwise, stay put
        if(direction == 'nr' and self.is_location(nr)):
            return nr
        elif(direction == 'ea' and self.is_location(ea)):
            return ea
        elif(direction == 'so' and self.is_location(so)):
            return so
        elif(direction == 'we' and self.is_location(we)):
            return we
        else:
            #default is to return to the same location
            return loc
        
        
    ################################# METHODS #################################
    
    """
        policy_evaluation(self, policy, discount, threshold)
    
    For a given policy, evaluates the values of the different states of the environment.  
    """
    def policy_evaluation(self, policy, discount, threshold):
        
        # Make sure delta is bigger than the threshold to start with
        delta = 2 * threshold
        
        #Get the reward and transition matrices
        R = self.get_reward_matrix()
        T = self.get_transition_matrix()
        
        # The value is initialised at 0
        V = np.zeros(policy.shape[0])
        # Make a deep copy of the value array to hold the update during the evaluation
        Vnew = np.copy(V)
        
        epoch = 0
        # While the Value has not yet converged do:
        while delta > threshold:
            epoch += 1
            for state_idx in range(policy.shape[0]):
                # If it is one of the absorbing states, ignore
                if(self.absorbing[0, state_idx]):
                    continue
                
                # Accumulator variable for the Value of a state
                tmpV = 0
                for action_idx in range(policy.shape[1]):
                    # Accumulator variable for the State-Action Value
                    tmpQ = 0
                    for state_idx_prime in range(policy.shape[0]):
                        tmpQ = tmpQ + T[state_idx_prime, state_idx, action_idx] * (R[state_idx_prime, state_idx, action_idx] + discount * V[state_idx_prime])
                    
                    tmpV += policy[state_idx, action_idx] * tmpQ
                    
                # Update the value of the state
                Vnew[state_idx] = tmpV
            
            # After updating the values of all states, update the delta
            # Note: The below is our example way of computing delta.
            #       Other stopping criteria may be used (for instance mean squared error).
            #       We encourage you to explore different ways of computing delta to see 
            #       how it can influence outcomes.
            delta =  max(abs(Vnew - V))
            # and save the new value into the old
            V = np.copy(Vnew)
            
        return V, epoch


    """
        policy_iteration(self, discount=0.9, threshold=0.0001)
    
    Takes initial values and random policy. Evaluate the values thanks to the policy, then 
    update the policy greedily thanks to the values; evalues the values thanks to the new policy, 
    etc... until convergence. 
    """
    def policy_iteration(self, discount=0.9, threshold = 0.0001):
        ## Slide 139 of the lecture notes for pseudocode ##
        
        # Transition and reward matrices, both are 3d tensors, c.f. internal state
        T = self.get_transition_matrix()
        R = self.get_reward_matrix()
        
        # Initialisation
        policy = np.zeros((self.state_size, self.action_size)) # Vector of 0
        policy[:, 0] = 1 # Initialise policy to choose action 1 systematically
        epochs = 0
        policy_stable = False # Condition to stop the main loop

        while not(policy_stable): 

            # Policy evaluation
            V, epochs_eval = self.policy_evaluation(policy, discount, threshold)
            epochs += epochs_eval # Increment epoch

            # Set the boolean to True, it will be set to False later if the policy prove unstable
            policy_stable = True

            # Policy iteration
            for state_idx in range(policy.shape[0]):
                
                # If not an absorbing state
                if not(self.absorbing[0,state_idx]):
                    
                    # Store the old action
                    old_action = np.argmax(policy[state_idx,:])
                
                    # Compute Q value
                    Q = np.zeros(4) # Initialise with value 0
                    for state_idx_prime in range(policy.shape[0]):
                        Q += T[state_idx_prime, state_idx, :] * (R[state_idx_prime, state_idx, :] + discount * V[state_idx_prime])

                    # Compute corresponding policy
                    new_policy = np.zeros(4)
                    new_policy[np.argmax(Q)] = 1  # The action that maximises the Q value gets probability 1
                    policy[state_idx] = new_policy
                
                    # Check if the policy has converged
                    if old_action != np.argmax(policy[state_idx]):
                        policy_stable = False
            
        return V, policy, epochs
                
        
    """
        value_iteration(self, discount=0.9, threshold=0.0001)
    
    Evaluate the state values iteratively by using action thanks to the argmax of the values themself (no explicit policy).
    It is the special case of policy_iteration where k=1 !
    """
    def value_iteration(self, discount = 0.9, threshold = 0.0001):
        ## Slide 144 of the lecture notes for the algorithm ##
        
        # Transition and reward matrices, both are 3d tensors, c.f. internal state
        T = self.get_transition_matrix()
        R = self.get_reward_matrix()
        
        # Initialisation
        epochs = 0
        delta = threshold # Setting value of delta to go through the first breaking condition
        V = np.zeros(self.state_size) # Initialise values at 0 for each state

        while delta >= threshold:
            epochs += 1 # Increment the epoch
            delta = 0 # Reinitialise delta value

            # For each state
            for state_idx in range(self.state_size):

                # If not an absorbing state
                if not(self.absorbing[0, state_idx]):
                  
                    # Store the previous value for that state
                    v = V[state_idx] 

                    # Compute Q value
                    Q = np.zeros(4) # Initialise with value 0
                    for state_idx_prime in range(self.state_size):
                        Q += T[state_idx_prime,state_idx,:] * (R[state_idx_prime,state_idx, :] + discount * V[state_idx_prime])
                
                    # Set the new value to the maximum of Q
                    V[state_idx]= np.max(Q) 

                    # Compute the new delta
                    delta = max(delta, np.abs(v - V[state_idx]))
            

        # When the loop is finished, fill in the optimal policy
        optimal_policy = np.zeros((self.state_size, self.action_size)) # Initialisation

        # For each state
        for state_idx in range(self.state_size):
             
            # Compute Q value
            Q = np.zeros(4)
            for state_idx_prime in range(self.state_size):
                Q += T[state_idx_prime,state_idx,:] * (R[state_idx_prime,state_idx, :] + discount * V[state_idx_prime])
            
            # The action that maximises the Q value gets probability 1
            optimal_policy[state_idx, np.argmax(Q)] = 1 

        return optimal_policy, epochs

    """
        initial_state(self, rng, random=True)

    Define the initial state of an episode. 
    """
    def initial_state(self, rng, random=True, risk=False):
        if random:
            if risk == False:
                possible_starting_states = [i for i in range(0, self.state_size) if (self.absorbing[0, i]==0 and i not in self.risky_states)]
            else:
                possible_starting_states = [i for i in range(0, self.state_size) if self.absorbing[0, i] == 0]
            s = rng.choice(possible_starting_states, 1)[0]
        else:
            s = self.starting_state
        return s

    """
        action_choice(self, s, Q, rng, epsilon)

    Choose a new action thanks to Q and being epsilon-greedy
    """
    def action_choice(self, s, Q, rng, epsilon=0.05):
        rnd_nb = rng.uniform(0, 1)

        a_star = np.argmax(Q[s, :])
        if rnd_nb > epsilon:
            a = a_star
        else:
            remaining_actions = self.action_space[:a_star] + self.action_space[a_star + 1:]
            a = rng.choice(remaining_actions, 1)[0]

        return a

    """
        env(a, rng)

    End up in a new state thanks to the old state, the action that was taken, the random generator
    and the transition matrix of the grid.
    """
    def env(self, s, a, rng):
        probas = self.T[:, s, a]
        probas_cum = np.cumsum(probas)
        assert probas_cum[-1] == 1
        rnd_nb = rng.uniform(0, 1)
        for i in range(0, len(probas_cum)):
            if rnd_nb <= probas_cum[i]:
                s_prime = i
                break
        return s_prime

    """
        simulate_episode(self, policy)

    Function that simulates an episode of this gridworld and retrieves the trace of this episode.
    The episode is simulated under a certain policy (the policy can be implicitely defined thanks 
    to a value function or a state-value function.)
    """
    def simulate_episode(self, Q, epsilon=0.05, seed=5, verbose=False, random_start=True, risky_start=False):
        
        rng = np.random.default_rng()

        # initial state
        s = self.initial_state(rng, random=random_start, risk=risky_start)

        # initial trace
        trace = np.array([s])

        keep_runing = True
        while keep_runing:
            # choose a new action - thanks to Q and being epsilon-greedy
            a = self.action_choice(s, Q, rng, epsilon)
            verbose and print("Chose action : ", a)

            # end up in a new state for it
            s_prime = self.env(s, a, rng)
            verbose and print("Ended in state : ", s_prime)

            # get a reward for it
            r = self.R[s_prime, s, a]
            verbose and print("Got reward : ", r)

            # collect everything 
            s = s_prime

            # update the trace
            trace = np.append(trace, [a, r, s_prime])

            # check if we need to keep runing
            keep_runing = (self.absorbing[0, s_prime] == 0)

        # return the trace
        return trace

    """
        montecarlo_on_control(self, epsilon=0.05, discount=0.9, learning_rate=1, threshold=0.0001, max_epoch=100)
    
    Monte Carlo control to find the best policy ! See slide 202 of the lecture note for the algorithm. 
    """
    def montecarlo_on_control(self, epsilon_init=0.05, decay_speed=0, decay_smooth=1, discount=0.9, learning_rate=1, threshold=0.0001, max_epoch=100, random_start=True, risky_start=False):
        ## Slide 202 of the lecture notes for the algorithm ##
        
        rng = np.random.default_rng()
        # initialize randomly Q
        Q = rng.uniform(-1, 1, size=(self.state_size, self.action_size))

        # initialize N, the number of occurence of each (s, a) couple
        N = np.zeros((self.state_size, self.action_size), dtype=int)
        
        # Initialisation
        epochs = 0
        delta = threshold # Setting value of delta to go through the first breaking condition
        rewards = []

        while delta >= threshold and epochs < max_epoch:
            epochs += 1 # Increment the epoch
            delta = 10 # Reinitialise delta value

            # generate an episode
            epsilon = (decay_smooth / (decay_smooth + decay_speed * epochs)) * epsilon_init

            trace = self.simulate_episode(Q, epsilon, seed=5, verbose=False, random_start=random_start, risky_start=risky_start)
            rewards += [sum(trace[2::3])]
            tabu = []
            nb_sa = int((len(trace) - 1) / 3)

            for i in range(0, nb_sa):
                s, a = int(trace[3 * i]), int(trace[3 * i + 1])
                if (s, a) in tabu:
                    pass
                else:
                    tabu += [(s, a)]
                    N[s, a] += 1
                    G = sum([(discount**i) * r for i, r in enumerate(trace[(3*i+2)::3])])
                    Q[s, a] = Q[s, a] + (learning_rate / N[s, a]) * (G - Q[s, a])

        # When the loop is finished, fill in the optimal policy
        optimal_policy = np.zeros((self.state_size, self.action_size)) # Initialisation

        # For each state
        for state_idx in range(self.state_size):
            # The action that maximises the Q value gets probability 1
            optimal_policy[state_idx, np.argmax(Q[state_idx, :])] = 1 

        return Q, optimal_policy, epochs, rewards
    
    """
        sarsa_on_td_control(self, epsilon=0.05, discount=0.9, learning_rate=0.3, threshold=0.0001, max_epoch=100)
    
    SARSA algo : on-policy Temporal Difference learning. See slide 212 for the algo !
    """
    def sarsa_on_td_control(self, epsilon_init=0.05, decay_speed=0, decay_smooth=1, discount=0.9, learning_rate=0.3, threshold=0.0001, max_epoch=100):
        ## Slide 212 of the lecture notes for the algorithm ##
        
        rng = np.random.default_rng()
        # initialize randomly Q
        Q = rng.uniform(-1, 1, size=(self.state_size, self.action_size))

        # because of the update rule, Q[absorbing_state, :] must be set to 0
        abs_idxs = np.where(self.absorbing[0] == 1)[0]
        Q[abs_idxs, :] = np.zeros(Q.shape[1])
        
        # Initialisation
        epochs = 0
        delta = threshold # Setting value of delta to go through the first breaking condition
        rewards = []

        while delta >= threshold and epochs < max_epoch:
            epochs += 1 # Increment the epoch
            delta = 10 # Reinitialise delta value

            # launch a new episode
            epsilon = (decay_smooth / (decay_smooth + decay_speed * epochs)) * epsilon_init

            # initialise state and take first action
            s = self.initial_state(rng, random=True)
            a = self.action_choice(s, Q, rng, epsilon)

            # sum of rewards in the episode (for plots)
            R = 0

            keep_runing = True
            while keep_runing:

                # end up in a new state for it
                s_prime = self.env(s, a, rng)

                # get a reward for it
                r = self.R[s_prime, s, a]
                R += r

                # choose a new action
                a_prime = self.action_choice(s, Q, rng, epsilon)

                # update Q
                Q[s, a] = Q[s, a] + learning_rate * (r + discount * Q[s_prime, a_prime] - Q[s, a])

                # update s, a  
                s = s_prime
                a = a_prime

                # check if we need to keep runing
                keep_runing = (self.absorbing[0, s] == 0)

            rewards += [R]

        # When the loop is finished, fill in the optimal policy
        optimal_policy = np.zeros((self.state_size, self.action_size)) # Initialisation

        # For each state
        for state_idx in range(self.state_size):
            # The action that maximises the Q value gets probability 1
            optimal_policy[state_idx, np.argmax(Q[state_idx, :])] = 1 

        return Q, optimal_policy, epochs, rewards
    
###########################################         
