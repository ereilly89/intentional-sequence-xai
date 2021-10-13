import numpy as np

# Define the states
location_to_state = {
    'L1' : 0,
    'L2' : 1,
    'L3' : 2,
    'L4' : 3,
    'L5' : 4,
    'L6' : 5,
    'L7' : 6,
    'L8' : 7,
}

# Define the actions
actions = [0,1,2,3,4,5,6,7]

# Define the rewards
rewards = np.array([[0,1,100,1,0,0,0,0],
              [0,0,0,0,0,0,1,0],
              [0,0,0,-100,0,0,0,1000],
              [0,0,0,0,0,0,0,0],
              [1,0,0,1,0,0,0,0],
              [0,0,0,0,1,0,1,0],
              [1,0,100,0,0,0,0,0],
              [0,0,0,0,0,0,0,0]])

# Maps indices to locations
state_to_location = dict((state,location) for location,state in location_to_state.items())

# Initialize parameters
gamma = 0.95 # Discount factor 
alpha = 0.9 # Learning rate 

# Initializing Q-Values
Q = np.array(np.zeros([9,9]))


class QAgent():
    
    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, alpha, gamma, location_to_state, actions, rewards, state_to_location, Q):
        
        self.gamma = gamma  
        self.alpha = alpha 
        
        self.location_to_state = location_to_state
        self.actions = actions
        self.rewards = rewards
        self.state_to_location = state_to_location
        
        self.Q = Q
        
    # Training the robot in the environment
    def training(self, iterations):
        
        rewards_new = np.copy(self.rewards)
        
        #ending_state = self.location_to_state[end_location]
        #rewards_new[ending_state, ending_state] = 999
        
        for i in range(iterations):
            current_state = np.random.randint(0,8) 
            playable_actions = []

            for j in range(8):
                if rewards_new[current_state,j] > 0:
                    playable_actions.append(j)
            if len(playable_actions) == 0:
                continue
    
            next_state = np.random.choice(playable_actions)
            TD = rewards_new[current_state,next_state] + \
                    self.gamma * self.Q[next_state, np.argmax(self.Q[next_state,])] - self.Q[current_state,next_state]
            
            self.Q[current_state,next_state] += self.alpha * TD

        #route = [start_location]
        #next_location = start_location
        
        # Get the route 
        #self.get_optimal_route(start_location, end_location, next_location, route, self.Q)
        
    # Get the optimal route
    def get_optimal_route(self, start_location, end_location, next_location, route, Q):
        
        while(next_location != end_location):
            starting_state = self.location_to_state[start_location]
            next_state = np.argmax(Q[starting_state,])
            next_location = self.state_to_location[next_state]
            route.append(next_location)
            start_location = next_location
        
        print(route)


qagent = QAgent(alpha, gamma, location_to_state, actions, rewards,  state_to_location, Q)
qagent.training(1000)
print(str(Q))