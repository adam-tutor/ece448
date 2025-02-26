'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    
    P = np.zeros((model.M, model.N, 4, model.M, model.N))
    
    for r in range(model.M): #rows
        for c in range(model.N): #columns
            
            if model.TS[r,c]: #terminal state
                P[r,c,:,:,:] = 0
                continue
            
            # 4 directions (up down left right)
            for direction in range(4):
                idx_row = r #following vars used for positions depending on current direction
                idx_col = c #all initialized to r and c respectively
                dest_left_row = r
                dest_left_col = c
                dest_right_row = r
                dest_right_col = c
                
                up_row = r-1 #used to calculate position changes
                down_row = r+1
                right_column  = c+1
                left_column = c-1

                intended_dir = model.D[r, c, 0] #3 different probabilities based on using idx_, dest_left_, dest_right_
                counterclockwise = model.D[r, c, 1]
                clockwise = model.D[r, c, 2]
            
                if direction == 0:
                    idx_col = max(0, left_column)
                    dest_left_row = min(model.M -1, down_row)
                    dest_right_row = max(0, up_row)
                if direction == 1:
                    idx_row = max(0, up_row)
                    dest_left_col = max(0, left_column)
                    dest_right_col = min(model.N -1, right_column)
                if direction == 2:
                    idx_col = min(model.N -1, right_column)
                    dest_left_row = max(0, up_row)
                    dest_right_row = min(model.M -1, down_row)
                if direction == 3:
                    idx_row = min(model.M -1, down_row)
                    dest_left_col = min(model.N -1, right_column)
                    dest_right_col = max(0, left_column)
                
                if model.W[idx_row, idx_col]:
                    P[r, c, direction, r, c] += intended_dir
                else:
                    P[r, c, direction, idx_row, idx_col] += intended_dir
                    
                if model.W[dest_left_row, dest_left_col]:
                    P[r, c, direction, r, c] += counterclockwise  
                else:
                    P[r, c, direction, dest_left_row, dest_left_col] += counterclockwise
                    
                if model.W[dest_right_row, dest_right_col]:
                    P[r, c, direction, r, c] += clockwise
                else:
                    P[r, c, direction, dest_right_row, dest_right_col] += clockwise
    return P

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    U_next = np.zeros((model.M, model.N)) #create MxN
    for r in range(model.M):
        for c in range(model.N):
            utility_array = []
            for direction in range(4):
                utility_array.append(np.sum(P[r, c, direction] * U_current))
            U_next[r,c] = model.R[r,c] + model.gamma * np.max(utility_array)
    return U_next

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    P = compute_transition(model)
    U = np.zeros((model.M, model.N))
    for i in range(100):
        U_next = compute_utility(model, U, P)
        if np.max(np.abs(U_next - U)) < epsilon:
            break
        else:
            U = U_next
    return U

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    U = np.zeros((model.M, model.N))
    U_next = np.zeros((model.M, model.N))
    for i in range(500):
        for r in range(model.M):
            for c in range(model.N):
                U_next[r,c] = model.R[r,c] + model.gamma * np.sum(model.FP[r, c] * U)
        if np.max(np.abs(U_next - U)) < epsilon:
            break
        U = U_next.copy()
    return U
