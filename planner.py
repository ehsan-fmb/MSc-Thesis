import time
from state import State
from copy import deepcopy
import random
import torch
import numpy as np

max_path_length=8
val_threshold=0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(root, budget,val_network):
    
    expansion(root,val_network)
    returns={}

    for i in range(len(root.children)):
        random_policy(root.children[i],budget,returns,val_network)
    
    return selection(returns,root)


def selection(returns,root):
    max_val=-1
    if bool(returns):
        for key in returns:
            if (max_val<returns[key][1]) or(max_val==returns[key][1] and random.random()>0.5):
                pos=returns[key][0]
                max_val= returns[key][1]
        return pos,True
    else:
        
        for child in root.children:
            if child.q>max_val or (max_val==child.q and random.random()>0.5):
                max_val=child.q
                pos=child.action_index
        
        if len(root.children)==0:
            # deadend happens
            return 0,False
        
        return pos,False


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def expansion(parent,network):
    for i in range(len(parent.game.action_map)):
        child = State(deepcopy(parent.game),parent,i)
        _, terminal= child.game.act(i)
        if not terminal and not (child.game.player_x == parent.game.player_x and child.game.player_y == parent.game.player_y and i != 0):
            parent.children.append(child)
            board=child.game.state()[:,:,0:2]
            ramping=np.ones((board.shape[0],board.shape[1]))*child.game.ramp_index
            input=np.dstack((board,ramping))
            with torch.no_grad():
                child.q=network(get_state(input))



def option_running(env,option):
    score=0
    done=False
    for action in option:
        game=deepcopy(env.get_game())
        _,done_sim=game.act(action)
        if not done_sim:
            r,done=env.act(action)
            score=score+r
        else:
            return score,done
        if done:
            break
    return score,done


def random_policy(node, budget,returns,network):
    
    start = time.time()
    max_subgoal_val=0
    subgoal=None
    trajectories=0
    
    while time.time()<=start+budget:
        
        root=deepcopy(node.game)
        novelty=False
        path_length=0
        trajectories+=1
        terminal=False
        option=[node.action_index]
        
        while not novelty and path_length<=max_path_length and not terminal :
            
            action=random.randint(0,len(root.action_map)-1)
            path_length+=1
            r,terminal=root.act(action)
            option.append(action)
            
            if r!=0:
                
                board=root.state()[:,:,0:2]
                ramping=np.ones((board.shape[0],board.shape[1]))*root.ramp_index
                input=np.dstack((board,ramping))
                
                with torch.no_grad():
                    input=(torch.tensor(input, device=device).permute(2, 0, 1)).unsqueeze(0).float()
                    new_val=network(input)

                if new_val>max_subgoal_val:
                    max_subgoal_val=new_val
                    subgoal=option
                novelty=True
    
    #print("trajectories: "+str(trajectories))
    if subgoal is not None and max_subgoal_val> val_threshold:
        returns[node.action_index]=[subgoal,max_subgoal_val]


