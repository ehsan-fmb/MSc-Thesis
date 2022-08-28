import time
from state import State
from copy import deepcopy
import multiprocessing
import random
import torch
import numpy as np

max_path_length=8
val_threshold=0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(root, budget,val_networks):
    expansion(root,val_networks)
    processes=[]
    manager = multiprocessing.Manager()
    returns= manager.dict()
    
    for i in range(len(root.children)):
        random_polciy(root.children[i],budget,returns,val_networks[i])
        # p=multiprocessing.Process(target=random_polciy,args=[root.children[i],budget,returns,val_networks[i]])
    #     p.start()
    #     processes.append(p)
    # for process in processes:
    #     process.join()
    
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
        return pos,False


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def expansion(parent,networks):
    for i in range(len(parent.game.action_map)):
        child = State(deepcopy(parent.game),parent,i)
        _, terminal= child.game.act(i)
        if not terminal and not (child.game.player_x == parent.game.player_x and child.game.player_y == parent.game.player_y and i != 0):
            parent.children.append(child)
            board=child.game.state()[:,:,0:2]
            ramping=np.ones((board.shape[0],board.shape[1]))*child.game.ramp_index
            input=np.dstack((board,ramping))
            with torch.no_grad():
                child.q=networks[i](get_state(input))


def value_estimation(network,state):
    return network(state)

def random_polciy(node, budget,returns,network):
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
                    new_val=value_estimation(network,get_state(input))
                if new_val>max_subgoal_val:
                    max_subgoal_val=new_val
                    subgoal=option
                novelty=True
    
    #print("trajectories: "+str(trajectories))
    if subgoal is not None and max_subgoal_val> val_threshold:
        returns[node.action_index]=[subgoal,max_subgoal_val]

