from logging import root
import time
from state import State
from copy import deepcopy
import random
import torch
import os
import numpy as np
import pickle
import math
from environments.seaquest import shot_cool_down


safety_steps=9
precision=0.005
oxygen_threshold=10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(root, budget,val_network,data_collecting,max_path_length,thresh=0):
    expansion(root,val_network,data_collecting)
    returns={}
    
    for i in range(len(root.children)):
        random_policy(root.children[i],budget,val_network,returns,data_collecting,max_path_length,thresh)
    
    return selection(returns,root)


def selection(returns,root):
    max_val=-1
    if bool(returns):
        keys=list(returns.keys())
        key=keys[np.random.randint(0,len(keys))]
        pos=returns[key][0]
        # for key in returns:
        #     if (max_val<returns[key][1]) or(abs(max_val-returns[key][1])<precision and random.random()>0.5):
        #         pos=returns[key][0]
        #         max_val= returns[key][1]
        return pos,True
    else:        
        for i in range(len(root.children)):
            child=root.children[i]
            if child.q>max_val or (abs(max_val-child.q)<precision and random.random()>0.5):
                max_val=child.q
                pos=child.action_index
        
        if len(root.children)==0:
            # deadend happens
            return 0,False
        
        return pos,False


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def expansion(parent,network,data_collecting):
    for i in range(len(parent.game.action_map)):
        child = State(deepcopy(parent.game),parent,i)
        _, terminal,_= child.game.act(i)
        if not terminal and not (child.game.shot_timer!=shot_cool_down and i == 5):
            parent.children.append(child)
            input=child.game.state()

            if data_collecting:
                child.q=data_collection(deepcopy(child.game))
            else:
                with torch.no_grad():
                    child.q=network(get_state(input))
        
        # this is only for gathering data
        if terminal and data_collecting:
            data_collection(deepcopy(child.game))


def data_collection(game):
    safe,total=safety_value_estimation(game,safety_steps)
    value=safe/total
    if os.path.exists("dataset/seaquest"):
        with open("dataset/seaquest", "rb") as fp:
            datalist = pickle.load(fp)
    else:
        datalist=[]
    datalist.append([game.state(),game.ramp_index,value])
    with open("dataset/seaquest", "wb") as fp:
        pickle.dump(datalist, fp)

    return value


def safety_value_estimation(game,steps):
    
    if game.terminal:
        return 0,math.pow(len(game.action_map),steps)
    
    if steps==3:
        uncertanity=1-(1/game.e_spawn_speed)
        # we have no divers but enough oxygen to take divers
        if game.oxygen>=oxygen_threshold:
            if game.sub_x<2 or game.sub_x>7:
                return math.pow(len(game.action_map),3)*uncertanity,math.pow(len(game.action_map),3)
            else:
                return math.pow(len(game.action_map),3),math.pow(len(game.action_map),3)
        
        # we have divers and time to go to the surface
        if game.diver_count>0 and game.oxygen>=game.sub_y:
            if game.sub_x<2 or game.sub_x>7:
                return math.pow(len(game.action_map),3)*uncertanity,math.pow(len(game.action_map),3)
            else:
                return math.pow(len(game.action_map),3),math.pow(len(game.action_map),3)
        
        # we have no divers and no time to take one.
        return 0,math.pow(len(game.action_map),steps) 

    safe=0
    total=0
    for i in range(len(game.action_map)):
        node=deepcopy(game)
        node.act(i)
        s,t=safety_value_estimation(node,steps-1)
        safe=safe+s
        total=total+t
    
    if game.sub_x<2 or game.sub_x>7:
        uncertanity=1-(1/game.e_spawn_speed)
        return uncertanity*safe,total
    else:
        return safe,total

def option_running(env,option):
    score=0
    done=False
    actions=0
    
    for action in option:
        game=deepcopy(env.get_game())
        r,done_sim,_=game.act(action)
        if (not done_sim) and not(r==0 and actions==len(option)-1):
            r,done,_=env.act(action)
            # state=env.state()
            # for i in range(4):
            #     print(state[:,:,i])
            #     print()
            #     print()
            # print("*"*20)
            score=score+r
            actions+=1
        else:
            return score,actions,done
        if done:
            break

    return score,actions,done

def random_policy(node, budget,network,returns,data_collecting,max_path_length,thresh):
    
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
            option.append(action)
            r,terminal,info=root.act(action)
            if info or r!=0 or (root.surface and (not terminal)):
                input=root.state()

                if data_collecting:
                    new_val=data_collection(deepcopy(root))
                else:
                    with torch.no_grad():
                        new_val=network(get_state(input))

                if new_val>thresh and random.random()>0.5:
                    max_subgoal_val=new_val
                    subgoal=option
                novelty=True
        
        # this is only for gathering data
        if data_collecting and not novelty:
            data_collection(deepcopy(root))
    
    #print("trajectories: "+str(trajectories))
    if subgoal is not None:
        returns[node.action_index]=[subgoal,max_subgoal_val]


