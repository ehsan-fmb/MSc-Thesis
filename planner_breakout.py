from logging import root
import time
from state import State
from copy import deepcopy
import random
import torch
import os
import pickle
import math

safety_steps=9
precision=0.05
score_estimation_length=10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(root, budget,val_network,data_collecting,max_path_length,seed,thresh,coeff):
    random.seed(seed)
    expansion(root,val_network,data_collecting)
    returns={}

    for i in range(len(root.children)):
        random_policy(root.children[i],val_network,budget,returns,data_collecting,max_path_length,thresh,coeff)
    
    return selection(returns,root)


def selection(returns,root):
    max_val=-1
    if bool(returns):
        for key in returns:
            if (max_val<returns[key][1]) or(abs(max_val-returns[key][1])<precision):
                pos=returns[key][0]
                max_val= returns[key][1]
        return pos,True
    else:        
        for i in range(len(root.children)):
            child=root.children[i]
            if child.q>max_val or (abs(max_val-child.q)<0.005 and random.random()>0.5):
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
        if not terminal and not (child.game.pos == parent.game.pos and i != 0):
            parent.children.append(child)
            input=child.game.state()

            if data_collecting:
                child.q=data_collection(deepcopy(child.game))
            else:
                with torch.no_grad():
                    child.q=network(get_state(input))

def data_collection(game):
    death,total=safety_value_estimation(game,safety_steps)
    value=(total-death)/total
    if os.path.exists("dataset/breakout"):
        with open("dataset/breakout", "rb") as fp:
            datalist = pickle.load(fp)
    else:
        datalist=[]
    datalist.append([game.state(),value])
    with open("dataset/breakout", "wb") as fp:
        pickle.dump(datalist, fp)

    return value


def safety_value_estimation(game,steps):
    
    if game.terminal:
        return math.pow(len(game.action_map),steps),math.pow(len(game.action_map),steps)
    
    if steps==0:
        return 0,1 

    death=0
    total=0
    for i in range(len(game.action_map)):
        node=deepcopy(game)
        node.act(i)
        d,t=safety_value_estimation(node,steps-1)
        death=death+d
        total=total+t
    
    return death,total

def score_estimation(game,network,coeff):
    score=0
    hit=False
    input=game.state()
    bounce_val=network(get_state(input))
    for _ in range(score_estimation_length):
        r,done,_=game.act(0)
        score=score+r
        if r!=0:
            input=game.state()
            safe_val=network(get_state(input))
            hit=True
        if done:
            break
    
    if not hit:
        score=coeff*bounce_val+(1-coeff)*score
    else:
        score=coeff*safe_val+(1-coeff)*score
    return score 

def option_running(env,option):
    score=0
    done=False
    actions=0
    
    for action in option:
        if not done:
            r,done,_=env.act(action)
            score=score+r
            actions+=1
        else:
            return score,actions,done

    return score,actions,done


def random_policy(node, network ,budget,returns,data_collecting,max_path_length,thresh,coeff):
    start = time.time()
    max_subgoal_val=thresh
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
            _,terminal,info=root.act(action)
            if info:
                
                new_val=score_estimation(root,network,coeff)
                # input=root.state()

                # if data_collecting:
                #     new_val=data_collection(deepcopy(root))
                # else:
                #     with torch.no_grad():
                #         new_val=network(get_state(input))

                if new_val>max_subgoal_val or (abs(new_val-max_subgoal_val)<precision):
                    max_subgoal_val=new_val
                    subgoal=option
                novelty=True
        
        # this is only for gathering data
        if data_collecting and not novelty:
            data_collection(deepcopy(root))
    
    #print("trajectories: "+str(trajectories))
    if subgoal is not None:
        returns[node.action_index]=[subgoal,max_subgoal_val]


