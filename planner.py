import time
from state import State
from copy import deepcopy
import random
import torch
import numpy as np

# for asterix
# max_path_length=8
# subgoal_thresh=0.7

# for freeway
# max_path_length=10
# subgoal_thresh=-1
# child_thresh=0.5

# for breakout
max_path_length=8
subgoal_thresh=-1
score_estimation_length=8

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
        # for asterix
        # for child in root.children:
        #     if child.q>max_val or (max_val==child.q and random.random()>0.5):
        #         max_val=child.q
        #         pos=child.action_index
        

        # for breakout
        for i in range(len(root.children)):
            child=root.children[i]
            if child.q>max_val or (max_val==child.q and random.random()>0.5):
                max_val=child.q
                pos=child.action_index
                index=i
        
        if len(root.children)>0:
            if root.children[index].q<0.2:
                if root.game.ball_x>root.game.pos:
                    pos=2
                if root.game.ball_x<root.game.pos:
                    pos=1
                # if root.children[index].action_index!=pos:
                #     print("man")
                # else:
                #     print("doos")
                

        # tweaking for freeway
        # pos=2
        # for child in root.children:
        #     if child.action_index==0 and child.q>child_thresh:
        #         pos=0
        # for child in root.children:    
        #     if child.action_index==1 and child.q>child_thresh:
        #         pos=1
        
        # for asterix and breakout; deadends are already addressed in freeway.
        if len(root.children)==0:
            # deadend happens
            return 0,False
        
        return pos,False


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def expansion(parent,network):
    for i in range(len(parent.game.action_map)):
        child = State(deepcopy(parent.game),parent,i)
        
        # for asterix
        #_, terminal,_= child.game.act(i)
        # if not terminal and not (child.game.player_x == parent.game.player_x and child.game.player_y == parent.game.player_y and i != 0):
        #     parent.children.append(child)
        #     board=child.game.state()[:,:,0:2]
        #     ramping=np.ones((board.shape[0],board.shape[1]))*child.game.ramp_index
        #     input=np.dstack((board,ramping))
        
        # for freeway
        # _,_,terminal= child.game.act(i)
        # if not terminal:
        #     parent.children.append(child)
        #     input=child.game.state()
        
        # for breakout
        _, terminal,_= child.game.act(i)
        if not terminal and not (child.game.pos == parent.game.pos and i != 0):
            parent.children.append(child)
            input=child.game.state()

            with torch.no_grad():
                child.q=network(get_state(input))



def option_running(env,option):
    score=0
    done=False
    
    for action in option:
        game=deepcopy(env.get_game())
        _,done_sim,info=game.act(action)
        
        # for asterix and freeway
        # if not done_sim and not info:
        
        # for breakout
        if not done:
            r,done,_=env.act(action)
            score=score+r
        else:
            return score,done
        if done:
            break

    return score,done


def breakout_estimation(game):
    score=0
    for _ in range(score_estimation_length):
        r,terminal,_=game.act(0)
        if terminal:
            break
        score=score+r
    return score

def random_policy(node, budget,returns,network):
    
    start = time.time()
    max_subgoal_val=-1
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

            # for asterix
            #r,terminal,_=root.act(action)

            # for freeway
            # r,_,terminal=root.act(action)

            # for breakout
            _,terminal,info=root.act(action)
            
            # for asterix and freeway
            #if r!=0:
            if info:
                
                # for asterix
                # board=root.state()[:,:,0:2]
                # ramping=np.ones((board.shape[0],board.shape[1]))*root.ramp_index
                # input=np.dstack((board,ramping))
                # with torch.no_grad():
                #     new_val=network(get_state(input))

                # for freeway
                # new_val=0

                # for breakout
                new_val=breakout_estimation(root)

                if new_val>max_subgoal_val:
                    max_subgoal_val=new_val
                    subgoal=option
                novelty=True
    
    #print("trajectories: "+str(trajectories))
    if subgoal is not None and max_subgoal_val> subgoal_thresh:
        returns[node.action_index]=[subgoal,max_subgoal_val]


