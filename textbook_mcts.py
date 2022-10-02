import math
import time
from state import State
import random
from copy import deepcopy



exploration_weight=math.sqrt(2)
score_estimation_length=8


def main(root,budget,max_simulation_length):
    start = time.time()
    simulations=0
    while time.time() <= start + budget:
        cur = selection(root)
        if cur.n<1:
            delta = simulation(cur,max_simulation_length)
            simulations+=1

        else:
            expansion(cur)
            # no children condition should be checked!
            if len(cur.children)>0:
                cur=cur.children[0]
            
            delta=simulation(cur,max_simulation_length)
            simulations+=1
        backpropagation(root,cur,delta)


    #print(simulations)
    if len(root.children)>0:
        return best_child(root,True).action_index,False
    else:
        return 0,False


def backpropagation(root,cur,delta):
    while cur!=root:
        cur.n+=1
        cur.q+=delta
        cur=cur.parent


def expansion(parent):
    for i in range(len(parent.game.action_map)):
        child = State(deepcopy(parent.game),parent,i)
        # for asterix
        r, terminal,_= child.game.act(i)
        child.q = child.q+r
        if not terminal and not (child.game.player_x == parent.game.player_x and child.game.player_y == parent.game.player_y and i != 0):
            parent.children.append(child)

        # for breakout
        # r, terminal,_= child.game.act(i)
        # child.q = child.q+r
        # if not terminal and not (child.game.pos == parent.game.pos and (i ==1 or i==2)):
        #     parent.children.append(child)


def selection(root):
    cur = root
    while len(cur.children) != 0:
        cur = best_child(cur)
    return cur

def best_child(parent,output=False):
    if not output:
        c=exploration_weight
    else:
        c=0
    children = parent.children
    max_val = -1000
    for i in range(len(children)):
        new_val=(children[i].q / children[i].n) + c * math.sqrt(
        (2 * math.log(parent.n)) / children[i].n)
        if new_val> max_val or(max_val==new_val and random.random()>0.5):
            pos=i
            max_val=new_val
    return children[pos]

def score_estimation(game):
    for _ in range(score_estimation_length):
        r,done,_=game.act(0)
        if r!=0 or done:
            return r
    return 0

def simulation(cur,max_simulation_length):
    done=False
    info=None
    path_length=0
    state=deepcopy(cur.game)
    delta=0
    while (not done) and (path_length<=max_simulation_length) and (not info):
        action=random.randint(0,len(state.action_map)-1)
        path_length+=1
        
        r,done,info=state.act(action)
        delta+=r
    
    if info:
        delta+=score_estimation(state)
    
    return delta
