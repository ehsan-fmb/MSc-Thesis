from distutils.log import info
import math
import time
from state import State
import random
from copy import deepcopy



exploration_weight=math.sqrt(2)
max_simulation_length=10



def main(root,budget):
    start = time.time()
    simulations=0
    while time.time() <= start + budget:
        cur = selection(root)
        if cur.n<1:
            delta = simulation(cur)
            simulations+=1

        else:
            expansion(cur)
            # no children condition should be checked!
            if len(cur.children)>0:
                cur=cur.children[0]
            
            delta=simulation(cur)
            simulations+=1
        backpropagation(root,cur,delta)


    #print(simulations)
    if len(root.children)>0:
        return best_child(root).action_index,False
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
        # r, terminal,_= child.game.act(i)
        # child.q = child.q+r
        # if not terminal and not (child.game.player_x == parent.game.player_x and child.game.player_y == parent.game.player_y and i != 0):
        #     parent.children.append(child)

        
        # for freeway
        # r, _,terminal= child.game.act(i)
        # child.q = child.q+r
        # if not terminal:
        #     parent.children.append(child)

        # for breakout
        r, terminal,_= child.game.act(i)
        child.q = child.q+r
        if not terminal and not (child.game.pos == parent.game.pos and i != 0):
            parent.children.append(child)

def selection(root):
    cur = root
    while len(cur.children) != 0:
        cur = best_child(cur)
    return cur

def best_child(parent):
    c=exploration_weight
    children = parent.children
    max_val = -1000
    for i in range(len(children)):
        new_val=(children[i].q / children[i].n) + c * math.sqrt(
        (2 * math.log(parent.n)) / children[i].n)
        if new_val> max_val or(max_val==new_val and random.random()>0.5):
            pos=i
            max_val=new_val
    return children[pos]

def simulation(cur):
    done=False
    info=False
    path_length=0
    state=deepcopy(cur.game)
    delta=0
    while (not done) and (path_length<=max_simulation_length) and (not info):
        action=random.randint(0,len(state.action_map)-1)
        path_length+=1
        
        r,done,info=state.act(action)
        delta+=r
    return delta
