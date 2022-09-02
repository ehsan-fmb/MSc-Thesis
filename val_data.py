import argparse
from environment import Environment
import pickle
from copy import deepcopy
import math
import time
import numpy as np


def expected_life(game,steps):

    if game.terminal:
        return math.pow(5,steps),math.pow(5,steps)

    if steps==0:
        return 0,1 

    death=0
    total=0
    state=game.state()[:,:,0:2]

    if state[max(0, game.player_x-1),game.player_y,1]==0:
        node=deepcopy(game)
        node.act(1)
        d,t=expected_life(node,steps-1)
        death=death+d
        total=total+t
    if state[min(9, game.player_x+1),game.player_y,1]==0:
        node=deepcopy(game)
        node.act(3)
        d,t=expected_life(node,steps-1)
        death=death+d
        total=total+t
    if state[game.player_x,max(1,game.player_y-1),1]==0:
        node=deepcopy(game)
        node.act(2)
        d,t=expected_life(node,steps-1)
        death=death+d
        total=total+t
    if state[game.player_x,min(8,game.player_y+1),1]==0:
        node=deepcopy(game)
        node.act(4)
        d,t=expected_life(node,steps-1)
        death=death+d
        total=total+t
    
    node=deepcopy(game)
    node.act(0)
    d,t=expected_life(node,steps-1)
    death=death+d
    total=total+t
    

    return death,total
        


def generate_dataset(name,size,steps,trials):
    dataset=[]
    env=Environment(name)
    
    for i in range(int(size)):
        env.random_reset()
        game=env.get_game()
        labels=[]
        for _ in range(trials):
            game.random=np.random.RandomState(np.random.randint(0,100000))
            death,total=expected_life(game,steps)
            if total>death:
                labels.append((total-death)/total)
        
        if len(labels)==6:
            label=min(labels)
            dataset.append([game.state(),game.ramp_index,label])
        
        if (i+1)%1000==0:
            print(i+1)

    with open("dataset/"+name+"_"+size+"_"+str(steps)+"_"+str(trials), "wb") as fp:
        pickle.dump(dataset, fp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--game',required=True)
    parser.add_argument('-s','--size',required=True)
    parser.add_argument('-l','--length',required=True)
    parser.add_argument('-t','--trial',required=True)
    args = parser.parse_args()
    generate_dataset(args.game,args.size,int(args.length),int(args.trial))    