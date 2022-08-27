import argparse
from environment import Environment
import pickle
from copy import deepcopy
import math
import time



def expected_life_a(game,steps):

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
        d,t=expected_life_a(node,steps-1)
        death=death+d
        total=total+t
    if state[min(9, game.player_x+1),game.player_y,1]==0:
        node=deepcopy(game)
        node.act(3)
        d,t=expected_life_a(node,steps-1)
        death=death+d
        total=total+t
    if state[game.player_x,max(1,game.player_y-1),1]==0:
        node=deepcopy(game)
        node.act(2)
        d,t=expected_life_a(node,steps-1)
        death=death+d
        total=total+t
    if state[game.player_x,min(8,game.player_y+1),1]==0:
        node=deepcopy(game)
        node.act(4)
        d,t=expected_life_a(node,steps-1)
        death=death+d
        total=total+t
    
    node=deepcopy(game)
    node.act(0)
    d,t=expected_life_a(node,steps-1)
    death=death+d
    total=total+t
    

    return death,total
        


def generate_dataset(name,size,steps):
    dataset=[]
    env=Environment(name)
    # determine how we want to calculate the expectation of life for different games
    if name=="asterix":
        func=expected_life_a
    
    for i in range(int(size)):
        env.random_reset()
        game=env.get_game()
        death,total=func(game,int(steps))
        label=(total-death)/total
        #validating
        # print("label: "+str(label))
        # print("move speed: "+str(game.move_speed))
        # print("spwan speed: "+str(game.spawn_speed))
        # s=game.state()
        # for i in range(2):
        #     print(s[:,:,i])
        #     print("*"*40)
        dataset.append([game.state(),game.ramp_index,label])

    with open("dataset/"+name+"_"+size+"_"+steps, "wb") as fp:
        pickle.dump(dataset, fp)


def load_dataset(address):
    with open(address, "rb") as fp:
        b = pickle.load(fp)
    return b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--game',required=True)
    parser.add_argument('-s','--size',required=True)
    parser.add_argument('-l','--length',required=True)
    args = parser.parse_args()
    generate_dataset(args.game,args.size,args.length)    