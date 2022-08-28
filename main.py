from environment import Environment
from state import State
import planner
import time
import numpy as np
import argparse
from val_learn import Network
import torch

#for asterix, number of input channels for value network is 3.


def run(budget,game):
    env= Environment(game,random_seed=np.random.randint(0,100000))
    env.reset()
    done=False
    score=0
    lifetime=time.time()
    if game=="asterix":
        in_channels=3
    val_networks=[]
    for _ in range(len(env.get_game().action_map)):
        network=Network(in_channels)
        network.load_state_dict(torch.load("value network/asterix.pt"))
        val_networks.append(network)
    while not done:
        env.display_state()
        root=State(env.get_game())
        #for textbook mcts only
        root.n+=1
        #start=time.time()
        option,found=planner.main(root,budget,val_networks)
        #action=textbook_mcts.main(root,budget)
        #end=time.time()
        if found:
            for action in option:
                r,done=env.act(action)
                score=score+r
        else:
            r,done=env.act(option)
            score=score+r
        #print(str(end-start)+" sec")
        #print("#"*30)
    
    env.close_display()
    lifetime=time.time()-lifetime
    return score,lifetime



def analyze(iterations,budget,game):
    total_score=0
    total_lifetime=0
    for _ in range(iterations):
        score,lifetime=run(budget,game)
        total_score+=score
        total_lifetime+=lifetime
    print("total score: "+str(total_score))
    print("total lifetime: "+str(total_lifetime))
    print("rate: "+str(total_score/total_lifetime))
    print("*"*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--trial',required=True)
    parser.add_argument('-b','--budget',required=True)
    parser.add_argument('-g','--game',required=True)
    args = parser.parse_args()
    analyze(int(args.trial),float(args.budget),args.game)