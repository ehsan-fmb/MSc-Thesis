from environment import Environment
from state import State
import multiprocessing
import planner
import time
import numpy as np
import argparse
from val_learn import Network
import textbook_mcts
import torch

#for asterix, number of input channels for value network is 3.

def run(budget,game,sticky_action):
    env= Environment(game,random_seed=np.random.randint(0,100000),sticky_action_prob=sticky_action)
    env.reset()
    done=False
    score=0
    lifetime=time.time()
    
    if game=="asterix":
        in_channels=3
    val_network=Network(in_channels)
    #network.cuda()
    val_network.load_state_dict(torch.load("value network/asterix.pt"))
    
    while not done:
        #env.display_state()
        root=State(env.get_game())
        
        #for textbook mcts only
        root.n+=1
        
        start=time.time()
        option,found=planner.main(root,budget,val_network)
        #option,found=textbook_mcts.main(root,budget)
        end=time.time()
        if found:
            r,done=planner.option_running(env,option)
            score=score+r
        else:
            r,done=env.act(option)
            score=score+r
        print(str(end-start)+" sec")
        #print("#"*30)
    
    #env.close_display()
    lifetime=time.time()-lifetime
    return score,lifetime



def analyze(iterations,budget,game,sticky):
    total_score=0
    total_lifetime=0
    for i in range(iterations):
        print("Trial: "+str(i+1))
        score,lifetime=run(budget,game,sticky)
        total_score+=score
        total_lifetime+=lifetime
    print("*"*20)
    print("results:")
    print("total score: "+str(total_score))
    print("total lifetime: "+str(total_lifetime))
    print("rate: "+str(total_score/total_lifetime))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--trial',required=True)
    parser.add_argument('-b','--budget',required=True)
    parser.add_argument('-g','--game',required=True)
    parser.add_argument('-s','--sticky',required=True)
    args = parser.parse_args()
    analyze(int(args.trial),float(args.budget),args.game,float(args.sticky))