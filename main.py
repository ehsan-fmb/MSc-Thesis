from environment import Environment
from state import State
import planner
import time
import numpy as np
import argparse
from val_learn import Network
import textbook_mcts
import torch

# for asterix:
# in_channels=3

# for freeway
in_channels=7

def run(budget,game,sticky_action):
    env= Environment(game,random_seed=np.random.randint(0,100000),sticky_action_prob=sticky_action)
    env.reset()
    done=False
    score=0
    lifetime=time.time()
    
    val_network=Network(in_channels)
    val_network.load_state_dict(torch.load("value network/"+game+".pt",map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        val_network.cuda()
    
    while not done:
        #env.display_state()
        root=State(env.get_game())
        
        #for textbook mcts only
        root.n+=1
        
        #start=time.time()
        option,found=planner.main(root,budget,val_network)
        #option,found=textbook_mcts.main(root,budget)
        #end=time.time()
        
        if found:
            r,done=planner.option_running(env,option)
            score=score+r
        else:
            r,done,_=env.act(option)
            score=score+r
        
        #print(str(end-start)+" sec")
        #print("#"*30)
    
    #env.close_display()
    lifetime=time.time()-lifetime
    return score,lifetime



def analyze(iterations,budget,game,sticky):
    scores=[]
    lifetimes=[]
    for i in range(iterations):
        print("Trial: "+str(i+1))
        score,lifetime=run(budget,game,sticky)
        scores.append(score)
        lifetimes.append(lifetime)
        
    print("*"*20)
    print("scores:"+str(scores))
    print("life times: "+str(lifetimes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--trial',required=True)
    parser.add_argument('-b','--budget',required=True)
    parser.add_argument('-g','--game',required=True)
    parser.add_argument('-s','--sticky',required=True)
    args = parser.parse_args()
    analyze(int(args.trial),float(args.budget),args.game,float(args.sticky))