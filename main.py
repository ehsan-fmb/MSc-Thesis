import importlib
from environment import Environment
from state import State
import matplotlib.pyplot as plt
import argparse
from val_learn import Network
import textbook_mcts
import torch
import pickle
import numpy as np

simulation_lengths=[5,6,7,8,9,10,11,12]
subgoal_threshs=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def run(budget,game,sticky_action,data_collecting,length,thresh,method):
    env= Environment(game,random_seed=np.random.randint(0,100000),sticky_action_prob=sticky_action)
    env.reset()
    planner=importlib.import_module("planner_"+game)
    done=False
    score=0
    thinking_time=0
    actions=0

    # defining network input channels
    if game=="asterix":
        in_channels=3
    elif game=="seaquest":
        in_channels=11
    elif game=="breakout":
        in_channels=4
    
    val_network=Network(in_channels)
    if method=="planner" and  not data_collecting:
        val_network.load_state_dict(torch.load("value network/"+game+".pt",map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        val_network.cuda()
    
    while not done:
        #env.display_state()
        root=State(env.get_game())
        
        #for textbook mcts only
        root.n+=1
        
        if method=="planner":
            option,found=planner.main(root,budget,val_network,data_collecting,length,thresh)
        elif method=="mcts":
            option,found=textbook_mcts.main(root,budget,length)
        
        
        if found:
            r,path_length,done=planner.option_running(env,option)
            score=score+r
            actions=actions+path_length
            thinking_time=thinking_time+budget
            
        else:
            r,done,_=env.act(option)
            # state=env.state()
            # for i in range(4):
            #     print(state[:,:,i])
            #     print()
            #     print()
            # print("*"*20)
            score=score+r
            actions=actions+1
            thinking_time=thinking_time+budget
    
    #env.close_display()
    return score,(thinking_time/actions)

def plot(game):
    with open("results/"+game+"/"+game+"_planner", "rb") as fp:
            planner = np.array(pickle.load(fp))
    with open("results/"+game+"/"+game+"_mcts", "rb") as fp:
            mcts = np.array(pickle.load(fp))
    
    plt.figure()
    plt.plot(simulation_lengths,planner[:,0],color="green",label="planner")
    plt.plot(simulation_lengths,mcts[:,0],color="red",label="mcts")
    plt.title("Average Score vs Simulation length",fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/"+game+"/average score vs simulation length.png")
    #plt.show()

    plt.figure()
    plt.plot(simulation_lengths,planner[:,1],color="green",label="planner")
    plt.plot(simulation_lengths,mcts[:,1],color="red",label="mcts")
    plt.title("Average Thinking Time",fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/"+game+"/average thinking time.png")
    # plt.show()
    

def score_checking(iterations,budget,game,sticky,data_collecting,method):
    for thresh in subgoal_threshs:
        print("thresh: "+str(thresh))
        score=0
        thinking_time=0
        for _ in range(iterations):
            s,t=run(budget,game,sticky,data_collecting,10,thresh,method)
            score=score+s
            thinking_time=thinking_time+t
        
        print("average score: "+str(score/iterations))
        print("average thinking time: "+str(thinking_time/iterations))
        print("*"*20)


def analyze(iterations,budget,game,sticky,data_collecting,method):
    results=[]
    #for subgoal_thresh in subgoal_threshs:
    for length in simulation_lengths:
        print("length: "+str(length))
        score=0
        thinking_time=0
        for _ in range(iterations):
            s,t=run(budget,game,sticky,data_collecting,length,0.6,method)
            score=score+s
            thinking_time=thinking_time+t
        
        results.append([score/iterations,(thinking_time/iterations)*1000])
        print("*"*20)
        print("average score: "+str(score/iterations))
        print("average thinking time: "+str(thinking_time/iterations))
        
    with open("results/"+game+"/"+game+"_"+method, "wb") as fp:
        pickle.dump(results, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--trial',required=True)
    parser.add_argument('-b','--budget',required=True)
    parser.add_argument('-g','--game',required=True)
    parser.add_argument('-s','--sticky',required=True)
    parser.add_argument('-d','--data',required=True)
    parser.add_argument('-m','--method',required=True)
    args = parser.parse_args()
    if args.data=="True":
        data_collecting=True
    else:
        data_collecting=False
    
    #score_checking(int(args.trial),float(args.budget),args.game,float(args.sticky),data_collecting,args.method)
    #analyze(int(args.trial),float(args.budget),args.game,float(args.sticky),data_collecting,args.method)
    plot(args.game)