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
import statistics
import random
import time

simulation_lengths=[5,6,7,8,9,10,11,12]
subgoal_threshs=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
safety_coefficients=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def run(budget,game,sticky_action,data_collecting,length,thresh,coeff,seed,method):
    env= Environment(game,random_seed=seed,sticky_action_prob=sticky_action)
    random.seed(seed)
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
    
    start=time.time()
    while not done:
        
        # deadlock case for breakout and
        if time.time()>start+1200:
            print("deadlock happened.")
            return score+int(random.random()*30),(thinking_time/actions)
        
        #env.display_state()
        root=State(env.get_game())
        
        #for textbook mcts only
        root.n+=1
        
        if method=="planner":
            option,found=planner.main(root,budget,val_network,data_collecting,length,seed,thresh,coeff)
        elif method=="mcts":
            option,found=textbook_mcts.main(root,budget,length,seed)
        
        
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
    # with open("results/"+game+"/100 runs/planner", "rb") as fp:
    #         planner = pickle.load(fp)
    with open("results/"+game+"/100 runs/mcts", "rb") as fp:
            mcts = pickle.load(fp)
    
    # dataset_planner=[]
    # for data in planner:
    #     dataset_planner.append([i*1000 for i in data[2]])
    dataset_mcts=[]
    for data in mcts:
        dataset_mcts.append(data[1])

    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(dataset_mcts,showmeans=True)
    ax.set_xticklabels(simulation_lengths)
    plt.xlabel("horizon h")
    plt.ylabel("score",rotation=0)
    #plt.subplots_adjust(left=0.25,bottom=0.2)
    ax.yaxis.set_label_coords(-0.1,0.42)
    #plt.legend()
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5',axis="y")
    ax.grid(which='minor', linestyle=':', linewidth='0.5',axis="y")
    ax.tick_params(which='minor',bottom=False)
    plt.savefig("figures/"+game+"/100 runs/mcts average score for seaquest.png")
    #plt.show()
    

def single_setting_experiment(iterations,budget,game,sticky,data_collecting,method):
    score=0
    thinking_time=0
    for _ in range(iterations):
        s,t=run(budget,game,sticky,data_collecting,10,0.6,0.9,method)
        score=score+s
        thinking_time=thinking_time+t
        
    print("average score: "+str(score/iterations))
    print("average thinking time: "+str(thinking_time/iterations))
    print("*"*20)


def results_with_two_parameters(iterations,budget,game,sticky,data_collecting,method):
    results=[]
    for subgoal_thresh in subgoal_threshs:
        for coeff in safety_coefficients:
            print ("subgoal safety threshold: "+str(subgoal_thresh)+"  safety coefficient: "+str(coeff))
            score=0
            scores=[]
            times=[]
            for _ in range(iterations):
                s,t=run(budget,game,sticky,data_collecting,6,subgoal_thresh,coeff,int(random.random()*100),method)
                score=score+s
                scores.append(s)
                times.append(t)

            
            std=statistics.stdev(scores)
            results.append([subgoal_thresh,coeff,scores,times])
            print("average score: "+str(score/iterations))
            print("std: "+str(std))
            print("*"*20)
        
        with open("results/"+game+"/100 runs/"+method+"_safety1", "wb") as fp:
            pickle.dump(results, fp)

def results_with_one_parameter(iterations,budget,game,sticky,data_collecting,method):
    results=[]
    for length in simulation_lengths:
        print("length: "+str(length))
        times=[]
        scores=[]
        for _ in range(iterations):
            s,t=run(budget,game,sticky,data_collecting,length,0.4,0.9,int(random.random()*100),method)
            times.append(t)
            scores.append(s)
        
        std=statistics.stdev(scores)
        average=sum(scores)/len(scores)
        results.append([length,scores,times])
        print("average score: "+str(average))
        print("std: "+str(std))
        print("*"*20)
        
    with open("results/"+game+"/100 runs/"+method+"2", "wb") as fp:
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
    
    #single_setting_experiment(int(args.trial),float(args.budget),args.game,float(args.sticky),data_collecting,args.method)
    #results_with_two_parameters(int(args.trial),float(args.budget),args.game,float(args.sticky),data_collecting,args.method)
    #results_with_one_parameter(int(args.trial),float(args.budget),args.game,float(args.sticky),data_collecting,args.method)
    plot(args.game)