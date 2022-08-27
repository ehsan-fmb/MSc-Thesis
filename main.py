from environment import Environment
import numpy as np
import re
import pickle


with open("dataset/asterix_10000_5", "rb") as fp:
    datalist = pickle.load(fp)

min=1
max=0
avg=0
index=-1
indexx=-1
labels=[]
for i in range(len(datalist)):
    avg=avg+datalist[i][2]
    if datalist[i][2]>=max:
        max=datalist[i][2]
    if datalist[i][2]<=min:
        min=datalist[i][2]
        indexx=index
        index=i
        
    labels.append(datalist[i][2])

avg=avg/len(datalist)
print("average: "+str(avg))
print("min: "+str(min))
print("max: "+str(max))
print("second min: "+str(labels[indexx]))

for i in range(2):
    print(datalist[indexx][0][:,:,i])
    print("*"*20)


if __name__ == '__main__':
    # a=re.split('_',"asterix_10000_5")[0]
    # print(a)
    # env= Environment("asterix")
    # env.reset()
    # state=env.state()
    # ramping=np.ones((state.shape[0],state.shape[1]))*5
    # print(ramping.shape)
    # print(state.shape)
    # state=np.dstack((state,ramping))
    # print(state.shape)
    # for i in range(5):
    #     print(state[:,:,i])
    #     print("*"*40)
    # print(start[:,:,0])
    # print(env.get_game().player_x)
    # print(env.get_game().player_y)
    # print("*"*50)
    # env.act(1)
    # next=env.state()
    # print(next[:,:,0])
    # print(env.get_game().player_x)
    # print(env.get_game().player_y)
    # print(10//3)
    pass