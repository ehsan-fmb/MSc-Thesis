import argparse
import pickle
import torch
import matplotlib.pyplot as plt
import re
import torch.nn.functional as F
import random

# success settings for seaquest: -b 256 -e 100 -s 0.001 kernel size: 3 filters:16 two last layers units: 256,32
# success settings for asterix: -b 256 -e 200 -s 0.001 kernel size: 3 filters:8 last layer units: 128
# success settings for breakout: -b 256 -e 100 -s 0.001 kernel size: 3 filters:32 last layer units: 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(torch.utils.data.Dataset):
  def __init__(self, states, labels):
        self.labels = labels
        self.states = states

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, index):
        X = self.states[index]
        y = self.labels[index]

        return X, y

class Network(torch.nn.Module):
    def __init__(self, in_channels):

        super(Network, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden1= torch.nn.Linear(in_features=num_linear_units, out_features=256)
        self.fc_hidden2 = torch.nn.Linear(in_features=256, out_features=32)
        self.value = torch.nn.Linear(in_features=32, out_features=1)


    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc_hidden1(x.view(x.size(0), -1)))
        x = F.relu(self.fc_hidden2(x))
        return self.value(x)


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).float()

def show_dataset(address,name):
    with open(address, "rb") as fp:
        datalist = pickle.load(fp)
    
    print(len(datalist))
    seeds=[]
    values=[]
    for i in range(21):
        seeds.append(i*0.05)
        values.append(0)
    for data in datalist:
        values[int(data[1]/0.05)]+=1
    
    print(values)
    plt.plot(seeds,values,color="magenta")
    plt.savefig("figures/"+name+".png")
    #plt.show()


def preprocess(address):
    with open(address, "rb") as fp:
        datalist = pickle.load(fp)
    
    states=[]
    for data in datalist:
        states.append(data)
    print(len(states))
    
    i=0
    while i<len(states):
        j=i+1
        while j<len(states):
            if (states[i][0]==states[j][0]).all() and states[i][1]==states[j][1]:
                del states[j]
            j+=1
        i+=1  
    
    print(len(states))
    # with open(address, "wb") as fp:
    #     pickle.dump(states, fp)


def load_dataset(address):
    with open(address, "rb") as fp:
        datalist = pickle.load(fp)
    
    print(len(datalist))

    X=[]
    labels=[]
    for data in datalist:
        state=data[0]
        X.append(get_state(state))
        labels.append(torch.tensor(data[1],device=device))
    
    dataset=Dataset(X,labels)
    return dataset

def train(dataset,batch_size,max_epochs,channels,step_size,game):
    training_generator = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    network=Network(channels)
    if torch.cuda.is_available():
        network.cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=step_size)
    measure=torch.nn.MSELoss()
    for _ in range(max_epochs):
        for inputs, targets in training_generator:
            optimizer.zero_grad()
            yhat=network(inputs)
            targets=targets.view(targets.shape[0],1)
            loss=measure(yhat,targets)
            loss.backward()
            optimizer.step()
            print("loss: "+str(loss))
    
    torch.save(network.state_dict(), "value network/"+game+".pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name',required=True)
    parser.add_argument('-b','--bsize',required=True)
    parser.add_argument('-e','--epochs',required=True)
    parser.add_argument('-c','--channels',required=True)
    parser.add_argument('-s','--ssize',required=True)
    args = parser.parse_args()
    #show_dataset("dataset/"+args.name,args.name)
    #preprocess("dataset/"+args.name)
    dataset=load_dataset("dataset/Breakout/"+args.name)
    train(dataset,int(args.bsize),int(args.epochs),int(args.channels),float(args.ssize),re.split('_',args.name)[0])