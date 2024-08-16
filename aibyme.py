# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:15:45 2020

@author: Satyam Garg
"""

#CREATING SELF DRIVING CAR WITH AI

#IMPORTING LIBRARIES

import numpy as np #for multidimensional arrays
import random #for random inputs
import os #for saving and loading the brain
import torch #for implementing nn
import torch.nn as nn #using inbuilt neural networks
import torch.nn.functional as F #using inbuilt cost and loss functions
import torch.optim as optim #for optimization
import torch.autograd as autograd #for using variable
from torch.autograd import Variable #for using tensors

#Creating the architecture of Neural Network

class Network(nn.Module): #inheriting module class from nn library
    def __init__(self,input_size,nb_action):
        super(Network, self).__init__() #initializing all funcitons of nn module in network class
        self.input_size=input_size #defining input size
        self.nb_action=nb_action #defining number of actions
        fc1 = nn.Linear(input_size,30) #connecting all neurons of input layer with the hidden layer having input size 30. Size best by trial can be changed also
        fc2 = nn.Linear(30,nb_action) #connecting all neurons of hidden layer with the output layer
        
    def forward(self,state):
        x = F.relu(self.fc1(state)) #activating the neurons of the hidden layer using rectified layer unit in functional module by passing the current state of object and fc1
        q_values = self.fc2(x) #getting q_values from the hidden layer of neurons using full connection 2
        return q_values

#CREATING EXPERIENCE REPLAY

#We are creating this class so that we can choose randomly a sample from certain last events and then make the decision for upcoming event as we know if we choose from only the past single event, the results aren't that promising. So we will create a memory class and store some of the past events, say 100.
class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity = capacity #capacity defines number of past events we want to consider so as to make the upcoming decision
        self.memory = [] #it is the list storing past capacity amount of events
    #tuples are of form (last_state,new_state,last_action,last_reward)
    def push(self,event): #this function pushes the event into the memory and makes sure that size of our list doesn't exceed the capacity
        self.memory.append(event) #adding latest event to list
        if len(self.memory) > self.capacity:
            del self.memory[0] #deleting the event that was added a capacity times ago so as to make sure list size doesn't exceed capacity
    
    def sample(self, batch_size): #we will pick random samples of size=batch_size from the memory list, which we will use to make upcoming decision
        samples = zip(*random.sample(self.memory,batch_size)) #here what we are doing is that suppose we have a list of tuples. eg. {(1,2,3),(4,5,6)} were tuple can be decoded as (state,action, reward), what zip would do is to encode it as random sample such {(1,2),(3,4),(5,6)}. Here we are doing this so that first tuple represents all states second all actions and third all rewards so that we can pass it to torch
        return map(lambda x: Variable(torch.cat(x,0)),samples) #here Variable is converting it into form of (tensor,gradient), so that we can distinguish between both of them and torch is concatenating our random variable x in one dimension so that every thing can be alligned, first dimesion has index 0
        #here x will become equal to sample once lambda function is applied
#Implementing Deep Q- learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma #it represents the discount factor gamma
        self.reward_window=[] #it represents the reward window where certain amount of previous rewards will be kept
        self.model = Network(input_size, nb_action) #initializing object of Network class and storing it in the model of our AI
        self.memory = ReplayMemory(100000) #it initialises object of ReplayMemory class given it a capacity of 100000
        self.optimizer = optim.Adam(self.model.parameters, lr=0.001) #here we are optimizing the model parameters and using learning rate(lr) as 0.001
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #unsqueeze here helps us to create a fake dimension
        self.last_action = 0 # it can be 0, 1 and 2 according to straight, right and left respectively
        self.reward=0 #as reward is number we are initializing it to zero
    
    #here in this section we will be using softmax function, the use of the function is to make sure that we take best action in current state and also keep exploring the track
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state,volatile=True))*7) #here *7 is for temperature parameter which increases the difference between highest prob and lowest prob, volatile=true enables us to drop the gradient descent and make the algorithm faster
        action = probs.multinomial() #to get a random value from the probability distribution of the given probabilities
        return action.data[0,0] #Since the action we got in previous step is a torch tensor, we are using data to simply return a value
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1) #here since output of neural networks in q values we will have to gather the best of these actions and since action is not in fake dimension whereas batch state is we will create another dimension for output and then later on squeeze as we want final output to be a simple vvariable
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward #this is used to calculate target that will be used in order to calculate loss
        td_loss = F.smooth_l1_loss(outputs, target) #its input is the outputs we got and the target as loss is 1/2(target - outputs)^2
        self.optimizer.zero_grad() #backpropagate the error and optimize the weights accordingly and perform stochastic gradient descent and in python we have to reinitialize the loop after each iteration using zero_grad
        td_loss.backward() #this is for backpropagation of error and retain variable=true helps us to save some memory
        self.optimizer.step() #this simple function helps to update the weight
    
    #everything we need to update after taking an action
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) #here the new state will be the signal we are getting from our map and we have to convert it into torch tensor    
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #this will stored in memory
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)