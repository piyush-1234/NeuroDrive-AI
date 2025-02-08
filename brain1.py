#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:49:21 2025

@author: kali

self driving car
"""

# importing libraries

import numpy as np # numpy provides array object which is 50x faster than normal array object and also provides speed to perform large complex mathematical operations
from random import random, randint # random is for random samples for experience replay, means while moving forword it stores past journey exp in its memody in form of experience batches
import os # os is use for to load the model when system is shutdown and we want to load the model from the last state it was trained
import torch # use for implementing neural network because it handle dynamic graphs
import torch.nn as nn # torch.nn contains all the modules that essential for neural network, also contains deep q network which takes 3 sensors +orientation and -orientation 
import torch.nn.functional as F # functional pckg from nn use to implement neural network ex we will use uber loss function
import torch.optim as optim # use to perform some stochastic random gradients
import torch.autograd as autograd # use to import variable class to make some conversion from tensors which are like more advance arrays to avoid all that contains a gradient, so its like we dont wanna have only a tensors by itslef, we want to put the tensor in variable that will also contains a variable
from torch.autograd import Variable

# creating the architecture of neural network

class Network(nn.Module):
        
    """
    init function has three input parameter
    first self
    second no of input neurons-input_size is 5 (3 signals and +orientation and -orientation) we could have gone for 360 signals but 3 signals are fine for self driving
    third no of output neurons-nb_action has three actions left straight and right
    """
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size;
        self.nb_action = nb_action;
        
        """
        fc1 stands for full connection 1 means all the neurons of the input layers fully connected to all the neurons of the hidden layer
        to make fc1 we use Linear function from nn
        Linear fn takes 2 args first no of input neurons which is 5 (3 directions 2 orientation) and 2nd hidden layer neurons which we take 30 neurons for better result
        """
        noOfHiddenNeurons = 30;
        self.fc1 = nn.Linear(input_size, noOfHiddenNeurons) # neural network 1
        self.fc2 = nn.Linear(noOfHiddenNeurons, nb_action)  # neural network 2
    
    """
    forword function activates neural network and also perform forword propogation
    this forword fn not only activate the neurons but also return q values for each posible action depending on input state
    """
    def forword(self, state):
        """
        first we activate fc1 we provide input state as input to go from input neuron to hidden neuron
        relu is a rectifier fn to activate hidden neuron
        """
        x = F.relu(self.fc1(state))
        
        """
        q_values will be out put neurons 
        we provided input as neurons of the fc1
        """
        q_values = self.fc2(x);
        
        return q_values;
    

# replay memory or implementation of experience replay to store past eperience in terms of random batches
class ReplayMemory(object):
    def __init__(self, capacity):
        """
        this is max no of past transition we want in our memory event
        """
        self.capacity = capacity
        """
        memory will contains last 100 transitions
        it will be array
        """
        self.memory = []
        
    """
    push function
    first it will append new transition in the memory second it will maintains 100 transistion all the time
    """
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    """
    sample function
    this fn provides past transition samples
    batch_size is nothing fixed sample size
    """
    def sample(self, batch_size):
        """
        varibale sample: this is just contains the sample of the memory
        we will get sample from memory
        we also need batch size and the samples are going to get contains batch size elements
        and we need pytorch to get the good format 
        
        sample fn from random library helps to get batch samples of fixed batch size
        
        zip fn helps to reshape sample batches how it does see below ex
        if list = {(1,2,3), (4,5,6)} then what zip(*list) = {(1,4),{2,3},(5,6)} so this is kind of reshaping of batch samples it does
        means every event consist of state, action and reward which is being stored in memory so sample batch has {1,2,3} which means s1,a1,r1 and s2,a2,r2 after reshaping sample batches will be s1,s2|a1,a2|r1,r2
        and further these random batches will wrapped in pytorch variable which contains tensor and gradiant
        """
        sample = zip(*random.sample(self.memory, batch_size))
        
        """
        x is the variable of the function lambda this will convert samples into torch variable and variable fn convert torch tensor to variable which tensor and gradiant
        and variable inside of which we convert x, x is sample after aplied lambda to it
        """
        return map(lambda x: Variable(torch.cat(x,0)), sample)
    
    
# implementing deep q learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        # gamma is a delay coeficent
        self.gamma = gamma;
        self.reward_window = []
        # creating neural network
        self.model = Network(input_size, nb_action) # we created 1 neural network with deep q learning model
        #creating memory
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # last state is a vector of 5 dimension and one fake dimension for batch, fake dim is at 0
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0;
        self.last_reward = 0;
        
    # implementing select action fn it will take input state 
    def select_action(self, state):
        """
        softmax provides best action to play but also same time we will explore different action we can achieve this using softmax which generate distribution of probabilities for each q values
        """
        probabilities = F.softmax(self.model(Variable(state, volatile = True)) * 7) # t=7 temperature parameter, the higher is the temperature parameter the higher probability of winning q value
        # now from softmax we take random draw from distribution to play final action
        action = probabilities.multinomial() # this multinomial return pytorch variable with fake batch
        return action.data[0,0]
    
    # we will learn deep neural network that is inside our artificial intelligence, means we are going to do forword and backward propogation
    def learn(self, batch_state, batch_next_state, batach_reward, batch_action):
        """
        as wwe have experience replay memory which has batches we will use those past transistion and learn neural network
        self.model will get output of all posible actions i.e 0,1,2 but thats not we want so gether fn comes in picture we only want action that is decided by the network to play at each time
        so we will gether only 1 action to play which will best
        the batch action has the same dimention as the batch state to ahieve this we used unsqueeze and last thing we need to do is to kill fake batch with squeeze why because we are out of the neural network
        we dont need to get back we have our output in form of simple tensor means vector of output
        """
        output = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        """
        next_output is going to be the result of our neural network
        now we will get max of all the q values so first we need to detach of all q values and we will get max q value from the next state which represented by 0
        """
        next_output = self.model(batch_next_state).detach().max(1)[0]
        """
        target = reward + gamma time next_output
        """
        target = self.gamma * next_output + batach_reward
        """
        now lets find temporal difference loss
        """
        td_loss = F.smooth_l1_loss(output, target)
        
        """
        now we will back propogate the loss to the neural network to update the wright with stochastic gradiant descent
        zero_grad will reinitialize the optimizer at each iterationof the loop
        """
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True) # this will improve back propogation improve memory usage
        self.optimizer.step() # this will update the weights
        
    """
    update fn update everything till date after ai reached new state, last action will current action last state will current and so
    """
    def update(self, reward, new_signal):
        """
        first we will convert normal array to torch tensor and we will create fake dimension of the batch with unsqueeze(0)
        """
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) 
        """
        after reaching to new state we will update memory
        """
        self.memory.push(self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))
        # now we play new action using select action
        action = self.select_action(new_state)
        
        """
        now we have to maintain memory aswell with updated action
        """
        if (self.memory.memory) > 100:
            """
            we will get random sample of 100 and ai will learn from them
            """
            batch_state, batch_next_state, batach_reward, batch_action = self.memory.sample(100) 
            """
            and after taking 100 batches from memory we will feed them to ai to learn
            """
            self.learn(batch_state, batch_next_state, batach_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        
        return action
    
    
    """
    score fn for compute the sliding window of the reward
    """
    def score(self):
        return sum(self.reward_window)(/len(self.reward_window)+1)
    
    """
    save fn save the model save the brain of the car whenever app get shutdown
    we will save last weights of the last iteration so we will save mode and optimizer
    """
    def save(self):
        torch.save({"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, 'last_brains.pth')
        
    
    """
    load the model whenever app goes back
    """
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading last checkpoint")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("=> no checkpoint found")
        
        