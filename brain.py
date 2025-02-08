#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:49:21 2025

@author: kali

self driving car
"""

# importing libraries

import numpy as np # numpy provides array object which is 50x faster than normal array object and also provides speed to perform large complex mathematical operations
import random as rnd # random is for random samples for experience replay, means while moving forword it stores past journey exp in its memody in form of experience batches
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
    def forward(self, state):
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
        if not hasattr(self, 'memory') or len(self.memory) < batch_size:
            raise ValueError("Not enough samples in memory")

        
        sample = zip(*rnd.sample(self.memory, batch_size))  # Ensure random.sample is used correctly
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
        #sample = zip(*random.sample(self.memory, batch_size))
        
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
        with torch.no_grad():  # Disable gradients for inference
            probabilities = F.softmax(self.model(state) * 1000, dim=1)  # Ensure softmax works correctly
        
        action = probabilities.multinomial(num_samples=1)  # Fix: Add num_samples=1
        return action.item()  # Fix: Use `.item()` to extract a Python scalar
        """
        softmax provides best action to play but also same time we will explore different action we can achieve this using softmax which generate distribution of probabilities for each q values
        """
        #probabilities = F.softmax(self.model(Variable(state, volatile = True)) * 100) # t=100 temperature parameter, the higher is the temperature parameter the higher probability of winning q value
        # now from softmax we take random draw from distribution to play final action
        #action = probabilities.multinomial() # this multinomial return pytorch variable with fake batch
        #return action.data[0,0]
    
    """ 
    we will learn deep neural network that is inside our artificial intelligence, means we are going to do forword and backward propogation
    batch_state is for current state, means our transistion are in form of batches
    """
    def learn(self, batch_state, batch_next_state, batach_reward, batch_action):
        """
        as we have experience replay memory which has batches we will use those past transistion and neural network will learn from those past transistion batches
        we are taking out
        in below line we want to get model output of input state means batch_state as model is expecting a batch of input state
        but self.model gives output of all posible action we want only the action that decided by network at each time
        but batch_state has fake dimension corresponding to this batch so we should kill that fake dimension using unsqueeze 1 will correspod to fake dim of the action 0 correspond to fake dim of state
        now we have our output we are out of the network but we dont want output in a batch we want simple tensors so we will use squeeze fake dim of action using squeeze
        """
        batch_action = torch.clamp(batch_action, 0, self.model(batch_state).size(1) - 1)
        output = self.model(batch_state).gather(1, batch_action.long().unsqueeze(1)).squeeze(1)
        """
        now we need next output to compute the target, and target = reward + gamma times max of q values
        next output is result of our neural network which has all the q values but we want to max of all q values so we use detach fn 
        and after detaching all the q values we want best action so we use max fn and provide 1 as action dim
        and [0] stand for q value for next_state that correspond to 0
        """
        next_output = self.model(batch_next_state).detach().max(1)[0]
        # target compute
        target = self.gamma*next_output+batach_reward
        # calculating loss, error of the prediction, td stands for temporal difference loss, this is nothing but the hubble loss
        # so we need to provide prediction means output of the neural network and second arg is target 
        # and now we have the loss error we can back propogate to neural network to update the weight with stocastic gradiant descent
        td_loss = F.smooth_l1_loss(output, target)
        """
        now optimizer comes in the picture we will apply it on loss error to perform stocastic gradiant in the sense of data weights
        and while using pytorch we need reinitialize optimizer at each iteration of loop of stocastic gradiant descent and this can be done by zero_grad at each iteration of the loop
        """
        self.optimizer.zero_grad()
        """
        now we will perform back propgation with our optimizer, means we take out td_loss and feed in to the neural network
        """
        td_loss.backward(retain_graph=True) # this will improve back propogation also we need to free memory because we will go saveral time on the loss and this will improve training performance
        """
        last step is to update the weights according to back propogation using step fn we will update the weights
        """
        self.optimizer.step()
        
    
    """
    update fn update everything as soon as the ai reaches new state
    when it reches new state last action will be new action and last reward will be new reward
    we will also use select fn to select action to play besides making all updates
    """
    def update(self, reward, new_signal):
        """
        we will get new state and new_signal is the new_state which is an arr of five element and after getting new state we will update it inside the replay memory 
        """
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) # to add dim to the batch
        """
        we added a new transistion to memory transistion consist three dir and 2 orientation
        """
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        """
        now we are in new state of the environment so we will play am action
        """
        action = self.select_action(new_state) # in this line we moved to a new state
        """
        now after playing an action we make ai to learn, and ai will learn from random samples batches of the memory
        """
        if len(self.memory.memory) > 100:
            # ai can learn
            """
            and here we will get random sample of transistion from the sample fn, and sample fn returns different batches to state at time t + 1 action at time t and rewards at time t
            """
            batch_state, batch_next_state, batach_reward, batch_action = self.memory.sample(100) # it will learn from the last 100 samples
            self.learn(batch_state, batch_next_state, batach_reward, batch_action) # here we feed last 100 samples of batches to learn
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        """
        now we need to update reward window aswell
        """
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window) + 1.)
    
    """
    save the model means save the last state of the model and after login on it will start from the last checkpoint
    we will save last weight of the last iteration of the model 
    so we will take last version of the states and also last version of the optimizer
    """
    def save(self):
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict}, 'last_brain.pth')
    
    """
    load fn will load what was save from the save fn
    """
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict']) # here we updated weight of the model
            self.optimizer.load_state_dict(checkpoint['optimizer']) # here we updated parameter of the of the optimizer
            print('done !')
        else:
            print('no checkpoint present....')
        
            
            
            