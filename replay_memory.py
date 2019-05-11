import numpy as np
import random

class Memory():
    def __init__(self,memory_size,height,width,length):
        self.memory_size=memory_size
        self.screen_width,self.screen_height,self.history_length=width,height,length
        self.dims=(self.screen_height,self.screen_width,self.history_length)
        self.actions=np.empty(self.memory_size,dtype=np.uint8)
        self.rewards=np.empty(self.memory_size,dtype=np.integer)
        self.terminals=np.empty(self.memory_size,dtype=np.bool)
        self.screens=np.empty((self.memory_size,self.screen_height,self.screen_width,self.history_length),dtype=np.float16)
        #Every 4(histroy_length) frames as a memory screen

        self.batch_size=32
        self.count=0
        #Current memory number
        self.current=0
        #Cover point of new memory

    def add(self,screen,reward,action,terminal):
        #print(screen.shape,self.dims)
        assert screen.shape==self.dims

        self.actions[self.current]=action
        self.rewards[self.current]=reward
        self.terminals[self.current]=terminal
        self.screens[self.current]=screen
        self.count=max(self.count,self.current)
        self.current=(self.current+1)%self.memory_size

    def sample(self):
        assert self.count>self.batch_size+1
        #Memory must has more than two memories

        indexs=np.zeros(self.batch_size,dtype=np.integer)
        for i in range(self.batch_size):
            indexs[i]=random.randint(0,self.count-2)
        #Index,index+1 mean pre,new state
        pre=self.screens[indexs]
        action=self.actions[indexs]
        reward=self.rewards[indexs]
        new=self.screens[(indexs+1)%self.memory_size]
        terminal=self.terminals[indexs]
        return pre,action,reward,terminal,new
