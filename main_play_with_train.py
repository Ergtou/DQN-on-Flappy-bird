from agent import *
from env import *
import tensorflow as tf

def file_log(greedy,score):
    with open("train.log","a") as f:
        f.write("E_greedy is {}\n".format(greedy))
        f.write("Best_score is {}\n".format(score))

def main(isTrain=True):
    if isTrain==True:
        env=Env(fps=120)
        ai=Agent(env)
        initial_e,final_e=1,0.1
        temp=5
        while True:
            ai.play()
            #Execuate update
            if ai.times>temp and ai.e_greedy>final_e:
                ai.e_greedy -= (initial_e-final_e)/100000
            if ai.times>temp:
                ai.q_learning_mini_batch()
                
            if ai.times>temp and ai.times%temp==0:
                ai.update_target_q_network()
                ai.save_weight()
    else:
        env=Env(fps=30)
        ai=Agent(env)
        ai.restore_weight()
        ai.e_greedy=0
        while True:
            ai.play()

if __name__=='__main__':
    main(isTrain=False)
