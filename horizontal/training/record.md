## The record of the PPO training

For the following traning situations,  
`policy_entropy_coefficient = 0.005`  
`self.reward -= (abs(self.steps) * 0.003)`  

The network gets 179 times success for 1000 training times when it has achieved 19 episodes.  
The network gets 336 times success for 1000 training times when it has achieved 29 episodes.   
The network gets 400 times success for 1000 training times when it has achieved 70 episodes.   
The network gets 453 times success for 1000 training times when it has achieved 80 episodes.


For the following traning situations,  
`policy_entropy_coefficient = 0.001`  
`self.reward -= (abs(self.steps) * 0.005)`  
