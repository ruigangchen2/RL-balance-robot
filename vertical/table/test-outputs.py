import torch

device = torch.device("cpu")
nnn = 512
policy1 = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.Tanh(),
                              torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                              torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))
policy1.load_state_dict(torch.load('C:/Users/Administrator/Desktop/Cases/RL-balance-robot/horizontal/training/outputs/5degrees-PPO-zero-torque.pth'))
policy1.to('cpu')
prob = policy1(torch.FloatTensor([[80,0,-3000]]))
print(prob)
