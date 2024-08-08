import torch

device = torch.device("cpu")
nnn = 512
policy1 = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.Tanh(),
                              torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                              torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))
policy1.load_state_dict(torch.load('C:/Users/Administrator/Desktop/Cases/RL-balance-robot/horizontal/training/outputs/5degrees_Policy_Net_Pytorch(-1,0,1)_3.pth'))
policy1.to('cpu')
prob = policy1(torch.FloatTensor([[-1.5, 3, 2]]))
print(prob)
