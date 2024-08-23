import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
torch.cuda.set_device(device)

nnn = 512
# 实例化策略网络
policy = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.Tanh(),  # 双曲正切激活函数
                             torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                             torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))  # 计算每个动作的概率
policy.to(device)
policy.load_state_dict(
    torch.load('C:/Users/Administrator/Desktop/Cases/RL-balance-robot/vertical/training/outputs/PPO_vertical_4.pth'))
policy.eval()
dummy_input = torch.randn(1, 3, device='cuda').reshape(1, 3)
torch.onnx.export(policy, dummy_input, "PPO.onnx", opset_version=11, verbose=True)
