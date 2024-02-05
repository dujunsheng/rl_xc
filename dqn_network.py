from torch import nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

    

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, input, context):
        # 对输入进行线性变换
        input = self.linear_in(input)
        context = self.linear_in(context)
        
        # 计算注意力权重
        attn_weights = torch.matmul(input, context.transpose(0, 1))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # 根据注意力权重计算加权上下文向量
        weighted_context = torch.matmul(attn_weights, context)
        
        # 将加权上下文向量和输入拼接，并进行线性变换
        output = torch.cat((input, weighted_context), dim=-1)
        output = self.linear_out(output)
        
        return output
