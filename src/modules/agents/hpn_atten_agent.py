import math
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from modules.layer.self_atten import SelfAttention
from torch.nn.parameter import Parameter

def kaiming_uniform_(tensor_w, tensor_b, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor_w.data, mode)
    std = gain / math.sqrt(fan)
    bound_w = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    bound_b = 1 / math.sqrt(fan)
    with th.no_grad():
        tensor_w.data.uniform_(-bound_w, bound_w)
        if tensor_b is not None:
            tensor_b.data.uniform_(-bound_b, bound_b)

class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)

class ATTRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ATTRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.n_heads = args.hpn_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.use_q_v = False
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.hyper_ally = nn.Sequential(
        #             nn.Linear(args.rnn_hidden_dim, args.hpn_hyper_dim),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(args.hpn_hyper_dim, args.rnn_hidden_dim * args.rnn_hidden_dim * self.n_heads)
        # )
        # self.unify_input_heads = Merger(self.n_heads, self.rnn_hidden_dim)

        # self.att = SelfAttention(args.rnn_hidden_dim, args.att_heads, args.att_embed_dim)
        # self.norm1 = nn.LayerNorm(args.rnn_hidden_dim)
        # self.norm2 = nn.LayerNorm(args.rnn_hidden_dim)

        self.selfpadding = "Zero"

        self.fc2 = nn.Linear(args.att_heads *  args.att_embed_dim, args.rnn_hidden_dim)
        self.fc_inter = nn.Sequential(nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim),
                                nn.ReLU(inplace=True))
        self.fc_last = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(args.rnn_hidden_dim,args.n_actions))

    def _reset_hypernet_parameters(self, init_type='kaiming'):
        gain = 2 ** (-0.5)
        # %%%%%%%%%%%%%%%%%%%%%% Hypernet-based API input layer %%%%%%%%%%%%%%%%%%%%
        for m in self.hyper_enemy.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)
        for m in self.hyper_ally.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)
            
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        device = self.fc1.weight.device
        inputs = inputs.to(device)
        hidden_state = hidden_state.to(device)

        # INPUT
        e = inputs.shape[-1]
        inputs = inputs.reshape(-1, self.args.n_agents,e)
        b, a, e = inputs.size()

        # RNN
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
    
        # ATT
        # inputs = inputs.view(-1,e)
        # hyper_ally_out = self.hyper_ally(h)
        # fc1_w_ally = hyper_ally_out.view(-1,self.rnn_hidden_dim,self.rnn_hidden_dim * self.n_heads)
        # embedding_allies = th.matmul(h.unsqueeze(1), fc1_w_ally).view(
        #     b, a, self.n_heads, self.rnn_hidden_dim
        # )
        # embed =  self.unify_input_heads(embedding_allies).view(b,a,-1)


        # att = self.att(embed)
        att = self.att(inputs.view(b, a, -1))
        att = F.relu(self.fc2(att), inplace=True).view(b, a, -1)
        # att_v = F.relu(self.fc2(self.att.values), inplace=True).view(-1, self.args.rnn_hidden_dim)
        
        # Q
        # print(att.shape,embed.shape)
        # q = th.cat((embed, att), dim=-1)
        q = th.cat((h, att), dim=-1)
        # q_v = th.cat((h, att_v), dim=-1)
        if self.selfpadding == "Zero":
            allay_mask = th.zeros_like(att) 
        else:
            allay_mask = th.randn_like(att)

        q_self = th.cat((h,allay_mask),dim=-1)
            

        inter = self.fc_inter(q)
        q = self.fc_last(inter)

        with th.no_grad():
            q_self = self.fc_last(self.fc_inter(q_self))
        

        # inter_v = self.fc_inter(q_v)
        # q_v = self.fc_last(inter_v).view(b,a,-1)
        # self.q_v = q_v
        
        # if self.use_q_v:
        #     return q_v.view(b, a, -1), inter.view(b, a, -1), h.view(b, a, -1)
        # else:
        return q.view(b, a, -1), inter.view(b,a,-1), h.view(b, a, -1),q_self.view(b,a,-1)
