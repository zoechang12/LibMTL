import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture

class MMoE(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(MMoE, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        self.num_experts = self.kwargs['num_experts'][0]
        self.experts_shared = nn.ModuleList([encoder_class() for _ in range(self.num_experts)])
        
        # 🔥 用一个dummy输入测试encoder输出维度
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            dummy_out = self.experts_shared[0](dummy)
        self.expert_dim = dummy_out.view(1, -1).size(1)

        # 🔥 gate 的输入维度应该是 expert_dim，而不是图片大小
        self.gate_specific = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(self.expert_dim, self.num_experts),
                nn.Softmax(dim=-1)
            )
            for task in self.task_name
        })

        
    def forward(self, inputs, task_name=None):
        experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])  # [num_experts, B, dim]
        out = {}
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue
        # 用第一个专家输出的特征做 gate 输入
            selector_input = experts_shared_rep[0].view(experts_shared_rep.size(1), -1)
            selector = self.gate_specific[task](selector_input)  # [B, num_experts]
            gate_rep = torch.einsum('i b d, b i -> b d', experts_shared_rep, selector)
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            out[task] = self.decoders[task](gate_rep)
        return out


    
    def get_share_params(self):
        return self.experts_shared.parameters()

    def zero_grad_share_params(self):
        self.experts_shared.zero_grad(set_to_none=False)
