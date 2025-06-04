import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
import logging
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype



# ++++
class ChannelAttention(nn.Module):
    "通道注意力，用于增强模型对通道特征的关注，创新点"

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        # MLP  用于学习通道之间的关系   通道数减少一半
        self.fc1 = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        "计算池化后的F(x)=F_ap + F_mp = Fc"
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 平均池化
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 最大池化

        out = avg_out + max_out  # 值相加

        return self.sigmoid(out)  # 返回非线性激活的out


class SpatialAttention(nn.Module):
    def __init__(self,  kernel_size=7 ):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ++++

class MixedOp(nn.Module):
    "定义了混合操作，创新点"

    def __init__(self, C, stride):  # 通道数和步长
        super(MixedOp, self).__init__()
        "super会执行MixedOp这个类的父类的init()方法"
        self._ops = nn.ModuleList()  # 用于存储不同的操作
        self.mp = nn.MaxPool2d(2, 2)  # 创建了一个最大池化操作 ++++

        self.k = 1  # 用于控制通道注意力机制中的通道分组数量 ++++
        self.ca = ChannelAttention(C)  # 创建通道注意力类  +++++
        self.sa = SpatialAttention()   #创建空间注意力类
        for primitive in PRIMITIVES:  # 遍历基本操作列表

            op = OPS[primitive](C // self.k, stride, False)  # 加入操作，并传入参数：通道数，步长，affine的布尔值
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))  # 若该操作是池化操作，将其包装在一个序列中并添加归一化操作
            self._ops.append(op)  # 添加操作进入操作列表

    def forward(self, x, weights):
        "混合操作的前向传播方法"
        dim_2 = x.shape[1]

        #
        Spatial = self.sa(x)
        x = x * Spatial
        num_list = self.ca(x)  # 经过通道注意力
        x = x * num_list  # x乘以通道注意力
        num_list = num_list


        """

        """

        num_dict = []  # 存放max_num_index第一个维度的
        slist = torch.sum(num_list, dim=0, keepdim=True)  # 对通道数进行求和
        values, max_num_index = slist.topk((dim_2 // self.k)*3//4, dim=1, largest=True, sorted=True)
        # values, max_num_index = slist.topk((dim_2 // self.k) , dim=1, largest=True, sorted=True)
        other_values, other_num_index = slist.topk(k=x.size(1) - ((dim_2 // self.k)*3//4), dim=1, largest=False, sorted=True)



        # 对通道注意力权重排序，获得1/K个最大通道注意力权重Fc索引

        #竞争选择
        other_num_index0 = []

        while len(other_num_index0) < (dim_2 // self.k)*1//4:
            random_index1 = torch.randint(0, len(other_num_index[0]), size=(1,))
            random_index2 = torch.randint(0, len(other_num_index[0]), size=(1,))
            # print("random", random_index1)
            while random_index1 == random_index2:
                random_index1 = torch.randint(0, len(other_num_index[0]), size=(1,))
                random_index2 = torch.randint(0, len(other_num_index[0]), size=(1,))

            choose1 = other_values[0][random_index1]
            choose2 = other_values[0][random_index2]
            # fc_gap = abs(choose1 - choose2)
            # fs_gap_1 = abs(choose1 - Spatial_att)
            # fs_gap_2 = abs(choose2 - Spatial_att)
            # final_gap_1 = fc_gap + fs_gap_1
            # final_gap_2 = fc_gap + fs_gap_2
            if  choose1 > choose2:   #选择的前者通道注意力值更大
                if random_index1 not in other_num_index0:
                    other_num_index0.append(random_index1)

            elif choose1 < choose2:
                finchoose = choose2
                if random_index2 not in other_num_index0:
                        other_num_index0.append(random_index2)
        other_num_index0_cuda  = [tensor.cuda() for tensor in other_num_index0]


        for i in range(0, len(max_num_index[0])):
            num_dict.append(max_num_index[0][i])  # 做了一个注意力前1/k的复制

        for i in range(0, len(other_num_index0_cuda)):
            num_dict.append(other_num_index0_cuda[i])    # 加入了竞争后的的复制


        xtemp = torch.index_select(x, 1, torch.tensor(num_dict).cuda())
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        # 这两行代码的作用是根据给定的权重对一组操作进行加权求和，并得到一个混合后的结果。
        if temp1.shape[2] == x.shape[2]:
            x[:, num_dict, :, :] = temp1[:, :, :, :]
        else:
            x = self.mp(x)
            x[:, num_dict, :, :] = temp1[:, :, :, :]
        # 确保混合操作后得到的张量能够与输入张量 x 的尺寸保持一致，以便进行后续的计算和处理。
        return x


class Cell(nn.Module):
    "细胞类"

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
            # 若前一个细胞是reduction cell那么进行降维操作
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            # 否则进行ReLU 激活函数和批量归一化
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        # 进行ReLU 激活函数和批量归一化
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()  # 用于存储操作
        self._bns = nn.ModuleList()  # 用于存储细胞中的批归一化层
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)

            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        "创新点"
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)

                n = 3
                start = 2
                weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)

                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)

                n = 3
                start = 2
                weights2 = F.softmax(self.betas_normal[0:2], dim=-1)

                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.betas_normal = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self.betas_reduce = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        "创新点"

        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0

            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # ++++
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                # ++++
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        # +++++
        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps - 1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
            # ++++
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        print('self.alphas_normal', F.softmax(self.alphas_normal, dim=-1))
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())
        print('self.alphas_reduce', F.softmax(self.alphas_reduce, dim= -1))
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )

        return genotype

