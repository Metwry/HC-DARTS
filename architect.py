"无修改"
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  "用于将一个由张量组成的列表 xs 连接成一个单一的张量"
  return torch.cat([x.view(-1) for x in xs])



class Architect(object):
  "用于构建神经网络架构的优化器和相关参数"

  def __init__(self, model, args):  #模型，一些参数
    self.network_momentum = args.momentum  #网络的动量
    self.network_weight_decay = args.weight_decay #权重衰减
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)  #基于Adam优化器，优化模型架构参数

  def _compute_unrolled_model(self, input, target, eta, network_optimizer): #输入数据，目标标签，步长，网络优化器
    "计算一个未展开的模型"
    loss = self.model._loss(input, target)   #计算模型的loss值
    theta = _concat(self.model.parameters()).data   #将其展平为一维张量
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    #计算损失函数关于模型参数的梯度，加上网络权重衰减的影响
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):  #训练集、验证集的输入数据，标签；步长，优化器，布尔值
    "用于执行一步优化过程"
    self.optimizer.zero_grad() #先将模型参数的梯度清零
    if unrolled: #为1
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else: #为0
        self._backward_step(input_valid, target_valid)
    self.optimizer.step() #参数更新

  def _backward_step(self, input_valid, target_valid):
    "执行模型在验证集的反向传播"
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    "对未展开模型进行反向传播"
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)   #获得一个未展开模型
    unrolled_loss = unrolled_model._loss(input_valid, target_valid) #计算在验证集的损失

    unrolled_loss.backward()  #计算梯度
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]   #未展开模型的架构参数的梯度列表
    vector = [v.grad.data for v in unrolled_model.parameters()]  #未展开模型的所有参数的梯度向量。
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
    #保存 Hessian 矩阵与梯度向量的乘积结果，这个乘积结果提供了关于损失函数在当前参数位置的二阶导数信息。

    for g, ig in zip(dalpha, implicit_grads):   #zip():将两个列表队友位置的元素配对为元组
      g.data.sub_(eta, ig.data)    #g.data -= eta * ig.data  牛顿法参数更新

    for v, g in zip(self.model.arch_parameters(), dalpha):   #前者返回模型架构参数，后者是架构参数的梯度列表
      if v.grad is None:  #若为None  使用初始化梯度
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)  #替换为g.data

  def _construct_model_from_theta(self, theta):
    "根据给定的参数向量构建一个新的模型，并加载该模型参数"
    model_new = self.model.new() #创建一个与原模型具有相同架构的新模型
    model_dict = self.model.state_dict() #获取原模型的状态字典，其中包含了原模型的所有参数

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())  #计算张量的总元素个数H*W*C
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    "实现了计算 Hessian 矩阵和一个向量的乘积的功能"
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)] #返回 Hessian 矩阵和向量的乘积的结果列表。

