from torch import nn
import torch
from options import activation_func


def cosine_similarity(input1, input2):
    # query_norm = torch.sqrt(torch.sum(input1 ** 2 + 0.00001, 1))
    # doc_norm = torch.sqrt(torch.sum(input2 ** 2 + 0.00001, 1))
    #
    # prod = torch.sum(torch.mul(input1, input2), 1)
    # norm_prod = torch.mul(query_norm, doc_norm)
    #
    # cos_sim_raw = torch.div(prod, norm_prod)
    simil = torch.tensor([])
    for name, param in input1.items():
        if param.ndim < 2:
            param = torch.unsqueeze(param, dim=0)
            input2[name] = torch.unsqueeze(input2[name], dim=0)
        query_norm = torch.sqrt(torch.sum(param**2+0.00001, 1))
        try:
            doc_norm = torch.sqrt(torch.sum(input2[name]**2+0.00001, 1))
        except:
            doc_norm = torch.sqrt(torch.sum(torch.unsqueeze(input2[name], dim=0) ** 2 + 0.00001, 1))
        prod = torch.sum(torch.mul(param, input2[name]), 1)
        norm_prod = torch.mul(query_norm, doc_norm)

        cos_sim_raw = torch.div(prod, norm_prod)
        cos_sim_raw = torch.unsqueeze(torch.mean(cos_sim_raw), dim=0)
        simil = torch.cat((simil, cos_sim_raw), dim=0)
    return simil


class Attention(torch.nn.Module):
    def __init__(self, n_k, activation='relu'):
        super(Attention, self).__init__()
        self.n_k = n_k
        self.fc_layer = torch.nn.Linear(self.n_k, self.n_k, activation_func(activation))
        self.soft_max_layer = torch.nn.Softmax()

    def forward(self, pu, mp):
        expanded_pu = pu.repeat(1, len(mp)).view(len(mp), -1)  # shape, n_k, pu_dim
        inputs = cosine_similarity(expanded_pu, mp)
        fc_layers = self.fc_layer(inputs)
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values
'''
先update reg, 然后根据两个reg计算reg_loss返回，只要频繁更新的adaptor+base_recommender'''
class Regularization(nn.Module):
    def __init__(self, weight_decay, max_param, activation='relu', p=2):  # model,
        super(Regularization, self).__init__()
        # self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = None      # self.get_weight(model)
        self.cuda = False
        # self.weight_info(self.weight_list)

        self.max_param = max_param  # 所有参数的总长
        self.fc_layer = torch.nn.Linear(self.max_param, self.max_param, activation_func(activation)).double()
        self.soft_max_layer = torch.nn.Softmax()
        self.his_grad_update_lr = 1e-4


    # 在训练过程中每次调用需要
    # 每个batch，在得到这个batch的gradients之后，先更新regualrizer之后再参与模型的更新
    def forward(self, his_grad, his_model, new_grad, new_model):
        # self.weight_list = self.get_weight(new_model)
        self.weight_list = new_model
        reg_model = self.update_reg(his_model, new_model, type="model")
        reg_grad = self.update_reg(his_grad, new_grad, type="grad")

        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, reg_model, reg_grad, p=self.p)
        return reg_loss


    def extend_value(self, weight):
        # 将所有参数拼接成一维
        # one_dim_weight = torch.tensor([])
        # for _, param in weight.items():
        #     if param.ndim > 1:
        #         param = param.reshape(-1)
        #     one_dim_weight = torch.cat((one_dim_weight, param))
        for name, param in weight.items():
            if param.size(-1) < self.max_param:
                param = torch.nn.functional.pad(param, (0, self.max_param - param.size(-1)))
                weight[name] = param
        return weight

    def get_merge_lr(self, his, new):
        # 应该要填充成最长的参数长度
        # expanded_pu = pu.repeat(1, len(mp)).view(len(mp), -1)  # shape, n_k, pu_dim
        # extend the dict to one dimension tensor  is too large
        his = self.extend_value(his)
        new = self.extend_value(new)

        inputs = cosine_similarity(his, new)
        fc_layers = self.fc_layer(inputs)
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values

    # 确定设备
    def to(self, device):
        self.device = device
        super().to(device)
        return self

    # 得到正则化参数
    def get_weight(self, model):
        weight_list = {}
        # for name, param in model.named_parameters():
        #     if 'weight' in name:
        #         weight = (name, param)
        #         weight_list.append(weight)
        # return weight_list
        for name, param in model.rec_model.named_parameters():  # input loading don't need train
            # if 'item_embedding' in name:  # the dimension of item_embedding is large
            #     continue
            # param.requires_grad = True
            # weight = (name, param)
            if self.cuda:
                weight_list[name] = param.cuda()
            else:
                weight_list[name] = param

        for name, param in model.adaptor.named_parameters():  # input loading don't need train
            # param.requires_grad = True
            # weight = (name, param)
            if self.cuda:
                weight_list[name] = param.cuda()
            else:
                weight_list[name] = param

        return weight_list

    # 求取正则化的和
    def regularization_loss2(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list.items():
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss

    # 打印有哪些参数
    def weight_info(self, weight_list):
        print("---------------regularization weight---------------")
        for name, w in weight_list.items():
            print(name)
        print("---------------------------------------------------")

    def get_param_after_grad(self, his_gradients, weight_list):
        # obtain the model parameters updated by his_gradients
        weight_after_grad = {}
        for name, w in weight_list.items():
            weight_after_grad[name] = w - self.his_grad_update_lr * his_gradients[name]
        return weight_after_grad

    def regularization_loss(self, weight_list, weight_decay, his_model, his_grad, p=2):
        # calculate the regularization loss based on his gradients and model of L2
        new_m_updated_hg = self.get_param_after_grad(his_grad, weight_list)  # obtain the model parameters updated by his_grad,
        reg_loss = 0
        for name, w in weight_list.items():
            # l2_reg = torch.norm(w, p=p)
            # reg_loss = reg_loss + l2_reg
            l2_grad_reg = torch.norm(new_m_updated_hg[name], p=p)
            l2_model_reg = torch.norm(his_model[name], p=p)
            reg_loss = reg_loss + l2_grad_reg + l2_model_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss

    def update_reg(self, his, new, type='grad'):
        # 一个可学习的速率，用于更新regularizer中的his model and his gradients
        # merge the his model and current model self.model, his gradients and current gradients
        # --------------update the two gradients---------------------- 应该先更新模型再更新regularizer
        if type=="grad":
            pass
        else:
            pass
        lr = self.get_merge_lr(his, new)  # merge the his and new reg
        i = 0
        for name, param in his.items():
            his[name] = param + lr[i] * new[name]
            i += 1
        return his


# learnable merge rate
class Merge(nn.Module):
    def __init__(self, args):
        super(Merge, self).__init__()
        # -------------设计一下模型融合机制
        # 不然就简单的mlp

    def forward(self, his, new):
        # output: merge rate
        return

    def forward(self, d1, d2):
        assert set(d1.keys()) == set(d2.keys()), "d1 and d2 don't have the same keys"

        product_sum = 0.0
        for name in d1.keys():
            product_sum += d1[name] * d2[name]

        norm_1 = sum([val ** 2 for val in d1.values()]) ** 0.5
        norm_2 = sum([val ** 2 for val in d2.values()]) ** 0.5

        w = product_sum / (norm_1 * norm_2)

        return w


    def update_reg(self, his, new):
        # 一个可学习的速率，用于更新regularizer中的his model and his gradients
        lr = self.forward(his, new)
        return his + lr * new



# reg_loss = Regularization(model, weight_decay, p=2).to(device)
#
# loss = criterion(y_pred, labels.long())
# loss = loss + reg_loss(model)  # 加入正则的损失
