# -*- coding: UTF-8 -*-
import torch
from options import config, default_info, activation_func
import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
from torch.autograd import Variable
import os

try:
    from tqdm import tqdm  # 进度条
except:
    pass
import torch.nn.functional as F
import numpy as np


class NCFbasemodel(torch.nn.Module):
    def __init__(self, emb_dim, layer_num, act_func, dataset_type='AD', classification=False):
        super(NCFbasemodel, self).__init__()
        self.layers = eval(layer_num)

        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.item_num = int(self.item_num)  # ************************
        # print("size",self.item_num, self.emb_size,self.user_num)

        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.mlp_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mlp_i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.mlp = nn.ModuleList([])
        pre_size = 3 * self.emb_size
        for i, layer_size in enumerate(self.layers):
            self.mlp.append(nn.Linear(pre_size, layer_size, bias=False))
            pre_size = layer_size
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.prediction = nn.Linear(pre_size + self.emb_size, 1, bias=False)

        # self.u_bias = nn.Embedding(self.user_num, 1)
        # self.i_bias = nn.Embedding(self.item_num, 1)

    def forward(self, user, item, ret_emb, state='his'):
        self.check_list = []
        # u_ids = feed_dict['user_id']  # [batch_size]
        # i_ids = feed_dict['item_id']  # [batch_size, -1]

        user = user.unsqueeze(-1).repeat((1, user.shape[1]))  # [batch_size, -1]

        mf_u_vectors = self.mf_u_embeddings(user)
        mf_i_vectors = self.mf_i_embeddings(item)
        mf_ri_vectors = self.mf_i_embeddings(ret_emb)
        # cal the sim of two items
        sim = torch.cosine_similarity(mf_i_vectors, mf_ri_vectors)

        mlp_u_vectors = self.mlp_u_embeddings(user)
        mlp_i_vectors = self.mlp_i_embeddings(item)
        mlp_ri_vectors = self.mlp_i_embeddings(ret_emb)

        mf_vector_i = mf_u_vectors * mf_i_vectors
        mf_vector_ri = mf_u_vectors * mf_ri_vectors
        mf_vector = max(sim, 1-sim) * mf_vector_i + min(sim, 1-sim) * mf_vector_ri

        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors, mlp_ri_vectors], dim=-1)
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)

        # user_bias = self.u_bias(u_ids).view_as(prediction)
        # item_bias = self.i_bias(i_ids).view_as(prediction)
        # prediction = prediction + user_bias + item_bias
        return prediction.view(len(user), -1)



class Adaptor(torch.nn.Module):
    def __init__(self, input_size, layers):
        super(Adaptor, self).__init__()
        self.layer_num = layers  # config['layer_num']
        self.input_size = input_size * 13   # for AD is 13,user and item attribution num; for adressa is inputsize*2, for yelp is 16
        last_size = self.input_size
        fc_layers = []
        act_func = 'relu'  # config['activate_func']
        for i in range(self.layer_num):
            out_dim = int(last_size / 2)
            linear_model = torch.nn.Linear(last_size, out_dim)
            fc_layers.append(linear_model)
            last_size = out_dim
            fc_layers.append(activation_func(act_func))

        self.fc = torch.nn.Sequential(*fc_layers)
        self.device = torch.device('cuda' if config['use_cuda'] else "cpu")
        # finals = [torch.nn.Linear(last_size, 1)]
        # self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, his_emb):
        adapt_emb = self.fc(his_emb)
        return adapt_emb


"""
y=f(u,i,Retrieve)
"""


class MLPbasemodel(torch.nn.Module):
    def __init__(self, emb_dim, layer_num, act_func, dataset_type='AD', classification=False):
        super(MLPbasemodel, self).__init__()
        self.embedding_dim = emb_dim  # config['embedding_dim']#初始化embedding layer
        if dataset_type == "amazon":
            self.input_size = self.embedding_dim * 3   # user item retrieve
        elif dataset_type == "yelp":
            self.input_size = self.embedding_dim * 10  # <user, item, retrieve> attribution num of it
        else:
            self.input_size = self.embedding_dim * 18  # input is <user, item, retrieve> retrieve is <item>
        self.layer_num = layer_num  # config['layer_num']
        last_size = self.input_size
        fc_layers = []
        act_func = act_func  # config['activate_func']
        for i in range(self.layer_num - 1):
            out_dim = int(last_size / 2)
            linear_model = torch.nn.Linear(last_size, out_dim).double()
            fc_layers.append(linear_model)
            last_size = out_dim
            fc_layers.append(activation_func(act_func))

        self.fc = torch.nn.Sequential(*fc_layers)
        self.device = torch.device('cuda' if config['use_cuda'] else "cpu")

        # self.his_adaptor = Adaptor(self.embedding_dim * 5, 2)


        # 根据任务的输出结果，决定最后一层的结构,n_y是items数量
        if classification:  # 推荐top_k，n_y表示finals需要输出几个，一般一个用户对一个movie输出一个评分,输出对所有Y标签的概率，选择最大的一个作为模型的预测输出
            finals = [torch.nn.Linear(last_size, default_info[dataset_type]['n_y']).double(), activation_func('relu')]  # sigmoid

        else:  # y的取值只有1个
            finals = [torch.nn.Linear(last_size, 1).double()]
        self.final_layer = torch.nn.Sequential(*finals)

    # def forward(self, emb_i, emb_u):
    #     if len(emb_u)>0:
    #         x = torch.cat([emb_i, emb_u], 1)  # catenete user_embedding and item_embedding
    #     else:
    #         x=emb_i
    #     out = self.fc(x)
    #     out = self.final_layer(out)
    #
    #     # return eu, ei, out  # 输出user对item的评分
    #     return out

    def forward2(self, user, item, his_retrieve, rct_retrieve):  # emb_i, emb_u  forward2
        # his_retrieve is through adaptor
        # train sample retrieve on two reservoir
        # h_a_emb = self.his_adaptor(his_retrieve)

        # emb_merge = 0.5 * h_a_emb + 0.5 * rct_retrieve
        emb_merge = 0.5 * his_retrieve + 0.5 * rct_retrieve

        # x = torch.cat([inter_emb, emb_merge], 1)
        inter = torch.cat((user, item), -1)
        x = torch.cat((inter, emb_merge), -1)

        out = self.fc(x)
        out = self.final_layer(out)

        # return eu, ei, out  # 输出user对item的评分
        return out

    # def forward2(self, inter_emb, ret_emb, state = 'rct'):
    #     '''
    #     :param inter_emb: this is embedding of <u,i>
    #     :param ret_emb: retrieve knowledge after encoding of u, like interaction <u,i>
    #     :param state: 'rct' means recent, 'his' means historical
    #     historical samples join to update the model
    #     '''
    #     if state=='his':
    #         i_a_emb = self.his_adaptor(inter_emb)
    #         h_a_emb = self.his_adaptor(ret_emb)
    #         x = torch.cat([i_a_emb, h_a_emb], 1)
    #     else:
    #         x = torch.cat([inter_emb, ret_emb], 1)
    #
    #     out = self.fc(x)
    #     out = self.final_layer(out)
    #     # return eu, ei, out  # 输出user对item的评分
    #     return out
    def forward(self, user, item, ret_emb, state='his'):
        # each sample retrieve on each reservoir
        '''
        :param inter_emb: this is embedding of <u,i>
        :param ret_emb: retrieve knowledge after encoding of u, like interaction <u,i>
        :param state: 'rct' means recent, 'his' means historical
        historical samples join to update the model
        '''

        inter = torch.cat((user, item), -1)
        x = torch.cat((inter, ret_emb), -1)

        out = self.fc(x)
        out = self.final_layer(out)
        # out = torch.exp(-torch.pow(out, 2))  #------------------------------------------------------------
        # return eu, ei, out  # 输出user对item的评分
        return out




class MFbasemodel(nn.Module):
    def __init__(self, num_user=0, num_item=0, laten_factor=10):
        super(MFbasemodel, self).__init__()
        self.user_bais = nn.Embedding(num_user, 1)
        self.item_bais = nn.Embedding(num_item, 1)
        self.user_laten = nn.Embedding(num_user,
                                       laten_factor)  # 有num_user个用户，每个用户用laten_factor维向量表征。Embedding的输入形状N×W，N是batch size，W是序列的长度，输出的形状是N×W×embedding_dim
        self.item_laten = nn.Embedding(num_item, laten_factor)
        self.user_num = num_user
        self.item_num = num_item
        self.hidden_dim = laten_factor

    def reset_parameters(self):
        self.user_bais.reset_parameters()
        self.user_laten.reset_parameters()
        self.item_bais.reset_parameters()
        self.item_laten.reset_parameters()

        self.his_adaptor.reset_parameters()

    # def forward(self, user, item,norm=False): #return u_emb, i_emb, interaction_score
    #     user_bais = self.user_bais(user).squeeze(-1)
    #     item_bais = self.item_bais(item).squeeze(-1)
    #     userembedding = self.user_laten(user)
    #     itemembedding = self.item_laten(item)
    #     interaction = torch.mul(userembedding,itemembedding).sum(dim=-1)#基于MF的推荐，用户与项目的交互
    #     if norm:
    #         interaction = interaction / (userembedding**2).sum(dim=-1).sqrt()
    #     result = interaction
    #     return userembedding,itemembedding,result

    def forward(self, user, item, his_r, rct_r, norm=False):  # return u_emb, i_emb, interaction_score
        user_bais = self.user_bais(user).squeeze(-1)
        item_bais = self.item_bais(item).squeeze(-1)
        userembedding = self.user_laten(user)
        itemembedding = self.item_laten(item)
        interaction = torch.mul(userembedding, itemembedding).sum(dim=-1)  # 基于MF的推荐，用户与项目的交互
        if norm:
            interaction = interaction / (userembedding ** 2).sum(dim=-1).sqrt()
        result = interaction
        return userembedding, itemembedding, result

    def test(self, inputs_data, topK=20):
        user = inputs_data[:, 0]  # x[:,n]表⽰在全部数组（维）中取第n个数
        pos_negs_items = inputs_data[:, 1:]  # all item
        user_embedding = self.user_laten(user)
        user_bias = self.user_bais(user)
        pos_negs_items_embedding = self.item_laten(pos_negs_items)  # batch * 999 * item_laten
        pos_negs_bias = self.item_bais(pos_negs_items)

        user_embedding_ = user_embedding.unsqueeze(1)  # add a din to user embedding
        pos_negs_interaction = torch.mul(user_embedding_, pos_negs_items_embedding).sum(-1)  # 基于MF的实例化应用，预测交互分数
        '''sum(-1):sum the inner dimension, eg. sum each element in a row. 
        regard this score as the interacted rating between user and item
        '''
        # pos_negs_interaction = torch.chain_matmul(user_embedding,pos_negs_items_embedding)  # we can use this code to compute the interaction

        pos_negs_scores = pos_negs_interaction
        # the we have compute all score for pos and neg interactions ,each row has scorces of one pos inter and neg_num(99)  neg inter
        '''torch.topk(input:tensor,k:int,dim:int,largest:bool)
        dim=0表示按照列求topn，dim=1表示按照行求topK，None情况下，dim=1.其中largest=True表示从大到小取元素
        每行都求topK,取数组的前k个元素进行排序。
        用来获取张量或者数组中最大或者最小的元素以及索引位置
        通常该函数返回2个值：第一个值为排序的数组，得到的values是原数组dim=1的四组从大到小的三个元素值；
                            第二个值为该数组中获取到的元素在原数组中的位置标号，得到的indices是获取到的元素值在原数组dim=1中的位置。
        '''
        _, rank = torch.topk(pos_negs_scores, topK)  # 从pos_negs_scores第一维d的每个向量中从大到小获取分数最高的前k个数据,分别返回值和下标
        pos_rank_idx = (rank < 1).nonzero()  # where hit 0, o is our target item  #因为只有第一个是正样本，将除正样本以外的下标设为0
        if torch.any((rank < 1).sum(-1) > 1):  # sum(-1)表示tensor最后一维元素求和之后输出。如果>1说明正样本被推荐
            print("user embedding:", user_embedding[0])
            print("item embedding:", pos_negs_items[0])
            print("score:", pos_negs_interaction[0])
            print("rank", rank)
            print("pos_rank:", pos_rank_idx)
            np.save("pos_negs.npy", pos_negs_interaction.data.cpu().numpy())
            raise RuntimeError("compute rank error")
        have_hit_num = pos_rank_idx.shape[0]  # pos_rank_idx第一维数量，即所有击中正样本的维
        have_hit_rank = pos_rank_idx[:, 1].float()
        # DCG_ = 1.0 / torch.log2(have_hit_rank+2)
        # NDCG = DCG_ * torch.log2(torch.Tensor([2.0]).cuda())
        if have_hit_num > 0:
            batch_NDCG = 1 / torch.log2(have_hit_rank + 2)  # NDCG is equal to this format
            batch_NDCG = batch_NDCG.sum()
        else:
            batch_NDCG = 0
        Batch_hit = have_hit_num * 1.0

        return Batch_hit, batch_NDCG, pos_rank_idx[:, 0]

    def test2(self, inputs_data, topK=20):
        user = inputs_data[:, 0]
        pos_negs_items = inputs_data[:, 1:]  # all item
        user_embedding = self.user_laten(user)
        user_bias = self.user_bais(user)
        pos_negs_items_embedding = self.item_laten(pos_negs_items)  # batch * 999 * item_laten
        pos_negs_bias = self.item_bais(pos_negs_items)

        user_embedding = user_embedding.unsqueeze(1)  # add a din to user embedding
        pos_negs_interaction = torch.mul(user_embedding, pos_negs_items_embedding).sum(
            -1)  # MF交互分数。通过分解之后的两矩阵内积，来填补缺失的数据，用来做预测评分
        # pos_negs_interaction = torch.chain_matmul(user_embedding,pos_negs_items_embedding)  # we can use this code to compute the interaction

        pos_negs_scores = pos_negs_interaction
        # the we have compute all score for pos and neg interactions ,each row has scorces of one pos inter and neg_num(99)  neg inter
        _, rank = torch.topk(pos_negs_scores, topK)
        pos_rank_idx = (rank < 1).nonzero()  # where hit 0, o is our target item
        have_hit_num = pos_rank_idx.shape[0]  # 选择推荐的正样本数量
        have_hit_rank = pos_rank_idx[:, 1].float()
        # DCG_ = 1.0 / torch.log2(have_hit_rank+2)
        # NDCG = DCG_ * torch.log2(torch.Tensor([2.0]).cuda())
        if have_hit_num > 0:
            batch_NDCG = 1 / torch.log2(have_hit_rank + 2)  # NDCG is equal to this format
            batch_NDCG = batch_NDCG.sum()
        else:
            batch_NDCG = 0
        Batch_hit = have_hit_num * 1.0
        return pos_rank_idx[:, 0], rank, Batch_hit, batch_NDCG

    def set_parameters(self, user_weight, item_weight):
        self.user_laten.weight.data.copy_(user_weight[:, 0:-1])
        self.user_bais.weight.data.copy_(user_weight[:, -1].unsqueeze(-1))
        self.item_laten.weight.data.copy_(item_weight[:, 0:-1])
        self.item_bais.weight.data.copy_(item_weight[:, -1].unsqueeze(-1))


class MF2(nn.Module):
    def __init__(self, num_user=0, num_item=0, laten_factor=10):
        super(MF2, self).__init__()
        self.user_bais = nn.Embedding(num_user, 1)
        self.item_bais = nn.Embedding(num_item, 1)
        self.user_laten = nn.Embedding(num_user, laten_factor)
        self.item_laten = nn.Embedding(num_item, laten_factor)
        self.user_num = num_user
        self.item_num = num_item
        self.hidden_dim = laten_factor

    def forward(self, user, item, neg_item=None):
        if neg_item is not None:  # used for train
            user_bais = self.user_bais(user).squeeze(-1)
            item_bais = self.item_bais(item).squeeze(-1)
            neg_item_bais = self.item_bais(neg_item).squeeze(-1)

            userembedding = self.user_laten(user)
            itemembedding = self.item_laten(item)
            neg_itemembedding = self.item_laten(neg_item)

            interaction = torch.mul(userembedding, itemembedding).sum(dim=-1)
            neg_interaction = torch.mul(userembedding, neg_itemembedding).sum(dim=-1)
            result_pos = user_bais + item_bais + interaction
            result_neg = user_bais + neg_item_bais + neg_interaction
            score = result_pos - result_neg
            bpr_loss = -torch.sum(F.logsigmoid(score))

            l2loss = torch.norm(userembedding, dim=-1).sum() + torch.norm(itemembedding, dim=-1).sum() + torch.norm(
                neg_itemembedding).sum()
            return bpr_loss, l2loss

        else:  # used for test
            user_bais = self.user_bais(user).squeeze(-1)
            item_bais = self.item_bais(item).squeeze(-1)
            userembedding = self.user_laten(user)
            itemembedding = self.item_laten(item)
            interaction = torch.mul(userembedding, itemembedding).sum(dim=-1)
            result = user_bais + item_bais + interaction
            return userembedding, itemembedding, result
