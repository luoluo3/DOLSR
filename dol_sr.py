print("---load maup---")
import copy
import math
# from datetime import time
import time
import numpy as np
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
import os
import torch
from torch import nn
import sys
import gc

sys.path.append('..')
from data_process.embedding import *

# from tqdm import tqdm
print("---load maup---1")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,1,3"
from Reservoir import Transformer, Reservoir
from options import *  # config, default_info, activation_func,
# from Encoder import His_encoder, Rct_encoder
from Model.Encoder import His_encoder, Rct_encoder
from data_process.input_loading import *
from evalution.evaluation2 import test_model
from Model.Regularizer import Regularization


def evaluate_method(predictions: np.ndarray, topk: list, metrics: list):
    """
    :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
    :param topk: top-K values list
    :param metrics: metrics string list
    :return: a result dict, the keys are metrics@topk
    """
    evaluations = dict()
    predictions = predictions.detach().cpu().numpy()
    sort_idx = (predictions).argsort(axis=1)
    gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
    for k in topk:
        hit = (gt_rank <= k)
        for metric in metrics:
            key = '{}@{}'.format(metric, k)
            if metric == 'HR':
                evaluations[key] = hit.mean(dtype=np.float16)
            elif metric == 'NDCG':
                evaluations[key] = (hit / np.log2(gt_rank + 1)).mean(dtype=np.float16)
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    return evaluations


# import torchviz
# import graphviz
# from torchviz import make_dot

class MLPbasemodel(torch.nn.Module):
    def __init__(self, emb_dim, layer_num, act_func, dataset_type='AD', classification=False):
        super(MLPbasemodel, self).__init__()
        self.embedding_dim = emb_dim  # config['embedding_dim']#初始化embedding layer
        if dataset_type == "adressa":
            self.input_size = self.embedding_dim * 3  # user item neg_item
        elif dataset_type == "yelp":
            self.input_size = self.embedding_dim * 10  # <user, item, retrieve> attribution num of it
        else:
            self.input_size = self.embedding_dim * 14  # 18  # input is <user, item, retrieve> retrieve is <item>
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

        if classification:
            finals = [torch.nn.Linear(last_size, default_info[dataset_type]['n_y']).double(),
                      activation_func('relu')]  # sigmoid

        else:
            finals = [torch.nn.Linear(last_size, 1).double()]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward2(self, user, item, his_retrieve, rct_retrieve):  # emb_i, emb_u  forward2
        # his_retrieve is through adaptor
        # train sample retrieve on two reservoir

        emb_merge = 0.5 * his_retrieve + 0.5 * rct_retrieve

        # x = torch.cat([inter_emb, emb_merge], 1)
        inter = torch.cat((user, item), -1)
        x = torch.cat((inter, emb_merge), -1)

        out = self.fc(x)
        out = self.final_layer(out)
        out = torch.exp(-torch.pow(out, 2))

        # return eu, ei, out
        return out

    def forward(self, user, item, merge_retrieval):  # emb_i, emb_u  forward2
        # his_retrieve is through adaptor
        # train sample retrieve on two reservoir

        emb_merge = merge_retrieval

        # x = torch.cat([inter_emb, emb_merge], 1)
        inter = torch.cat((user, item), -1)
        x = torch.cat((inter, emb_merge), -1)

        out = self.fc(x)
        out = self.final_layer(out)
        out = torch.exp(-torch.pow(out, 2))

        return out

    def forward1(self, user, item, ret_emb, state='his'):
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
        out = torch.exp(-out)  # ------------------------------------------------------------
        # return eu, ei, out
        return out


class Adaptor(torch.nn.Module):
    def __init__(self, input_size, layers):
        super(Adaptor, self).__init__()
        self.layer_num = layers  # config['layer_num']
        self.input_size = input_size
        last_size = self.input_size
        fc_layers = []
        act_func = 'relu'  # config['activate_func']
        for i in range(self.layer_num):
            # out_dim = int(last_size / 2)
            linear_model = torch.nn.Linear(last_size, last_size).double()
            fc_layers.append(linear_model)
            # last_size = out_dim
            fc_layers.append(activation_func(act_func))

        self.fc = torch.nn.Sequential(*fc_layers)
        self.device = torch.device('cuda' if config['use_cuda'] else "cpu")
        # finals = [torch.nn.Linear(last_size, 1)]
        # self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, his_emb):
        adapt_emb = self.fc(his_emb)
        return adapt_emb


"""
add or concate
this is transformer for fusing two retrieve_emb
"""


class RetrieveFuse(object):
    def __init__(self, emb_dim):
        self.fuse_layer = Transformer.MultiheadAttention(emb_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def fuse(self, his_emb, rct_emb):
        # two methods to fuse the retrieve results
        out = self.fuse_layer(torch.cat([his_emb, rct_emb], 1))

        out = his_emb + rct_emb

        return out


def ndcg(ground_truth, test_result, topk=[5]):
    # pred_y = torch.argmax(test_result, dim=1)
    pred_y = test_result.view(-1)
    sort_real_y, sort_real_y_index = ground_truth.clone().detach().sort(descending=True)
    sort_pred_y, sort_pred_y_index = pred_y.clone().detach().sort(descending=True)
    ndcg = []
    hr = []
    for top_k in topk:
        pred_sort_y = ground_truth[sort_pred_y_index][:top_k]
        top_pred_y, _ = pred_sort_y.sort(descending=True)

        ideal_dcg = 0
        n = 1
        for value in sort_real_y[:top_k]:
            i_dcg = (2 ** float(value + 1) - 1) / math.log2(n + 1)
            ideal_dcg += i_dcg
            n += 1

        pred_dcg = 0
        n = 1
        for value in top_pred_y:
            p_dcg = (2 ** float(value + 1) - 1) / math.log2(n + 1)
            pred_dcg += p_dcg
            n += 1

        ndcg.append(pred_dcg / ideal_dcg)

        hits = 0
        for i in sort_pred_y_index[:top_k]:
            if i in sort_real_y_index[:top_k]:
                hits += 1
        hr.append(hits / top_k)
    return ndcg, hr


class AdaptorRecommender(torch.nn.Module):
    def __init__(self, args, rec_module, u_emb_module, i_emb_module, his_encoder,
                 rct_encoder):  # , his_transformer, rct_transformer, his_reservoir, rct_reservoir):  # Adaptor(self.embedding_dim, 2)
        super(AdaptorRecommender, self).__init__()

        self.emb_dim = args.embedding_dim
        self.user_embedding = u_emb_module
        self.item_embedding = i_emb_module
        self.rec_model = rec_module  # BaseModel.MLPbasemodel(self.emb_dim, self.rec_layers, self.act_func)

        # as embedding layer
        self.his_encoder = his_encoder
        self.rct_encoder = rct_encoder
        self.retrieve_aggregator = torch.nn.Linear(args.i_attr * self.emb_dim * 2, args.i_attr * self.emb_dim).double()
        # self.user_aggregator = torch.nn.Linear(self.emb_dim * 2, self.emb_dim).double()
        self.item_aggregator = torch.nn.Linear(args.i_attr * self.emb_dim * 2, self.emb_dim).double()

        # self.retrieval_merge_layer = nn.Linear(args.i_attr * self.emb_dim * 2, args.i_attr * self.emb_dim).double()

        # *****************
        # self.his_reservoir = his_reservoir
        # self.rct_reservoir = rct_reservoir
        # self.his_transformer = his_transformer
        # self.rct_transformer = rct_transformer
        # ********************

        self.loss_func = torch.nn.BCELoss(reduction='none')

        # self.u_agg_layer = torch.nn.Linear(2, 1)
        # self.u_i_agg_layer = torch.nn.Linear(2, 1)

    def forward1(self, user, item, retrieve, state='his'):
        ei = self.item_embedding(item)
        eu = self.user_embedding(user)
        if state == 'his':
            # if len(eu) > 0:
            #     inter = torch.cat([eu, ei], 1)  # catenete user_embedding and item_embedding
            # else:
            #     inter = ei
            # inter = torch.cat([eu, ei], 1)

            # inter = self.his_encoder(inter)
            # retrieve = self.his_encoder(retrieve)
            ei = self.his_encoder(ei)
            retrieve = self.his_encoder(retrieve)

            # i_a_emb = self.his_adaptor(inter)
            # h_a_emb = self.his_adaptor(retrieve)
            rec_value = self.rec_model(eu, ei, retrieve)
        else:

            ei = self.rct_encoder(ei)
            retrieve = self.rct_encoder(retrieve)
            rec_value = self.rec_model(eu, ei, retrieve)
        return rec_value

    def forward(self, user, item, his_retrieve, rct_retrieve):
        """  forward 2
        train sample retrieve from two reservoir than to merge
        :param user, item: train sample
        :param his_retrieve: retrieve knowledge from historical reservoir
        :param rct_retrieve: retrieve knowledge from recent reservoir
        """
        ei = self.item_embedding(item)
        eu = self.user_embedding(user)
        eu = torch.squeeze(eu, dim=0)
        # his_retrieve = self.item_embedding(his_retrieve)
        # rct_retrieve = self.item_embedding(rct_retrieve)
        his_r_emb = self.his_encoder(his_retrieve)  # double
        rct_r_emb = self.rct_encoder(rct_retrieve)
        # inter = self.rct_encoder(inter)

        # ei = self.rct_encoder(ei.double())

        # his_r_emb = self.adaptor(his_r_emb)

        eir = self.rct_encoder(ei.double())
        eih = self.his_encoder(ei.double())

        # print("-his_r_emb--",his_r_emb.size(),rct_r_emb.size())
        # print("-item input--",eir.size(),eih.size())

        enhanced_ret_emb = self.retrieve_aggregator(torch.cat((his_r_emb, rct_r_emb), dim=-1))

        enhanced_item_emb = self.item_aggregator(torch.cat((eih, eir), dim=-1))

        # merged_retrieve = self.retrieval_merge_layer(torch.cat((his_r_emb, rct_r_emb), dim=-1))
        # print("--eu device--",eu.device, ei.device,merged_retrieve.device)
        # print("-merged_retrieve--",merged_retrieve.size())
        # print("---",eu.size(),enhanced_item_emb.size(),enhanced_ret_emb.size())
        pos_pred = self.rec_model(eu.double(), enhanced_item_emb, enhanced_ret_emb)

        # pos_pred = self.rec_model(eu.double(), ei, his_r_emb, rct_r_emb)

        bpr_loss = torch.nn.functional.mse_loss(pos_pred, torch.ones(pos_pred.shape).cuda().double())
        return bpr_loss

    def forward_neg(self, user, item, neg_item, his_retrieve, rct_retrieve):
        """  forward 2
        train sample retrieve from two reservoir than to merge
        :param user, item: train sample
        :param his_retrieve: retrieve knowledge from historical reservoir
        :param rct_retrieve: retrieve knowledge from recent reservoir
        """
        ei = self.item_embedding(item)
        eu = self.user_embedding(user)
        eni = self.item_embedding(neg_item)

        # *********************
        # his_retrieve = self.his_reservoir.retrieve(eu, 100)
        # rct_retrieve = self.rct_reservoir.retrieve(eu, 100)
        # his_retrieve = torch.stack(his_retrieve, dim=0)
        # rct_retrieve = torch.stack(rct_retrieve, dim=0)
        # his_retrieve = torch.unsqueeze(his_retrieve, dim=1)
        # rct_retrieve = torch.unsqueeze(rct_retrieve, dim=1)

        # his_retrieve = self.his_transformer(his_retrieve, his_retrieve, his_retrieve, his_retrieve)  # .double()
        # rct_retrieve = self.rct_transformer(rct_retrieve, rct_retrieve, rct_retrieve, rct_retrieve)
        # *********************

        # his_r_emb = self.his_encoder(torch.unsqueeze(his_retrieve, dim=0))  # double
        # rct_r_emb = self.rct_encoder(torch.unsqueeze(rct_retrieve, dim=0))
        his_r_emb = self.his_encoder(his_retrieve)  # double
        rct_r_emb = self.rct_encoder(rct_retrieve)
        # inter = self.rct_encoder(inter)

        ei = self.rct_encoder(ei.double())
        eni = self.rct_encoder(eni.double())

        his_r_emb = self.adaptor(his_r_emb)

        pos_pred = self.rec_model(eu.double(), ei, his_r_emb, rct_r_emb)
        neg_pred = self.rec_model(eu.double(), eni, his_r_emb, rct_r_emb)

        score = pos_pred - neg_pred
        # bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(score))         #----------------------------------------------------------
        bpr_loss = -torch.sum(torch.nn.functional.logsigmoid(score))
        return bpr_loss
        # g = make_dot(pos_pred)
        # g.render(filename='graph', view=False)

        # pos_loss = torch.nn.BCELoss(pos_pred, 1.)

        # neg_loss = torch.nn.BCELoss(neg_pred, 0.)

        # bpr_loss = pos_loss + neg_loss
        pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_pred) + 1e-15))
        neg_loss = -torch.mean(torch.log(1 - torch.sigmoid(neg_pred) + 1e-15))
        loss = pos_loss + neg_loss
        # loss = BCEloss(pos_pred, neg_pred)

        return loss  # bpr_loss

    # using to
    def test1(self, user, item, neg_items, rct_retrieve, topK=5):
        # user, item, negative items
        # retrieve only from corresponding reservoir
        ei = self.item_embedding(item)
        eu = self.user_embedding(user)

        ei = self.rct_encoder(ei)
        rct_r_emb = self.rct_encoder(rct_retrieve)

        pos_value = self.rec_model(eu, ei, rct_r_emb)

        pos_negs_scores = [pos_value]
        for n_item in neg_items:
            neg_ei = self.item_embedding(n_item)
            neg_ei = self.rct_encoder(neg_ei)
            neg_score = self.rec_model(eu, neg_ei, rct_r_emb)
            pos_negs_scores.append(neg_score)
        pos_negs_scores = torch.stack(pos_negs_scores)

        _, rank = torch.topk(pos_negs_scores, topK)
        pos_rank_idx = (rank < 1).nonzero()
        if torch.any((rank < 1).sum(-1) > 1):
            print("user embedding:", user)
            print("item embedding:", item, neg_items)
            print("score:", pos_negs_scores[0])
            print("rank", rank)
            print("pos_rank:", pos_rank_idx)
            # np.save("pos_negs.npy",pos_negs_interaction.data.cpu().numpy())
            raise RuntimeError("compute rank error")
        have_hit_num = pos_rank_idx.shape[0]
        have_hit_rank = pos_rank_idx[:, 1].float()
        # DCG_ = 1.0 / torch.log2(have_hit_rank+2)
        # NDCG = DCG_ * torch.log2(torch.Tensor([2.0]).cuda())
        if have_hit_num > 0:
            batch_NDCG = have_hit_num / torch.log2(have_hit_rank + 2)  # NDCG is equal to this format
            batch_NDCG = batch_NDCG.sum()
        else:
            batch_NDCG = 0
        Batch_hit = have_hit_num * 1.0

        return Batch_hit, batch_NDCG, pos_rank_idx[:, 0]

    def test(self, user, item, neg_items, his_retrieve, rct_retrieve, topK=5, type="his"):
        # user, item, negative items
        # retrieve only from corresponding reservoir
        ei = self.item_embedding(item)
        eu = self.user_embedding(user)

        # ei = self.his_encoder(ei.double())
        his_r_emb = self.his_encoder(his_retrieve)
        rct_r_emb = self.rct_encoder(rct_retrieve)
        enhanced_ret_emb = self.retrieve_aggregator(torch.cat((his_r_emb, rct_r_emb), dim=-1))

        eir = self.rct_encoder(ei.double())
        eih = self.his_encoder(ei.double())
        enhanced_item_emb = self.item_aggregator(torch.cat((eih, eir), dim=-1))

        neg_items = self.item_embedding(neg_items)
        # neg_items = self.rct_encoder(neg_items.double())
        neir = self.rct_encoder(neg_items.double())
        neih = self.his_encoder(neg_items.double())
        enhanced_nitem_emb = self.item_aggregator(torch.cat((neih, neir), dim=-1))

        # merged_retrieve = self.retrieval_merge_layer(torch.cat((his_r_emb, rct_r_emb), dim=-1))

        if eu.dim() != ei.dim():
            # pos_value = self.rec_model(eu, ei, his_r_emb, rct_r_emb)
            pos_score = self.rec_model(torch.unsqueeze(eu, dim=1).repeat_interleave(ei.shape[1], dim=1),
                                       enhanced_item_emb,
                                       torch.unsqueeze(enhanced_ret_emb, dim=1).repeat_interleave(ei.shape[1], dim=1))
        else:
            pos_score = self.rec_model(eu, enhanced_item_emb, enhanced_ret_emb)
        if eu.dim() != neg_items.dim():
            neg_score = self.rec_model(torch.unsqueeze(eu, dim=1).repeat_interleave(neg_items.shape[1], dim=1),
                                       enhanced_nitem_emb,
                                       torch.unsqueeze(enhanced_ret_emb, dim=1).repeat_interleave(neg_items.shape[1],
                                                                                                  dim=1))
        else:
            neg_score = self.rec_model(eu, enhanced_nitem_emb, enhanced_ret_emb)

        # eu = torch.unsqueeze(eu, dim=1).repeat_interleave(neg_items.shape[1], dim=1)
        if neg_score.dim() != pos_score.dim():
            neg_score = torch.squeeze(neg_score)
        pos_negs_scores = torch.cat((pos_score, neg_score), dim=1)
        pos_negs_scores = torch.squeeze(pos_negs_scores, dim=-1)

        test_result = evaluate_method(pos_negs_scores, [5, 10, 20], ['HR', 'NDCG'])
        return test_result
        # pos_negs_scores =  pos_negs_interaction
        # the we have compute all score for pos and neg interactions ,each row has scorces of one pos inter and neg_num(99)  neg inter

        _, rank = torch.topk(pos_negs_scores, topK)
        pos_rank_idx = (rank < topK).nonzero()
        if torch.any((rank < 1).sum(-1) > 1):
            print("user embedding:", user)
            print("item embedding:", item, neg_items)
            print("score:", pos_negs_scores[0])
            print("rank", rank)
            print("pos_rank:", pos_rank_idx)
            # np.save("pos_negs.npy",pos_negs_interaction.data.cpu().numpy())
            raise RuntimeError("compute rank error")
        have_hit_num = pos_rank_idx.shape[0]
        have_hit_rank = pos_rank_idx[:, 1].float()
        # DCG_ = 1.0 / torch.log2(have_hit_rank+2)
        # NDCG = DCG_ * torch.log2(torch.Tensor([2.0]).cuda())
        if have_hit_num > 0:
            batch_NDCG = self.cal_ndcg(have_hit_rank,
                                       topK)  # 1 / torch.log2(have_hit_rank + 2)  # NDCG is equal to this format
            # batch_NDCG = batch_NDCG.sum()
        else:
            batch_NDCG = 0
        Batch_hit = have_hit_num * 1.0

        return Batch_hit / topK, batch_NDCG, pos_rank_idx[:, 0]

    def cal_ndcg(self, have_hit_rank, topk):
        dcg = 0
        idcg = 0
        for idx in have_hit_rank:
            dcg += 1 / math.log2(idx + 2)
        for k in range(1, 1 + topk):
            idcg += 1 / math.log2(k + 2)
        return dcg / idcg


class MAUpdate(nn.Module):
    def __init__(self, args):
        super(MAUpdate, self).__init__()
        print("----load MAUpdate---")
        self.cuda = args.cuda
        self.data_name = args.data_name
        self.classification = args.classification
        self.off_tr_batch_size = args.off_tr_batch_size
        self.onl_tr_batch_size = args.onl_tr_batch_size
        self.topK = args.topK

        self.attr_num = args.u_attr + args.i_attr
        self.retrieve_attr = args.i_attr

        self.h_r_size = args.his_reservoir_size
        self.r_r_size = args.rct_reservoir_size

        self.lr_trans = args.lr_trans
        self.lr_reg = args.lr_reg

        self.m = args.m
        self.epoch = args.epoch
        self.lr_rec = args.lr_rec
        self.maxnorm_grad = args.maxnorm_grad  # gradients clip when update the model
        self.clip_grad = args.clip_grad
        # self.writer = SummaryWriter(args.writer_path)

        self.retrieve_max_num = args.max_num  # max num of retrieve results

        self.emb_dim = args.embedding_dim
        self.rec_layers = args.rec_layers
        self.act_func = args.act_func

        self.h_r_size = args.his_reservoir_size
        self.r_r_size = args.rct_reservoir_size

        self.u_attr = args.u_attr
        self.i_attr = args.i_attr

        # if self.data_name == 'AD':
        #     self.item_load = ml_item(config)
        #     self.user_load = ml_user(config)
        # elif self.data_name == 'yelp':
        #     self.item_load = yelp_item(config)
        #     self.user_load = yelp_user(config)
        # else:
        #     self.item_load = adressa_item(config)
        #     self.user_load = adressa_user(config)
        # self.item_emb = ItemEmbedding(self.layer_num, default_info[self.data_name]['i_in_dim'] * self.embedding_dim,
        #                               self.embedding_dim, activation=activation_func('relu'))  # .to(self.device)
        # self.user_emb = UserEmbedding(self.layer_num, default_info[self.data_name]['u_in_dim'] * self.embedding_dim,
        #                               self.embedding_dim, activation=activation_func(
        #         'relu'))

        if self.data_name == 'AD':
            self.item_emb = ad_item_emb(config)
            self.user_emb = ad_user_emb(config)
        elif self.data_name == 'yelp':
            self.item_emb = yelp_item_emb(config)
            self.user_emb = yelp_user_emb(config)
        else:  # Adressa
            self.item_emb = adressa_item_emb(config)
            self.user_emb = adressa_user_emb(config)

        self.his_reservoir = Reservoir.HisReservoir(self.h_r_size, self.emb_dim, self.u_attr,
                                                    self.i_attr)
        self.rct_reservoir = Reservoir.RecentReservoir(self.r_r_size, self.emb_dim, self.u_attr, self.i_attr)

        self.his_transformer = Transformer.RetrieveTransformer(self.emb_dim * self.retrieve_attr,
                                                               # * self.retrieve_max_num,
                                                               1)  # i_emb = self.item_attention(i_emb, i_emb, i_emb, i_emb)
        self.rct_transformer = Transformer.RetrieveTransformer(self.emb_dim * self.retrieve_attr,
                                                               1)  # (self.emb_dim * self.retrieve_max_num, 1)

        # # as embedding layer
        self.his_encoder = His_encoder(self.emb_dim * self.retrieve_attr)
        self.rct_encoder = Rct_encoder(self.emb_dim * self.retrieve_attr)

        # self.retrieve_merge = RetrieveFuse(self.emb_dim)  # add or concat+convolution

        # self.his_adaptor = Adaptor(self.emb_dim * self.i_attr, 2)  # to each attr

        self.recommender = MLPbasemodel(self.emb_dim, self.rec_layers, self.act_func,
                                        dataset_type=self.data_name, classification=self.classification)

        # load layer, embedding layer, encoder, adaptor, recommender
        # self.RMAOU_Recommender = AdaptorRecommender(self.recommender, self.user_emb, self.item_emb, self.user_load, self.item_load, self.his_adaptor)    # self, rec_module, u_emb_module, i_emb_module, u_load, i_load, his_adaptor)

        self.RMAOU_Recommender = AdaptorRecommender(args, self.recommender, self.user_emb, self.item_emb,
                                                    self.his_encoder,
                                                    self.rct_encoder)  # , self.his_transformer, self.rct_transformer, self.his_reservoir, self.rct_reservoir)  # tess using self.RMAOU_Recommender.test()
        # define the regularization
        # self.reg = None
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        if self.cuda:
            self.RMAOU_Recommender.cuda()
            self.his_transformer.cuda()
            self.rct_transformer.cuda()
            # self.last_grad = self.last_grad.cuda()
            # self.last_model = self.last_model.cuda()

        self.last_grad = {}
        self.last_model = {}
        self.max_param = 0
        self.get_regularizer()
        # self.max_param = 100
        self.reg = Regularization(1e-5, self.max_param)
        # if self.cuda:
        self.reg.to(device)

        self.optimizer = torch.optim.Adam(self.RMAOU_Recommender.parameters(), lr=self.lr_rec)
        self.online_optimizer = torch.optim.Adam([{'params': self.RMAOU_Recommender.rec_model.parameters()},
                                                  {'params': self.RMAOU_Recommender.rct_encoder.parameters()}],
                                                 lr=self.lr_rec)

        # self.transformers_optimizer = torch.optim.Adam(self.his_transformer.parameters(), lr=self.lr_rec,
        #                                                weight_decay=0)
        self.transformers_optimizer = torch.optim.Adam(
            [{'params': self.his_transformer.parameters()}, {'params': self.rct_transformer.parameters()}],
            lr=self.lr_trans)
        self.reg_optimizer = torch.optim.Adam(self.reg.parameters(), lr=self.lr_reg)

    def get_regularizer(self):
        for name, param in self.RMAOU_Recommender.named_parameters():
            if 'his_encoder' in name:
                continue
            # self.max_param = max(param.size(-1), self.max_param)
            self.max_param += 1
            if self.cuda:
                self.last_model[name] = torch.zeros(param.size()).cuda()
                self.last_grad[name] = torch.zeros(param.size()).cuda()
            else:
                self.last_model[name] = torch.zeros(param.size())
                self.last_grad[name] = torch.zeros(param.size())

        # for name, param in self.RMAOU_Recommender.rec_model.named_parameters():  # input loading don't need train
        #     # if 'item_embedding' in name:  # the dimension of item_embedding is large
        #     #     continue
        #     param.requires_grad = True
        #     # self.max_param = max(param.numel(), self.max_param)  # too big
        #     self.max_param = max(param.size(-1), self.max_param)
        #     if self.cuda:
        #         self.last_model[name] = torch.zeros(param.size()).cuda()
        #         self.last_grad[name] = torch.zeros(param.size()).cuda()
        #     else:
        #         self.last_model[name] = torch.zeros(param.size())
        #         self.last_grad[name] = torch.zeros(param.size())
        #
        # for name, param in self.RMAOU_Recommender.adaptor.named_parameters():  # input loading don't need train
        #     param.requires_grad = True
        #     self.max_param = max(param.size(-1), self.max_param)
        #     if self.cuda:
        #         self.last_model[name] = torch.zeros(param.size()).cuda()
        #         self.last_grad[name] = torch.zeros(param.size()).cuda()

    def get_reg_param(self, new_grad):
        model = {}
        grad = {}
        i = 0
        for name, param in self.RMAOU_Recommender.named_parameters():
            if 'his_encoder' in name:
                i += 1
                continue
            model[name] = param
            grad[name] = new_grad[i]
            i += 1
        return model, grad

    def freeze_parameter(self):
        # freeze two encoder parameter
        for name, param in self.RMAOU_Recommender.his_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.RMAOU_Recommender.rct_encoder.named_parameters():
            param.requires_grad = False

    def forward(self, user, item, neg_item, his_retrieve, rct_retrieve):  # train_xs, train_ys   forward2
        # retrieve from two reservoir
        '''
        :param user:
        :param item:
        :param neg_item:
        :param his_retrieve, rct_retrieve: after transformer
        :return: loss
        '''

        pos_score = self.RMAOU_Recommender(user, item, his_retrieve, rct_retrieve)

        neg_score = self.RMAOU_Recommender(user, neg_item, his_retrieve, rct_retrieve)

        # item_score = torch.mul(user_weight_new, item_weight_new).sum(dim=-1)
        # negitem_score = torch.mul(user_weight_new, negitem_weight_new).sum(dim=-1)
        pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_score) + 1e-15))
        neg_loss = -torch.mean(torch.log(1 - torch.sigmoid(neg_score) + 1e-15))
        loss = pos_loss + neg_loss

        # neg_loss = torch.nn.BCELoss(neg_score, 0.)
        # pos_loss = -torch.mean(torch.log(torch.sigmoid(item_score) + 1e-15))
        # neg_loss = -torch.mean(torch.log(1 - torch.sigmoid(negitem_score) + 1e-15))
        #     bpr_loss = pos_loss + neg_loss
        # else:
        #     score = item_score - negitem_score
        #     if norm:
        #         user_ = (user_weight_new ** 2).sum(dim=-1).sqrt()
        #         score = score / user_
        #     if adpative:
        #         pass
        #     bpr_loss = -torch.sum(F.logsigmoid(score))
        return loss

    def forward_1(self, user, item, neg_item, y, ret_emb, d_type="his"):
        # historical samples join to update the model

        pos_score = self.RMAOU_Recommender(user, item, ret_emb, state=d_type)

        neg_score = self.RMAOU_Recommender(user, neg_item, ret_emb, state=d_type)

        pos_loss = torch.nn.BCELoss(pos_score, y)

        neg_loss = torch.nn.BCELoss(neg_score, 0)
        bpr_loss = pos_loss + neg_loss
        return bpr_loss

    def forward_allinter(self, user, item, y, his_retrieve, rct_retrieve):  # train_xs, train_ys
        '''
        retrieve from two reservoir
        use interaction to update the model, regardless of whether it clicks or not
        :param his_retrieve, rct_retrieve: after transformer
        :return: loss
        '''

        pred_y = self.RMAOU_Recommender(user, item, his_retrieve, rct_retrieve)

        # item_score = torch.mul(user_weight_new, item_weight_new).sum(dim=-1)
        # negitem_score = torch.mul(user_weight_new, negitem_weight_new).sum(dim=-1)

        loss = torch.nn.BCELoss(pred_y, y)
        return loss

    @torch.no_grad()
    def momentum_update_rct_encoder(self):
        """
        Momentum update of the recent encoder
        self.m is a hyperparameter
        """
        for param_h, param_r in zip(  # update the parameters in each dimension
                self.his_encoder.parameters(), self.rct_encoder.parameters()
        ):
            param_h.data = param_h.data * self.m + param_r.data * (1.0 - self.m)

    '''Quadratic gradient return to update the adaptor '''

    def online_update_adaptor_recommender(self, d_t, test_data, period):
        # self.RMAOU_Recommender.load_state_dict(torch.load("Recommender.pt"))
        clip = self.clip_grad

        s_time = time.time()
        users, items, neg_items = d_t[0], d_t[1], d_t[2]
        batches = math.floor(len(users) / self.onl_tr_batch_size)

        if self.cuda:
            self.RMAOU_Recommender.cuda()
            # self.reg.cuda()
            self.rct_transformer.cuda()
            self.his_transformer.cuda()
            self.his_reservoir.pool = self.his_reservoir.pool.cuda()
            self.rct_reservoir.pool = self.rct_reservoir.pool.cuda()
        ###############################
        torch.backends.cudnn.enabled = False
        for epoch in range(50):  # self.epoch ---------------------------------
            self.RMAOU_Recommender.train()
            self.reg.train()
            # self.freeze_parameter()
            self.his_transformer.eval()
            self.rct_transformer.eval()

            momentum_update_time = 0
            loss_all = 0
            print("---before online training----")
            self.online_test(test_data, int(period) + 1)
            for i in range(batches):
                try:
                    b_users = list(users[self.onl_tr_batch_size * i:self.onl_tr_batch_size * (i + 1)])
                    b_items = list(items[self.onl_tr_batch_size * i:self.onl_tr_batch_size * (i + 1)])
                    # b_neg_items = list(neg_items[self.off_tr_batch_size * i:self.off_tr_batch_size * (i + 1)])
                except IndexError:
                    continue

                # self.RMAOU_Recommender.zero_grad()
                # self.reg.zero_grad()
                self.online_optimizer.zero_grad()

                self.reg_optimizer.zero_grad()
                if b_users == []:
                    print("-----", i, " no user")
                    continue
                if self.cuda:
                    b_users = torch.squeeze(torch.stack(b_users)).cuda()
                    b_items = torch.squeeze(torch.stack(b_items)).cuda()
                    # b_neg_items = torch.squeeze(torch.stack(b_neg_items)).cuda()
                else:
                    b_users = torch.squeeze(torch.stack(b_users))
                    b_items = torch.squeeze(torch.stack(b_items))
                    # b_neg_items = torch.squeeze(torch.stack(b_neg_items))

                u_emb = self.RMAOU_Recommender.user_embedding(b_users)
                if self.data_name == "AD":
                    u_emb = torch.unsqueeze(u_emb, dim=1)
                else:
                    u_emb = torch.unsqueeze(u_emb, dim=0)
                his_retrieve = self.his_reservoir.retrieve(u_emb, self.retrieve_max_num)

                rct_retrieve = self.rct_reservoir.retrieve(u_emb, self.retrieve_max_num)
                his_retrieve = [torch.stack(h_r, dim=0) for h_r in his_retrieve]
                his_retrieve = torch.stack(his_retrieve, dim=0)
                his_retrieve = his_retrieve.permute(1, 0, 2)  # reshape letting size = tar_len * b_size * emb_dim
                rct_retrieve = [torch.stack(h_r, dim=0) for h_r in rct_retrieve]
                rct_retrieve = torch.stack(rct_retrieve, dim=0)
                rct_retrieve = rct_retrieve.permute(1, 0, 2)

                # his_retrieve.requires_grad = True
                # rct_retrieve.requires_grad = True

                his_r_emb = self.his_transformer(his_retrieve, his_retrieve, his_retrieve,
                                                 his_retrieve)  # .double()
                rct_r_emb = self.rct_transformer(rct_retrieve, rct_retrieve, rct_retrieve, rct_retrieve)

                del his_retrieve, rct_retrieve
                loss_batch = self.RMAOU_Recommender(b_users, b_items, his_r_emb, rct_r_emb)

                # only have rec_model and adaptor
                grad = torch.autograd.grad(loss_batch, self.RMAOU_Recommender.parameters(), retain_graph=True)
                reg_m, reg_g = self.get_reg_param(grad)
                reg_loss = self.reg(self.last_grad, self.last_model, reg_g, reg_m)
                del grad
                print("average loss %f of batches %d" % (loss_batch.item(), i))
                loss_batch = loss_batch + reg_loss
                loss_batch.backward()
                loss_all += loss_batch.data

                if clip:
                    # torch.nn.utils.clip_grad_value_(self.transfer.parameters(),2)
                    max_norm = self.maxnorm_grad
                    torch.nn.utils.clip_grad_norm_(self.RMAOU_Recommender.parameters(), max_norm, norm_type=2)

                self.online_optimizer.step()
                self.reg_optimizer.step()

                self.last_grad = reg_g
                self.last_model = reg_m
                # momentum_update_time += 1
                # if momentum_update_time % 4 == 0:
                self.momentum_update_rct_encoder()
                gc.collect()
                torch.cuda.empty_cache()
                self.online_test(test_data, int(period) + 1)
            # self.writer.add_scalars('online_loss', loss_all / self.off_tr_batch_size, self.epoch)

            print(period, "period: %d epoch's loss:%d" % (epoch, loss_all / batches))
            print(epoch, " epoch train time cost:", time.time() - s_time)
        period = int(period)

        torch.save(self.RMAOU_Recommender.state_dict(),
                   "./save_model/" + self.data_name + "/merge_two_res/" + "Recommender%d.pt" % (period))
        torch.save(self.reg.state_dict(),
                   "./save_model/" + self.data_name + "/merge_two_res/" + "regularizer%d.pt" % (period))
        torch.save(self.last_grad,
                   "./save_model/" + self.data_name + "/merge_two_res/" + "last_grad%d.pt" % (period))
        torch.save(self.last_model,
                   "./save_model/" + self.data_name + "/merge_two_res/" + "last_model%d.pt" % (period))  # str(period) +

    def offline_update_recommender(self, train_sets, test_data):

        # offline update the overall model and obtain the well initial parameters and hiper-parameters
        # two transfer, two encoder, adaptor, recommender, reservoirs
        # update the RMAOU_Recommender and two transformers respectively

        print("------begin offline update-----")
        clip = self.clip_grad
        if self.data_name == "yelp":
            self.his_reservoir.pool = torch.from_numpy(np.load('yelp_his_reservoir.npy'))
            self.rct_reservoir.pool = torch.from_numpy(np.load('yelp_rct_reservoir.npy'))
        elif self.data_name == "AD":
            self.his_reservoir.pool = torch.from_numpy(
                np.load('ad_his_reservoir.npy'))  # ad_his_reservoir.npyad_his_reservoir.npy
            self.rct_reservoir.pool = torch.from_numpy(np.load('ad_rct_reservoir.npy'))  # ad_rct_reservoir.npy
        else:
            self.his_reservoir.pool = torch.from_numpy(np.load('adressa_his_reservoir.npy'))
            self.rct_reservoir.pool = torch.from_numpy(np.load('adressa_rct_reservoir.npy'))
        # print("----pool--",self.his_reservoir.pool.shape)

        if self.cuda:
            self.his_reservoir.pool = self.his_reservoir.pool.cuda()
            self.rct_reservoir.pool = self.rct_reservoir.pool.cuda()
        # self.his_reservoir.pool = np.load('his_reservoir.npy') # torch.from_numpy(np.load('his_reservoir.npy'))
        # self.rct_reservoir.pool = np.load('recent_reservoir.npy')# torch.from_numpy(np.load('recent_reservoir.npy'))

        # if self.cuda:
        #    self.his_reservoir.pool.cuda()
        #    self.rct_reservoir.pool.cuda()
        users, items, neg_items = train_sets[0], train_sets[1], train_sets[2]
        batches = math.floor(len(users) / self.off_tr_batch_size)
        for epoch in range(self.epoch):
            print("---------epoch", epoch)
            # self.his_reservoir.init_pool(train_sets[:self.h_r_size, :])
            # self.rct_reservoir.init_pool(train_sets[-self.r_r_size:, :])
            s_time = time.time()
            # self.RMAOU_Recommender.eval()  # ***************************************************************************
            self.RMAOU_Recommender.train()

            self.his_transformer.train()
            self.rct_transformer.train()

            momentum_update_time = 0  # to control the rct_encoder momentum update
            loss_all = 0
            # for batch_id, (u_id, user, item, neg_item, y) in enumerate(train_data):
            # for batch_id, (user, item, neg_item) in enumerate(train_data):
            for i in range(batches):
                try:
                    b_users = list(users[self.off_tr_batch_size * i:self.off_tr_batch_size * (i + 1)])
                    b_items = list(items[self.off_tr_batch_size * i:self.off_tr_batch_size * (i + 1)])
                    # b_neg_items = list(neg_items[self.off_tr_batch_size * i:self.off_tr_batch_size * (i + 1)])
                except IndexError:
                    continue

                self.RMAOU_Recommender.zero_grad()
                self.his_transformer.zero_grad()
                self.rct_transformer.zero_grad()
                self.reg.zero_grad()
                self.optimizer.zero_grad()
                self.transformers_optimizer.zero_grad()
                self.reg_optimizer.zero_grad()
                if self.cuda:
                    b_users = torch.squeeze(torch.stack(b_users)).cuda()
                    b_items = torch.squeeze(torch.stack(b_items)).cuda()
                    # b_neg_items = torch.squeeze(torch.stack(b_neg_items)).cuda()
                else:
                    b_users = torch.squeeze(torch.stack(b_users))
                    b_items = torch.squeeze(torch.stack(b_items))
                    # b_neg_items = torch.squeeze(torch.stack(b_neg_items))
                # --------------------------------------------------
                u_emb = self.RMAOU_Recommender.user_embedding(b_users)

                if self.data_name == "AD" or self.data_name == "yelp":
                    u_emb = torch.unsqueeze(u_emb, dim=1)  # AD
                elif self.data_name == "adressa":
                    u_emb = torch.squeeze(u_emb, dim=0)

                his_retrieve = self.his_reservoir.retrieve(u_emb, self.retrieve_max_num)

                rct_retrieve = self.rct_reservoir.retrieve(u_emb, self.retrieve_max_num)
                his_retrieve = [torch.stack(h_r, dim=0) for h_r in his_retrieve]
                his_retrieve = torch.stack(his_retrieve, dim=0)
                his_retrieve = his_retrieve.permute(1, 0, 2)  # reshape letting size = tar_len * b_size * emb_dim
                rct_retrieve = [torch.stack(h_r, dim=0) for h_r in rct_retrieve]
                rct_retrieve = torch.stack(rct_retrieve, dim=0)
                rct_retrieve = rct_retrieve.permute(1, 0, 2)

                his_r_emb = self.his_transformer(his_retrieve, his_retrieve, his_retrieve,
                                                 his_retrieve)  # .double()
                rct_r_emb = self.rct_transformer(rct_retrieve, rct_retrieve, rct_retrieve, rct_retrieve)
                # print("------",his_r_emb.size())

                # b_loss = self.RMAOU_Recommender(b_users, b_items, b_neg_items, his_r_emb, rct_r_emb)
                b_loss = self.RMAOU_Recommender(b_users, b_items, his_r_emb, rct_r_emb)
                # y = y.cuda().long()
                # Adressa only has id not attributions
                # user = user.long()
                # item = item.long()
                # neg_item = neg_item.long()

                grad = torch.autograd.grad(b_loss, self.RMAOU_Recommender.parameters(), retain_graph=True)
                temp = copy.deepcopy(
                    self.RMAOU_Recommender.state_dict())  # ------------------------------------------------------------------------------
                reg_m, reg_g = self.get_reg_param(grad)  # needed parameters
                reg_loss = self.reg(self.last_grad, self.last_model, reg_g, reg_m)
                # print("average loss %f of batches %d" % (b_loss.item(), i)) ##############################
                b_loss = b_loss + reg_loss
                b_loss.backward()
                if clip:
                    max_norm = self.maxnorm_grad
                    torch.nn.utils.clip_grad_norm_(self.RMAOU_Recommender.parameters(), self.maxnorm_grad, norm_type=2)
                    torch.nn.utils.clip_grad_value_(self.his_transformer.parameters(), 1.1)
                    torch.nn.utils.clip_grad_value_(self.rct_transformer.parameters(), 1.1)
                self.optimizer.step()
                self.transformers_optimizer.step()
                self.reg_optimizer.step()

                self.last_grad = reg_g
                self.last_model = reg_m
                # temp = copy.deepcopy(self.RMAOU_Recommender.state_dict().keys())
                # print(self.RMAOU_Recommender.state_dict().keys()==temp)
                # b_losses = torch.stack(b_losses).mean(0)
                loss_all += b_loss.item()
                # print("average loss %f of batches %d" % (loss.item(), i))
                # self.writer.add_scalar('offline_loss in a epoch', b_loss.item(), batches)

                gc.collect()
                torch.cuda.empty_cache()

                # print("is model changed:",temp.values() != self.RMAOU_Recommender.state_dict().values())  # ---------------------------------------------------------------
            # loss_all = loss_all / (i + 1)  # average loss of all batch

            # self.writer.add_scalar('offline_loss of a epoch', loss_all / self.off_tr_batch_size, self.epoch)
            print("average loss %f of epoch %d" % (loss_all / batches, epoch))

            # momentum_update_time += 1
            # if momentum_update_time % 4 == 0:
            #     self.momentum_update_rct_encoder()
            print("---one epcohs train time cost----", time.time() - s_time)

            # test model----------------------------------
            self.online_test(test_data, 26)
            if epoch > 50:
                torch.save(self.RMAOU_Recommender.state_dict(),
                           "./save_model/" + self.data_name + "_dol/Recommender.pt")
                torch.save(self.his_transformer.state_dict(),
                           "./save_model/" + self.data_name + "_dol/his-transformer.pt")
                torch.save(self.rct_transformer.state_dict(),
                           "./save_model/" + self.data_name + "_dol/rct-transformer.pt")
                torch.save(self.last_grad,
                           "./save_model/" + self.data_name + "_dol/last_grad.pt")
                torch.save(self.last_model,
                           "./save_model/" + self.data_name + "_dol/last_model.pt")
                torch.save(self.reg.state_dict(),
                           "./save_model/" + self.data_name + "_dol/regularizer.pt")
        # reservoir_data = self.get_reservoir_data(train_sets, self.emb_dim)
        # self.his_reservoir.update(reservoir_data[:len(train_sets)-self.r_size, :])
        # self.rct_reservoir.update(reservoir_data[-self.r_size:, :])
        # np.save('his_reservoir.npy', self.his_reservoir.pool)
        # np.save('recent_reservoir.npy', self.rct_reservoir.pool)

    def offline_train(self, off_datasets, test_data):
        print("-------------offline train the recommender----------")
        self.offline_update_recommender(off_datasets, test_data)
        # self.online_test(test_data, 26)

        # print("-------------offline train two transformers---------")
        # self.offline_update_transformers(off_datasets)

        # update and store two reservoirs
        r_data = self.get_reservoir_data(off_datasets, self.emb_dim)
        self.his_reservoir.update(r_data)
        self.rct_reservoir.update(r_data)
        torch.save(self.his_reservoir.pool,
                   "./save_model/" + self.data_name + "/his_reservoir_.npy")

        torch.save(self.rct_reservoir.pool,
                   "./save_model/" + self.data_name + "/rct_reservoir_.npy")

    def get_reservoir_data(self, data, emb_dim, data_name="AD"):  # [u_id, ad_id]
        # set the row data to [[u_id, item_emb],[]]  ************************
        users, items = data[0], data[1]
        if self.cuda:
            users = users.cuda()
            items = items.cuda()
        new_data = np.empty([len(users), emb_dim * self.attr_num], dtype=float)
        # for index, (user, item, _) in enumerate(data):
        for i in range(len(users)):
            u_emb = self.RMAOU_Recommender.user_embedding(users[i])
            i_emb = self.RMAOU_Recommender.item_embedding(items[i])
            u_i = torch.cat((u_emb, i_emb), dim=1)
            new_data[i] = u_i.detach().numpy()
        return new_data

    def online_test(self, test_data, period):
        # now_test = torch.utils.data.DataLoader(test_data,
        #                                        batch_size=1024,
        #                                        num_workers=2,
        #                                        pin_memory=False
        #                                        )
        # recall, ndcg, hr = self.test_model(test_data, topK=self.topK)
        test_result = self.test_model(test_data, topK=self.topK)
        # print(
        #     "Testsets is next train set, Test on period:", period,
        #     "---before train transfer test:recall:{:.4f} ndcg:{:.4f} hr:{:.4f}".format(recall, ndcg, hr))
        # print("Testsets is next train set, Test on period:", period, test_result)

    def test_model(self, test_set, topK=5, batch_size=32):
        print("-----begin test---------")
        self.RMAOU_Recommender.eval()
        self.rct_transformer.eval()
        self.his_transformer.eval()
        self.reg.eval()

        if self.data_name == "adressa":
            self.his_reservoir.pool = torch.from_numpy(np.load('adressa_his_reservoir.npy'))
            self.rct_reservoir.pool = torch.from_numpy(np.load('adressa_rct_reservoir.npy'))
        elif self.data_name == "AD":
            self.his_reservoir.pool = torch.from_numpy(np.load('ad_his_reservoir.npy'))
            self.rct_reservoir.pool = torch.from_numpy(np.load('ad_rct_reservoir.npy'))
        else:
            self.his_reservoir.pool = torch.from_numpy(np.load('yelp_his_reservoir.npy'))
            self.rct_reservoir.pool = torch.from_numpy(np.load('yelp_rct_reservoir.npy'))

        if self.cuda:
            self.his_reservoir.pool = self.his_reservoir.pool.cuda()
            self.rct_reservoir.pool = self.rct_reservoir.pool.cuda()

        num_test = 0
        recall_all, ndcg_all = [], []
        all_result = []
        # gpu_id = model.user_laten.weight.device.index
        device = self.RMAOU_Recommender.his_encoder.rnn.weight_ih_l0.device
        # print(device)
        users, items, neg_items = test_set[0], test_set[1], test_set[2]
        batches = math.ceil(len(users) / batch_size)

        for i in range(batches):
            try:
                if i == batches - 1:
                    b_users = list(users[batch_size * i:])
                    if self.data_name == "AD":
                        b_items = list(items[batch_size * i:][:, :, 0])  # -------------------------------------
                    else:
                        b_items = list(items[batch_size * i:][:, 0])
                    b_neg_items = list(neg_items[batch_size * i:])
                else:
                    b_users = list(users[batch_size * i:batch_size * (i + 1)])
                    if self.data_name == "AD":
                        b_items = list(
                            items[batch_size * i:batch_size * (i + 1)][:, :, 0])  # ------------------------------
                    else:
                        b_items = list(items[batch_size * i:batch_size * (i + 1)][:, 0])
                    b_neg_items = list(neg_items[batch_size * i:batch_size * (i + 1)])

                if len(b_users) != len(b_items):
                    b_users = b_users[:min(len(b_users), len(b_items))]
                    b_items = b_items[:min(len(b_users), len(b_items))]
                    b_neg_items = b_neg_items[:min(len(b_users), len(b_items))]
                if b_users == [] or b_items == []: continue
            except IndexError:
                continue
            if device != torch.device('cpu'):
                # print("************test on gpu")
                b_users = torch.squeeze(torch.stack(b_users)).to(device)
                b_items = torch.squeeze(torch.stack(b_items)).to(device)
                b_neg_items = torch.squeeze(torch.stack(b_neg_items)).to(device)
            else:
                b_users = torch.squeeze(torch.stack(b_users))
                b_items = torch.squeeze(torch.stack(b_items))
                b_neg_items = torch.squeeze(torch.stack(b_neg_items))

            u_emb = self.RMAOU_Recommender.user_embedding(b_users)
            if self.data_name == "AD" or self.data_name == "yelp":
                # u_emb = torch.unsqueeze(u_emb, dim=1)  # AD
                u_emb = torch.squeeze(u_emb, dim=0)
            elif self.data_name == "adressa":
                u_emb = torch.squeeze(u_emb, dim=0)
            his_retrieve = self.his_reservoir.retrieve(u_emb, self.retrieve_max_num)

            rct_retrieve = self.rct_reservoir.retrieve(u_emb, self.retrieve_max_num)
            his_retrieve = [torch.stack(h_r, dim=0) for h_r in his_retrieve]
            his_retrieve = torch.stack(his_retrieve, dim=0)
            his_retrieve = his_retrieve.permute(1, 0, 2)  # reshape letting size = tar_len * b_size * emb_dim
            rct_retrieve = [torch.stack(h_r, dim=0) for h_r in rct_retrieve]
            rct_retrieve = torch.stack(rct_retrieve, dim=0)
            rct_retrieve = rct_retrieve.permute(1, 0, 2)

            his_retrieve.requires_grad = True
            rct_retrieve.requires_grad = True

            his_r_emb = self.his_transformer(his_retrieve, his_retrieve, his_retrieve,
                                             his_retrieve)  # .double()
            rct_r_emb = self.rct_transformer(rct_retrieve, rct_retrieve, rct_retrieve, rct_retrieve)
            # batch_hit, batch_ndcg, _
            b_test_result = self.RMAOU_Recommender.test(b_users, b_items, b_neg_items, his_r_emb, rct_r_emb,
                                                        topK=topK)  # test2(self,user, item, neg_items, his_retrieve, rct_retrieve, topK=20)
            # print("----------------", b_test_result)#batch_hit, batch_ndcg)
            recall_all.append(b_test_result["HR@5"])  # batch_hit)
            ndcg_all.append(b_test_result["NDCG@5"])  # batch_ndcg)
            num_test += b_users.shape[0]
            all_result.append(b_test_result)

        print("-----ndcg5:{:<.4f}, hr5: {:<.4f}, ndcg10:{:<.4f}, hr10: {:<.4f}, ndcg20:{:<.4f}, hr20: {:<.4f}".format(
            sum([res["NDCG@5"] for res in all_result]) / len(all_result),
            sum([res["HR@5"] for res in all_result]) / len(all_result),
            sum([res["NDCG@10"] for res in all_result]) / len(all_result),
            sum([res["HR@10"] for res in all_result]) / len(all_result),
            sum([res["NDCG@20"] for res in all_result]) / len(all_result),
            sum([res["HR@20"] for res in all_result]) / len(all_result)))
        return all_result
        return sum(recall_all) / num_test, sum(ndcg_all) / num_test, sum(recall_all) / num_test / topK

    # store the model and load the model
    def store_model(self, period):
        torch.save(self.RMAOU_Recommender.state_dict(),
                   "./save_model/" + self.data_name + str(period) + "-Recommender.pt")

