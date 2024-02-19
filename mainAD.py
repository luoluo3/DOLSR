import argparse

import numpy as np
import pandas as pd
import torch

from Reservoir.Reservoir import HisReservoir, RecentReservoir, set_data
from data_process.DataGenerator import OfflineDataset, OnlineDatasets
from Model.Momentum_Adaptor import MAUpdate
# from Model.MA_static import MAUpdate
# from Model.MA_noneg import MAUpdate
from Model.MAall_noneg import MAUpdate

from Model.MA_tworeg import MAUpdate

def get_parse():
    parser = argparse.ArgumentParser(description='parameters in our RRMA.')

    parser.add_argument('--data_name', default="AD",  # i.e Adressa
                        help='dataset name: yelp or adressa or AD')
    parser.add_argument('--data_path', default='./datasets/AD/streaming/',  #/home/RRMA-master/datasets/
                        help='dataset path')

    parser.add_argument('--off_path', default='./datasets/AD/offline/',  # /home/RRMA-master/datasets/
                        help='dataset path')
    parser.add_argument('--online_path', default='./datasets/AD/online/',  # /home/RRMA-master/datasets/
                        help='dataset path')

    parser.add_argument('--his_reservoir_size', default=50000,  #
                        help='reservoir size')
    parser.add_argument('--rct_reservoir_size', default=5000,  #
                        help='reservoir size')
    parser.add_argument('--embedding_dim', default=32,  #
                        help='embedding size')
    parser.add_argument('--classification', default=False,  #
                        help='if the task is muti classification')
    parser.add_argument('--rec_layers', default=3,  #
                        help='the layer of base recommender')


    parser.add_argument('--cuda', default=False,  #
                        help='gpu can be used')
    parser.add_argument('--off_tr_batch_size', default=128,  #
                        help='offline train batch size')
    parser.add_argument('--onl_tr_batch_size', default=256,  #
                        help='online train batch size')
    parser.add_argument('--topK', default=5,  #
                        help='topK')

    parser.add_argument('--m', default=0.9,  #
                        help='the coefficient of momentum update the rct_encoder')

    parser.add_argument('--epoch', default=100,  #
                        help='the train epoches')
    parser.add_argument('--batch_size', default=64,  #
                        help='the train batch size')
    parser.add_argument('--lr_rec', default=1e-5,  #
                        help='the learning rate of recommender')

    parser.add_argument('--max_num', default=50,  #
                        help='the max num of retrieve results')
    parser.add_argument('--act_func', default='gelu',  #  relu
                        help='the activate function')

    parser.add_argument('--neg_num', default=95,  #
                        help='the num of the negative items')

    parser.add_argument('--maxnorm_grad', default=3.0,  #
                        help='the num of the negative items')
    parser.add_argument('--clip_grad', default=True,  # whether or not to clip the gradients?       False
                        help='whether or not to clip the gradients')
    parser.add_argument('--clip_value', default=1.1,  # whether or not to clip the gradients?       False
                        help='the value to clip the gradients')
    parser.add_argument('--lr_trans', default=1e-5,  # whether or not to clip the gradients?`
                        help='leanring rate of two transformers')

    parser.add_argument('--lr_reg', default=1e-5,  # whether or not to clip the gradients?`
                        help='leanring rate of two transformers')
    parser.add_argument('--writer_path', default='./log/AD',  # whether or not to clip the gradients?
                        help='path to store the tensorboard')

    parser.add_argument('--u_attr', default=8,  # whether or not to clip the gradients?
                        help='attribution num of user')
    parser.add_argument('--i_attr', default=5,  # whether or not to clip the gradients?
                        help='attribution num of item')

    return parser

def get_reservoir_data(data, emb_dim, item_emb, data_name="AD"):  # [u_id, ad_id]
    # set the row data to [[u_id, item_emb],[]]  ************************
    new_data = np.empty([len(data), emb_dim + 1], dtype=np.long)
    for u_id, _, item, _ in enumerate(data):
        u_i = item_emb(item)
        u_i = torch.cat([torch.tensor([u_id]), u_i])
        new_data = np.append(new_data, u_i, axis=0)
    return new_data

# for AD: 20:12 for split the offline train sets and online test sets
if __name__ == "__main__":
    parser = get_parse()
    args = parser.parse_args()

    print("**********  RRMAOU parameters ***************")
    print(args)

    off_path = args.off_path  # '/data/wcx123/'
    online_path = args.online_path
    data_path = args.data_path
    data_name = args.data_name  # 'news'

    u_i_attr = args.u_attr+args.i_attr

    # off_train_list = [str(i) for i in range(0, 20)]
    # online_train_list = [str(j) for j in range(20, 32)]
    # online_test_list = [str(j) for j in range(20, 32)]  # 对应online_train中的每个train set的下一个中采样作为test
    # off_train_list = [str(i) for i in range(0, 18)]
    # online_train_list = [str(j) for j in range(18, 28)]
    # online_test_list = [str(j) for j in range(28, 32)]  # 对应online_train中的每个train set的下一个中采样作为test
    off_train_list = [str(i) for i in range(0, 24)]
    online_val_list = [str(j) for j in range(24, 26)]
    online_train_list = [str(j) for j in range(26, 32)]

    RRMAOU = MAUpdate(args)  # containing RRMA_recommender (recommender, two encoder, adaptor), two transformer, regularizer

    # inital two reservoirs
    his_r = HisReservoir(args.his_reservoir_size, args.embedding_dim, args.u_attr, args.i_attr)
    rct_r = RecentReservoir(args.rct_reservoir_size, args.embedding_dim, args.u_attr, args.i_attr)
    # based on 0.csv
    init_data = pd.read_csv(
        filepath_or_buffer=data_path + '0.csv',  # 文件的路径 + 名称
        sep=',',  # 分隔符 ---默认就是 ','
        encoding='utf-8',  # 文件的编码格式
        # header=0,  # 列索引---默认infer（自动识别）； 可以强制指定行下标 将该行作为列索引
        # index_col=0,  # 行索引---默认为None（自动生成序号作为行索引）；也可以指定列下标 将该列作为行索
        engine='python',  # 指定读取所用的引擎
    )
    # init the reservoir
    # init_data = set_data(init_data, args.embedding_dim * u_i_attr, RRMAOU.item_emb, RRMAOU.user_emb)
    # his_r.init_pool(init_data)
    # rct_r.init_pool(init_data)
    # np.save('ad_rct_reservoir.npy', rct_r.pool)
    # np.save('ad_his_reservoir.npy', his_r.pool)
    his_r.pool = np.load('ad_his_reservoir.npy')
    rct_r.pool = np.load('ad_rct_reservoir.npy')

    test_path = "./datasets/AD/online/test/onlineTest/"
    online_datasets = OnlineDatasets(test_path, online_train_list, online_train_list)

    test_set = online_datasets.get_test(26)

    # # offline train the overall model---------------------------------
    offline_datasets = OfflineDataset(off_path, off_train_list, args.neg_num)
    t_d = torch.load("train.pth")  # "user":users, "item":items, "neg_items"
    t_d = (t_d["user"], t_d["item"], t_d["neg_items"])
    RRMAOU.offline_train(t_d, test_set)

    test_path = "./datasets/AD/online/test/"
    for i in online_train_list:
        d_t = online_datasets.get_train(i, test_path)
        # update on each online train period
        # load last period model and output the test results on all online test sets
        if args.cuda:
            model_dict = torch.load("./save_model/" + data_name + "/Recommender2.pt")
            his_transformer_dict = torch.load("./save_model/" + data_name + "/his-transformer2.pt")
            rct_transformer_dict = torch.load("./save_model/" + data_name + "/rct-transformer2.pt")
            last_grad = torch.load(
                "./save_model/" + data_name + "/last_grad2.pt")
            last_model = torch.load(
                "./save_model/" + data_name + "/last_model2.pt")
            reg_dict = torch.load("./save_model/" + data_name + "/regularizer.pt")
        else:
            model_dict = torch.load("./save_model/" + data_name + "/Recommender2.pt", map_location='cpu')
            his_transformer_dict = torch.load("./save_model/" + data_name + "/his-transformer2.pt", map_location='cpu')
            rct_transformer_dict = torch.load("./save_model/" + data_name + "/rct-transformer2.pt", map_location='cpu')
            last_grad = torch.load(
                "./save_model/" + data_name + "/last_grad2.pt", map_location='cpu')
            last_model = torch.load(
                "./save_model/" + data_name + "/last_model2.pt", map_location='cpu')
            reg_dict = torch.load("./save_model/" + data_name + "/regularizer.pt", map_location='cpu')

        RRMAOU.RMAOU_Recommender.load_state_dict(model_dict)   # str(i - 1)

        # is it need to re-load two transformers
        RRMAOU.his_transformer.load_state_dict(his_transformer_dict)
        RRMAOU.rct_transformer.load_state_dict(rct_transformer_dict)
        RRMAOU.last_grad = last_grad
        RRMAOU.last_model = last_model
        RRMAOU.his_reservoir.pool = torch.from_numpy(np.load('ad_his_reservoir.npy'))
        RRMAOU.rct_reservoir.pool = torch.from_numpy(np.load('ad_rct_reservoir.npy'))
        RRMAOU.reg.load_state_dict(reg_dict)

        # test the model and output the performance
        #---------------first to test the model--------------------

        # d_t = torch.load("online_train1.pth")
        # d_t = (d_t["user"], d_t["item"], d_t["neg_items"])
        RRMAOU.online_update_adaptor_recommender(d_t, int(i))
        test_set = online_datasets.get_test(i+1)
        RRMAOU.online_test(test_set, i)


