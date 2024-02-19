import numpy as np
import pandas
import torch
from data_process.DataGenerator import ad_user_converting, ad_item_converting, get_info

'''
The new raw data is encoded and then stored into reservoir
change the row data to vector information, can be retrieved 

deal the row data [u_id, ad_id] with set_data() from DataGenerator before updating reservoir 
'''


class Data_transfer():
    def __init__(self):
        pass

    def encoder(self, sample):
        pass

    def update(
            self):  # In the process of back propagation, the recommender can be fixed and then the data encoder can be updated
        pass


class Reservious(object):
    def __init__(self, length):
        super(Reservious, self).__init__()
        self.t = 0
        self.len = length
        self.pool = np.zeros((length, 2), dtype=np.long)  # as reservoir store the data   structure: u_id  item_emb
        print("pool size:", self.pool.shape)
        self.pool_have = 0

    def updata(self, new_data):  # sampling new data to update the reservoir with replacement
        if self.t <= self.len:
            new_num = new_data.shape[0]
            max_id = min(self.len, self.pool_have + new_num)
            self.pool[self.pool_have:max_id] = new_data[:max_id - self.pool_have]
            if max_id != self.len:
                new_data = new_data[max_id - self.pool_have:]
            self.pool_have = self.pool_have + max_id
            self.t = max_id
        new_num = new_data.shape[0]
        p = self.len * 1.0 / (self.t + np.arange(new_num) + 1)
        m = np.random.rand(new_num)
        select_data = new_data[np.where(m < p)]
        for i in range(select_data.shape[0]):
            idx = np.random.randint(0, self.len, 1)
            self.pool[idx] = select_data[i]
        self.t += new_num

    def init_pool(self, new_data):
        num = new_data.shape[0]
        rand_idx = np.random.randint(0, num, self.len)
        # self.pool[:] = new_data[rand_idx]
        # self.pool[:] = new_data[-self.len:]
        # self.pool_have = self.len
        if len(new_data) < self.len:
            self.pool[:len(new_data)] = new_data
        else:
            self.pool[:] = new_data[-self.len:]
        self.pool_have = len(self.pool)
        self.t = num

    '''obtain the related information by retrieving the reservoir based on target user id'''

    def retrieve(self, u_emb, max_ret):
        # result = [self.pool[i] for i in range(self.size) if self.pool[i][0] == u_id]
        # result = self.pool[np.where(m < p)]
        result = np.empty([0, self.e])

    def store(self, path, type='his'):
        np.save(path + type + "_reservoir.npy", self.pool)


'''
first to init_pool() when offline train the model
'''


class HisReservoir(Reservious):
    def __init__(self, size, emb_dim, u_attr, i_attr):
        super(Reservious, self).__init__()
        self.t = 0
        self.len = size
        self.dim = emb_dim * (u_attr + i_attr)
        self.pool = np.zeros((size, self.dim),
                             dtype=np.float)  # as reservoir store the data   structure: u_id  item_emb
        print("pool size:", self.pool.shape)
        self.pool_have = 0
        self.u_size = emb_dim * u_attr
        self.i_size = emb_dim * i_attr

    def init_pool(self, new_data):
        num = new_data.shape[0]
        rand_idx = np.random.randint(0, num, self.len)
        # self.pool[:] = new_data[rand_idx]
        if len(new_data) <= self.len:
            self.pool[:len(new_data)] = new_data
        else:
            self.pool[:] = new_data[-self.len:]
        self.pool_have = len(self.pool)
        self.t = num

    # update the historical reservoir based on recent reservoir
    def update(self, r_res):  # update the historical reservoir based on recent reservoir
        self.pool = self.pool.cpu().numpy()
        # self.pool = np.zeros((self.len, self.dim),
        #                      dtype=np.float)
        if self.t <= self.len:
            new_num = r_res.shape[0]
            max_id = min(self.len, self.pool_have + new_num)
            self.pool[self.pool_have:max_id] = r_res[:max_id - self.pool_have]
            if max_id != self.len:
                r_res = r_res[max_id - self.pool_have:]
            self.pool_have = self.pool_have + max_id
            self.t = max_id
        new_num = r_res.shape[0]
        p = self.len * 1.0 / (self.t + np.arange(new_num) + 1)
        m = np.random.rand(new_num)
        select_data = r_res[np.where(m < p)]
        for i in range(select_data.shape[0]):
            idx = np.random.randint(0, self.len, 1)
            self.pool[idx] = select_data[i]
        self.t += new_num
        self.pool = torch.from_numpy(self.pool)

    def retrieve(self, u_emb, max_ret):
        '''
        caculate u_emb with pool based on similarity, select topk
        :param u_id: target user to retrieve
        :param max_ret: the max number of the retrieve results
        :return:{i_emb}
        pool size: [[[u_emb, i_emb]], [[u_emb, i_emb]], ...]
        '''
        # users_emb = torch.from_numpy(self.pool[:, :self.u_size])
        users_emb = self.pool[:, :self.u_size]

        result = []
        for u in u_emb:
            sim = torch.nn.functional.cosine_similarity(u, users_emb)
            topk_sim, topk_idx = torch.topk(sim, max_ret)
            sam = [self.pool[i][self.u_size:] for i in topk_idx.tolist()]
            result.append(sam[::-1])
        return result
        # for i in range(len(self.pool)):
        #     # if self.pool[i][0] == np.long(u_id):
        #     #     result = np.append(result, self.pool[i], axis=0)
        #     torch.cosine_similarity(u_emb, self.pool[i][:, :self.u_size])
        # return result[:max_ret, 1:]


# 存储的数据结构<u_id, item_emb, y> or <u_id, item_emb>
# only store the positive samples
class RecentReservoir(Reservious):
    def __init__(self, size, emb_dim, u_attr, i_attr):
        super(Reservious, self).__init__()
        self.t = 0
        self.len = size
        self.dim = emb_dim * (u_attr + i_attr)
        self.pool = np.zeros((size, self.dim),
                             dtype=np.float)  # as reservoir store the data   structure: u_id  item_emb
        print("pool size:", self.pool.shape)
        self.pool_have = 0
        self.u_size = emb_dim * u_attr

    def update(self, new_data):  # update the recent reservoir based on a fixed time new data
        self.pool = np.zeros((self.len, self.dim),
                             dtype=np.float)        # after offline trained the model, fully update the reservoir, due to the trained embedding
        if self.t <= self.len:
            new_num = new_data.shape[0]
            max_id = min(self.len, self.pool_have + new_num)
            self.pool[self.pool_have:max_id] = new_data[:max_id - self.pool_have]
            if max_id != self.len:
                new_data = new_data[max_id - self.pool_have:]
            self.pool_have = self.pool_have + max_id
            self.t = max_id
        new_num = new_data.shape[0]
        p = self.len * 1.0 / (self.t + np.arange(new_num) + 1)
        m = np.random.rand(new_num)
        select_data = new_data[np.where(m < p)]
        for i in range(select_data.shape[0]):
            idx = np.random.randint(0, self.len, 1)
            self.pool[idx] = select_data[i]
        self.t += new_num
        self.pool = torch.from_numpy(self.pool)

    def init_pool(self, new_data):
        num = new_data.shape[0]
        rand_idx = np.random.randint(0, num, self.len)
        # self.pool[:] = new_data[rand_idx]
        if len(new_data) < self.len:
            self.pool[:len(new_data)] = new_data
        else:
            self.pool[:] = new_data[-self.len:]
        self.pool_have = len(self.pool)
        self.t = num

    def retrieve(self, u_emb, max_ret):
        '''
        :param u_id: target user to retrieve
        :param max_ret: the max number of the retrieve results
        :return:
        '''
        # result = self.pool[self.pool[:, 0] == u_emb]   # np.long(u_id)
        # users_emb = torch.from_numpy(self.pool[:, :self.u_size])
        users_emb = self.pool[:, :self.u_size]

        result = []
        for u in u_emb:
            sim = torch.nn.functional.cosine_similarity(u, users_emb)
            topk_sim, topk_idx = torch.topk(sim, max_ret)
            sam = [self.pool[i][self.u_size:] for i in topk_idx.tolist()]
            result.append(sam[::-1])
        return result



# if __name__=="__main__":
#     his_r = HisReservoir()
#     rct_r = RecentReservoir()
#     his_r.init_pool()
#     rct_r.init_pool()

# deal the row data [u_id, ad_id] with set_data() from DataGenerator before updating reservoir
u_info, i_info = get_info("./datasets/AD/", 'user_profile.csv', 'ad_feature.csv')


def set_data(data: pandas.DataFrame, emb_dim, ad_item_emb, ad_user_emb, data_name="AD"):  # [u_id, ad_id]
    # set the row data to [[u_id, item_emb],[]]  ************************
    new_data = np.empty([len(data), emb_dim], dtype=np.float)

    for index, row in data.iterrows():
        if len(row) == 0:
            continue
        if data_name == "AD":
            item = i_info[i_info["adgroup_id"] == row[1]]
            item = ad_item_converting(item)

            user = u_info[u_info["userid"] == row[0]]
            if len(user) == 0:
                continue
            user = ad_user_converting(user)
        elif data_name == "adressa":
            user = torch.tensor(row['u_idx']).long()
            item = torch.tensor(row['clk_new']).long()
        else:  # yelp
            user = torch.tensor(row['user_id']).long()
            item = torch.tensor(row['item_id']).long()

        item_emb = ad_item_emb(item)
        user_emb = ad_user_emb(user)

        # u_i = np.insert(item_emb, 0, user_emb)
        u_i = torch.cat((user_emb, item_emb), dim=-1)
        new_data[index] = u_i.detach().numpy()
    return new_data

# reservoir存储的是one-hot 向量，然后检索items经过transformer然后经过item_embedding
# def set_data2(data, emb_dim, item_emb, data_name="AD"):  # [u_id, ad_id]
#     # set the row data to [[u_id, item_emb],[]]  ************************
#     new_data = np.empty([len(data), emb_dim*2], dtype=np.long)
#     for u_id, _, item, _ in enumerate(data):
#         u_i = item_emb(item)
#         u_i = torch.cat([torch.tensor([u_id]), u_i])
#         new_data = np.append(new_data, u_i, axis=0)
#     return new_data
