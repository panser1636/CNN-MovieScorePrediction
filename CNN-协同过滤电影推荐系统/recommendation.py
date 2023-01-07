import numpy as np
import tensorflow as tf

import os
import pickle
import random
import tkinter as tk


features = pickle.load(open('model/features.p', 'rb'))
target_values = pickle.load(open('model/target.p', 'rb'))
title_length, title_set, genres2int, feature, target_value, \
        ratings, users, movies, data, movies_orig, users_orig \
    = pickle.load(open('model/params.p', mode='rb'))      # feature target_value去掉s

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第五行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}
sentences_size = title_length  # 16
load_dir = './save_model/'
movie_feature_size = user_feature_size = 512
movie_matrix_path = 'movie_matrix.p'
user_matrix_path = 'user_matrix.p'


# 获取 Tensors
def get_tensors(loaded_graph):
    uid = loaded_graph.get_tensor_by_name('uid:0')
    user_gender = loaded_graph.get_tensor_by_name('user_gender:0')
    user_age = loaded_graph.get_tensor_by_name('user_age:0')
    user_job = loaded_graph.get_tensor_by_name('user_job:0')
    movie_id = loaded_graph.get_tensor_by_name('movie_id:0')
    movie_year = loaded_graph.get_tensor_by_name('movie_year:0')
    movie_categories = loaded_graph.get_tensor_by_name('movie_categories:0')
    movie_titles = loaded_graph.get_tensor_by_name('movie_titles:0')
    targets = loaded_graph.get_tensor_by_name('targets:0')
    dropout_keep_prob = loaded_graph.get_tensor_by_name('dropout_keep_prob:0')

    inference = loaded_graph.get_tensor_by_name('inference/MatMul:0')
    movie_combine_layer_flat = loaded_graph.get_tensor_by_name('movie_fc/Reshape:0')
    user_combine_layer_flat = loaded_graph.get_tensor_by_name('user_fc/Reshape:0')
    return uid, user_gender, user_age, user_job, movie_id, movie_year, movie_categories, movie_titles, targets, \
           dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat


# --------------------------------预测指定用户对指定电影的评分---------------------------------------------------------------
# 这部分就是对网络做正向传播，计算得到预测的评分
def rating_movie(user_id, movie_id_val):
    loaded_graph = tf.Graph()
    with tf.compat.v1.Session(graph=loaded_graph) as sess:
        # load save model
        loader = tf.compat.v1.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)  # 恢复模型
        # get tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_year, movie_categories, movie_titles, targets, \
        dropout_keep_prob, inference, _, __ = get_tensors(loaded_graph)
        # print(uid)
        categories = np.zeros([1, 19])
        categories[0] = movies.values[movieid2idx[movie_id_val]][2]
        titles = np.zeros([1, sentences_size])
        titles[0] = movies.values[movieid2idx[movie_id_val]][1]
        feed = {
            uid: np.reshape(users.values[user_id - 1][0], [1, 1]),
            user_gender: np.reshape(users.values[user_id - 1][1], [1, 1]),
            user_age: np.reshape(users.values[user_id - 1][2], [1, 1]),
            user_job: np.reshape(users.values[user_id - 1][3], [1, 1]),
            movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
            movie_year: np.reshape(movies.values[movieid2idx[movie_id_val]][3], [1, 1]),
            movie_categories: categories,  # x.take(6,1)
            movie_titles: titles,  # x.take(5,1)
            dropout_keep_prob: 0.5 # 1
        }
        # get prediction
        inference_val = sess.run([inference], feed)
        return (inference_val)


# -----------------------------生成movie特征矩阵，将训练好的电影特征组合成电影特征矩阵并保存到本地-----------------------------------
# 对每个电影进行正向传播
def save_movie_feature_matrix():
    loaded_graph = tf.Graph()
    movie_matrics = []
    with tf.compat.v1.Session(graph=loaded_graph) as sess:
        # load saved model
        loader = tf.compat.v1.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # get tensor from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_year,\
        movie_categories, movie_titles, targets, dropout_keep_prob, \
        _, movie_combine_layer_flat, __ = get_tensors(loaded_graph)

        for item in movies.values:
            categories = np.zeros([1, 19])
            categories[0] = item.take(2)
            titles = np.zeros([1, sentences_size])
            titles[0] = item.take(1)
            feed = {
                movie_id: np.reshape(item.take(0), [1, 1]),
                movie_year: np.reshape(item.take(3), [1, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5, 1)
                dropout_keep_prob: 0.5,
            }
            movie_representation = sess.run([
                movie_combine_layer_flat], feed)
            movie_matrics.append(movie_representation)
    movie_matrics = np.array(movie_matrics).reshape(-1, movie_feature_size)
    pickle.dump(movie_matrics, open(movie_matrix_path, 'wb'))


# 生成user特征矩阵
# 将训练好的用户特征组合成用户特征矩阵并保存到本地
# 对每个用户进行正向传播
def save_user_feature_matrix():
    loaded_graph = tf.Graph()
    users_matrix = []
    with tf.compat.v1.Session(graph=loaded_graph) as sess:
        # load saved model
        loader = tf.compat.v1.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        uid, user_gender, user_age, user_job, movie_id, movie_year,\
        movie_categories, movie_titles, targets, dropout_keep_prob, \
        _, __, user_combine_layer_flat = get_tensors(loaded_graph)

        for item in users.values:
            feed = {
                uid: np.reshape(item.take(0), [1, 1]),
                user_gender: np.reshape(item.take(1), [1, 1]),
                user_age: np.reshape(item.take(2), [1, 1]),
                user_job: np.reshape(item.take(3), [1, 1]),
                dropout_keep_prob: 0.5
            }
            user_representation = sess.run([user_combine_layer_flat], feed)
            users_matrix.append(user_representation)
    users_matrix = np.array(users_matrix).reshape(-1, user_feature_size)
    pickle.dump(users_matrix, open(user_matrix_path, 'wb'))


def load_feature_matrix(path):
    if os.path.exists(path):
      pass
    elif path == movie_matrix_path:
        save_movie_feature_matrix()
    else:
        save_user_feature_matrix()
    return pickle.load(open(path, 'rb'))


# ----------------------------------------------使用电影特征矩阵推荐同类型的电影-----------------------------------------------
# 思路是计算指定电影的特征向量与整个电影特征矩阵的余弦相似度，取相似度最大的top_k个，
# ToDo: 加入随机选择，保证每次的推荐稍微不同
def recommend_same_type_movie(movie_id, top_k=5):
    loaded_graph = tf.Graph()
    movie_matrix = load_feature_matrix(movie_matrix_path)
    movie_feature = movie_matrix[movieid2idx[movie_id]].reshape([1, movie_feature_size])  # 给定电影的representation

    with tf.compat.v1.Session(graph=loaded_graph) as sess:
        loader = tf.compat.v1.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # 计算余弦相似度
        norm_movie_matrix = tf.sqrt(tf.reduce_sum(
            tf.square(movie_matrix), axis=1, keepdims=True))  # 计算每个representation的长度 ||x||
        normalized_movie_matrix = movie_matrix / (norm_movie_matrix * norm_movie_matrix[movie_id])
        probs_similarity = tf.matmul(movie_feature, tf.transpose(normalized_movie_matrix))
        # 得到对于给定的movie id，所有电影对它的余弦相似值
        sim = probs_similarity.eval()
    # print('和电影：{} 相似的电影有：\n'.format(movies_orig[movieid2idx[movie_id]]))

    sim = np.squeeze(sim)  # 将二维sim转为一维
    res_list = np.argsort(-sim)[:top_k]  # 获取余弦相似度最大的前top k个movie信息
    results = list()
    for res in res_list:
        movie_info = movies_orig[res]
        results.append(movie_info)
        # print(movie_info)
    return results


# -----------------------------给定指定用户，推荐其喜欢的电影(基于产品的协同过滤算法)--------------------------------------------------
# 思路是使用用户特征向量与电影特征矩阵计算所有电影的评分，取评分最高的top_k个
# ToDo 加入随机选择
def recommend_your_favorite_movie(user_id, top_k):
    loaded_graph = tf.Graph()
    movie_matrix = load_feature_matrix(movie_matrix_path)
    users_matrix = load_feature_matrix(user_matrix_path)
    user_feature = users_matrix[user_id - 1].reshape([1, user_feature_size])  # 是否需要减一

    with tf.compat.v1.Session(graph=loaded_graph) as sess:
        loader = tf.compat.v1.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # 获取图中的 inference，然后用sess运行
        probs_similarity = tf.matmul(user_feature, tf.transpose(movie_matrix))
        sim = (probs_similarity.eval())
        sim = np.squeeze(sim)
        res_list = np.argsort(-sim)[:top_k]  # 获取该用户对所有电影可能评分最高的top k
        results = []
        for res in res_list:
            movie_info = movies_orig[res]
            results.append(movie_info)
            # for mov in movie_info:
            #     print(mov)
        return results


# -----------------------------------基于用户的协同过滤推荐------------------------------------------------------------------
# 看过这个电影的人还可能（喜欢）哪些电影
# 首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量
# 然后计算这几个人对所有电影的评分
# 选择每个人评分最高的电影作为推荐
# ToDo 加入随机选择


def recommend_other_favorite_movie(movie_id, top_k):
    loaded_graph = tf.Graph()
    movie_matrix = load_feature_matrix(movie_matrix_path)
    users_matrix = load_feature_matrix(user_matrix_path)
    movie_feature = (movie_matrix[movieid2idx[movie_id]]).reshape([1, movie_feature_size])
    # print('您看的电影是：{}'.format(movies_orig[movieid2idx[movie_id]]))

    with tf.compat.v1.Session(graph=loaded_graph) as sess:
        loader = tf.compat.v1.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # 计算对给定movie，所有用户对其可能的评分
        users_inference = tf.matmul(movie_feature, tf.transpose(users_matrix))
        favorite_users_id = np.argsort(users_inference.eval())[0][-top_k:]

        # print('喜欢看这个电影的人是：{}'.format(users_orig[favorite_users_id - 1]))  # user_id 处理时是否需要减一

        # print("以下是给您的推荐：")
        results = []
        for user in favorite_users_id:
            movie = recommend_your_favorite_movie(user, top_k=top_k)
            results.append(movie)
        return results, users_orig[favorite_users_id - 1]


# test every recommendation functions here
# 生成user和movie的特征矩阵，并存储到本地
# save_movie_feature_matrix()
# save_user_feature_matrix()


# 预测给定user对给定movie的评分*
user_id_test = 234
movie_id_test = 1401
print('------------------------预测给定user对给定movie的评分------------------------------------------------------------')
prediction_rating = rating_movie(user_id=user_id_test, movie_id_val=movie_id_test)
print("for user:", user_id_test, "predicting the rating for movie:", movie_id_test, "is", prediction_rating[0][0])


# 对给定的电影，推荐相同类型的其他top k 个电影
print('------------------------推荐相同类型的其他top k 个电影(基于产品的协同过滤算法)---------------------------------------------')
recommend_same_type_movie(movie_id=movie_id_test, top_k=5)


# 对给定用户，推荐其可能喜欢的top k个电影(用户特征矩阵和电影特征矩阵计算电影得分)
print('------------------------对给定用户，推荐其可能喜欢的top k个电影(计算电影对于用户的得分)--------------------------------------')
print('以下是给您的推荐：')
recommend_your_favorite_movie(user_id=user_id_test, top_k=5)


# 看过这个电影的人还可能喜欢看那些电影（基于用户的协同过滤）
print('--------------------------基于用户的协同过滤-----------------------------------------------------------------------')
recommend_other_favorite_movie(movie_id=movie_id_test, top_k=5)

# ---------------------------------------------可视化---------------------------------------------------------------------
window = tk.Tk()
window.title("电影推荐系统")

label1 = tk.Label(window, text="用户id:")
label1.grid(row=0)
label2 = tk.Label(window, text="电影id:")
label2.grid(row=1)
label3 = tk.Label(window, text="top_k:")
label3.grid(row=2)

uid = tk.Variable(window, value=int())
mid = tk.Variable(window, value=int())
top_k = tk.Variable(window, value=int())

entry_uid = tk.Entry(window, textvariable=uid)
entry_uid.grid(row=0, column=1)
entry_mid = tk.Entry(window, textvariable=mid)
entry_mid.grid(row=1, column=1)
entry_top_k = tk.Entry(window, textvariable=top_k)
entry_top_k.grid(row=2, column=1)

text = tk.Text(window)
text.grid(row=8, column=1)

button_quit = tk.Button(window, text='退出', command=window.quit)
button_quit.grid(row=9, column=0)


def clear():
    text.delete('1.0', 'end')
    return None


def rat():
    prediction_rating = rating_movie(user_id=int(uid.get()), movie_id_val=int(mid.get()))
    text.insert("insert", "##### 对用户:{} 预测电影:{} 的评分为: {} \n\n".format(uid.get(), mid.get(), prediction_rating[0][0][0]))
    return None


def rec_same():
    same_type_movie = recommend_same_type_movie(movie_id=int(mid.get()), top_k=int(top_k.get()))
    text.insert("insert", "##### 对电影:{},相同类型的电影推荐：\n".format(mid.get()))
    for i in range(int(top_k.get())):
        text.insert("insert", "{}.{}\n".format(i + 1, same_type_movie[i]))
    text.insert("insert", "\n\n")
    return None


def rec_itemCF():
    item_base = recommend_your_favorite_movie(user_id=int(uid.get()), top_k=int(top_k.get()))
    text.insert("insert", "##### 基于电影的协同过滤电影推荐：\n")
    for i in range(int(top_k.get())):
        text.insert("insert", "{}.{}\n".format(i + 1, item_base[i]))
    text.insert("insert", "\n\n")
    return None


def rec_userCF():
    user_base, favorite_user = recommend_other_favorite_movie(movie_id=int(mid.get()), top_k=int(top_k.get()))

    text.insert("insert", "#####基于用户的协同过滤电影推荐：\n")
    i = 0
    for user in favorite_user:
        text.insert("insert", "$用户：{}\n".format(user))
        for j in range(int(top_k.get())):
            text.insert("insert", "{}.{}\n".format(j + 1, user_base[i][j]))
        i += 1
    text.insert("insert", "\n\n")

    return None


button_pre_rat = tk.Button(window, text="预测评分", command=rat)
button_pre_rat.grid(row=3, column=1)

button_rec_same = tk.Button(window, text="相同类型", command=rec_same)
button_rec_same.grid(row=4, column=1)

button_rec_itemCF = tk.Button(window, text="itemCF", command=rec_itemCF)
button_rec_itemCF.grid(row=5, column=1)

button_rec_userCF = tk.Button(window, text="userCF", command=rec_userCF)
button_rec_userCF.grid(row=6, column=1)

button_clear = tk.Button(window, text="清空内容", command=clear)
button_clear.grid(row=7, column=1)

window.mainloop()