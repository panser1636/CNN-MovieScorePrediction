import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import pickle

"""
数据预处理
对原始电影数据，user数据进行处理
"""


def user_data_processing():
	"""
		对原始user数据进行处理
		UserID：保持不变
		JobID：保持不变
		Gender字段：需要将‘F’和‘M’转换成0和1。
		Age字段：要转成7个连续数字0~6
		舍弃： zip-code
	"""
	print('user_data_processing....')
	user_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
	users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=user_title, engine='python')
	users = users.filter(regex='UserID|Gender|Age|JobID')
	users_orig = users.values  # a list

	gender_to_int = {'F': 0, 'M': 1}
	users['Gender'] = users['Gender'].map(gender_to_int)
	age2int = {val: ii for ii, val in enumerate(set(users['Age']))}  # Age字段：要转成7个连续数字0~6
	users['Age'] = users['Age'].map(age2int)
	print('users:', users)
	print('users_orig:', users_orig)
	return users, users_orig


def movie_data_processing(title_length=16):  # 使用的嵌入层是（N,16）
	"""
		对原始movie数据不作处理
		Genres字段：进行int映射，因为有些电影是多个Genres的组合,需要再将每个电影的Genres字段转成数字列表.
		Title字段：首先去除掉title中的year。然后将title映射成数字列表。（int映射粒度为单词而不是整个title）
		Genres和Title字段需要将长度统一，这样在神经网络中方便处理
		空白部分用‘< PAD >’对应的数字填充。
	"""
	print('movie_data_processing....')
	movies_title = ['MovieID', 'Title', 'Genres']
	movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python', encoding="unicode_escape")  # header=None表示第一行不当作标签行
	movies_orig = movies.values  # length:3883
	# title处理，首先将year过滤掉
	pattern = re.compile(r'^(.*)\((\d+)\)$')  # 数据集中的title格式 Toy Story (1995)
	title_re_year = {val: pattern.match(val).group(1) for val in set(movies['Title'])}  # group(1)第一个子模式，即是Toy Story (1995)中的Toy Story
	title_year = [pattern.match(val).group(2) for val in set(movies['Title'])]  # group(2)第二个子模式，即是Toy Story (1995)中的Toy Story
	movies['Title'] = movies['Title'].map(title_re_year)
	movies['year'] = title_year
	# title的int映射
	title_set = set()
	for val in movies['Title'].str.split():
		title_set.update(val)
	title2int = {val: ii for ii, val in enumerate(title_set)}  # length:5215
	title2int['PADDING'] = 0

	# 构建title_map，每个title映射成一个int list，然后对于长度不足16的使用pad进行补全
	title_set.add('PADDING')
	title_map = {val: [title2int[row] for row in val.split()] for val in set(movies['Title'])}
	for key in title_map.keys():
		padding_length = title_length - len(title_map[key])
		padding = [title2int['PADDING']] * padding_length
		title_map[key].extend(padding)
		# for cnt in range(title_length - len(title_map[key])):
		# 	title_map[key].insert(len(title_map[key]) + cnt, title2int['PADDING'])
	movies['Title'] = movies['Title'].map(title_map)
	print(len(movies['Title'][0]))

	# 电影类型转为数字字典
	genres_set = set()
	for val in movies['Genres'].str.split('|'):
		genres_set.update(val)
	genres2int = {val: ii+1 for ii, val in enumerate(genres_set)}  # length:19
	genres2int['PADDING'] = 0

	# 和title的处理相同，对每个电影的genres构建一个等长的int list映射
	genres_set.add('PADDING')
	genres_map = {val: [genres2int[row] for row in val.split('|')] for val in set(movies['Genres'])}
	for key in genres_map:
		padding_length = len(genres_set) - len(genres_map[key])
		padding = [genres2int['PADDING']] * padding_length
		genres_map[key].extend(padding)
		# for cnt in range(max(genres2int.values()) - len(genres_map[key])):
		# 	genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
	movies['Genres'] = movies['Genres'].map(genres_map)
	print('movie:', movies)
	print('movie_genres:', movies['Genres'])
	print('len(genres_set)', len(genres_set))
	print('len(movie_genres):', len(movies['Genres'][0]))
	print('movie_orig:', movies_orig)
	print('genres2int:', genres2int)
	print('title2int', title2int)
	print('title_set:', title_set)
	print(genres2int['PADDING'])
	return movies, movies_orig, genres2int, title_set


def rating_data_processing():
	"""
	rating数据处理，只需要将timestamps舍去，保留其他属性即可
	"""
	print('rating_data_processing....')
	ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
	ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
	ratings = ratings.filter(regex='UserID|MovieID|ratings')
	print(ratings)
	return ratings


def get_feature():
	"""
	将多个方法整合在一起，得到movie数据，user数据，rating数据。
	然后将三个table合并到一起，组成一个大table。
	最后将table切割，分别得到features 和 target（rating）
	"""
	title_length = 16
	users, users_orig = user_data_processing()
	movies, movies_orig, genres2int, title_set = movie_data_processing()
	ratings = rating_data_processing()

	# merge three tables
	data = pd.merge(pd.merge(ratings, users), movies)
	print(data)
	# split data to feature set:X and lable set:y
	target_fields = ['ratings']
	feature_pd, tragets_pd = data.drop(target_fields, axis=1), data[target_fields]
	features = feature_pd.values
	targets = tragets_pd.values

	# print(type(feature_pd))
	# print(feature_pd.head())

	# 将处理后的数据保存到本地
	with open('model/features.p', 'wb') as f:
		# ['UserID' 'MovieID' 'Gender' 'Age' 'JobID' 'Title' 'Genres']
		pickle.dump(features, f)

	with open('model/target.p', 'wb') as f:
		pickle.dump(targets, f)

	with open('model/params.p', 'wb') as f:
		pickle.dump((title_length, title_set, genres2int, features, targets, ratings, users, movies, data, movies_orig, users_orig), f)

	title_vocb_num = len(title_set)+1  # 5216
	genres_num = len(genres2int)  # 19
	movie_id_num = max(movies['MovieID'])+1  # 3953
	# print(title_vocb_num, genres_num, movie_id_num)
	f = open('model/argument.p', 'wb')
	pickle.dump((movie_id_num, title_length, title_vocb_num, genres_num), f)  #

	return features, targets


if __name__ == '__main__':
	# 用户数据处理
	# print("--------------------------------用户数据处理-------------------------------------------")
	# user_data_processing()

	# 电影数据处理
	# print("--------------------------------电影数据处理-------------------------------------------")
	# movie_data_processing()

	# 排名数据处理
	# print("--------------------------------得分数据处理-------------------------------------------")
	# rating_data_processing()

	# 排名数据处理
	print("------------------features(users and movie) & targets(rating)--------------------------------")
	get_feature()