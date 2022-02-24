# -*- coding: utf-8 -*-
"""
@author: 苏铉元
"""

# 0 pycharm经验

# 0.1 运行程序的常见方式
# run/debug/局部python console运行
# run，不能局部运行只能一条龙，适合已跑通的程序
# debug，可以通过加断点的方式逐条执行；在单次debug结束前，可以局部python console运行，改变的内存会被后续未debug的程序识别使用
# 在单次debug结束后，相关内存会被清除
# 局部python console运行，只要不清除console，内存会永久保留，但可能会出有内存堆积的弊端

# 0.2 快捷键
# debug: F5     终止debug: Shift + F2     重新debug: Ctrl + F5      断点间执行debug: Shift + F9
# run: Shift + Enter      局部python console执行: Shift + Alt + E
# 全局替换: Ctrl + R        快速查找变量/文件/函数: Shift + Shift
# 打开函数帮助: 光标置于函数上 & Ctrl + Q       打开函数源码: 光标置于函数上 & Ctrl + B       打开函数简要帮助: 按住Ctrl鼠标点击
# 将当前语句上/下移动一行: Shift + Ctrl + up / down


# 1 Python基础功能

# 1.1 使用 % 命名
# 一般用于print语句中
print('我的名字叫 %s ' % '马月明')
for i in range(20, 25):
    print('我今年 %d ' % i)
    print('我的名字叫 %s , 我今年 %d ' % ('马月明', i))

# 1.2 使用{} .format
# 一般用于print语句， 不会出现与 % 冲突的情况
a = 0.1234567
b = 'sxy'
print('数字: {:.3f}, 名字: {}'.format(a, b))  # 通过 :.3f 设置3位有效数字

# 1.3 使用 + str()
# 一般用于简单的print连接
for i in range(10, 20, 5):
    print('我今年已经' + str(i) + '岁啦')

# 1.4 使用dict动态赋值
# 创建动态数组
import numpy as np
dict = {}
for i in range(1, 10):
    dict['A%d' % i] = np.arange(i)
# 遍历字典
for key in dict:
    print(str(key) + ' is ' + str(dict[key]))
for key, value in dict.items():
    print(key, ' --> ', value)

# 1.5 for循环技巧
# for a, b in zip(A, B):      从A、B顺序中依次取数，循环次数按照A、B中最少元素数目确定
A = [2, 3, 4, 7]
B = [4, 5, 6]
for a, b in zip(A, B):
    print(str(a) + ', ' + str(b))
# for i, a in enumerate(A)     i代表当前循环次数, a代表当前子元素
for i, a in enumerate(A):
    print(str(i) + ', ' + str(a))
# [a ** 2 for a in A]     一种循环遍历 + 元素处理 + 新list生成的压缩式写法
A_new = [a ** 2 for a in A]
# 一个例子
name = ['苏铉元', '马月明', '苏湘茗']
for i, value in enumerate(name):       # 通过enumerate函数，i捕获了当前序号，value捕获了name[i]
    print('第 {} 个人的名字是 {} .'.format(i + 1, value))
# break: 直接跳出所有循环
for i in range(10):
    if i > 5:
        break  # 满足i > 5则终止所有循环
    else:
        print(i)
# continue: 仅跳出某次循环
for i in range(10):
    if i % 2 == 0:
        continue  # 满足i % 2 == 0则跳过这次循环
    else:
        print(i)

#1.6 print字符串换行
print('abc\nde')  # 在字符串间添加'\n'即可实现字符串打印换行，前后不需要添加空格

# 1.7 判断数据类型
# 判断变量数据类型
A = 2
print(type(A))  # 只能打印输出
Type = str(type(A))  # 要赋值成变量, 必须通过str()处理一下

# 1.8 异常尝试语句
# (1) 试错功能
try:
	float(b)  # 若此句报错，直接执行except:后语句
except:
	print('b不能进行float转换')
	
# (2) 报错提示功能
assert a >= b, 'xxx'  # 如果条件语句不成立，则print后续字符串xxx

# (3) 类别查看功能
A = np.array([0, 1, 2])
if type(A) is numpy.ndarray:
	print('fuck')
A = [0, 1, 2]
if type(A) is list:
	print('fuck')
	


# 2 自定义函数导入 & 路径获取

# 2.1 python结构树
# >>> AAA/      #(package)
# >>>    bbb.py     #(module)
# >>>    _init_.py
# >>>    CCC/
# >>>       c1.py(def1, def2, ..., defn)    #(def：function)
# >>>       c2.py
# >>>    DDD/
# >>>    ...

# 2.2 导入自定义函数
# (1) 相对路径导入
# 在run/debug模式下，由于pycharm已经在当前py文件下创建了配置文件，相当于定位到此文件夹位置；可以使用相对路径导入
# 任何当前文件夹下的文件和数据
# (2) 绝对路径导入
# 在python console模式下，由于在python后台运行，因此未获得任何与当前py文件位置相关信息；任何文件均要使用绝对路径导入
# import sys
# sys.path.append(r'此处填写AAA文件夹路径')    # 将环境添加到自定义文件夹位置
# import AAA.CCC.c1 as FUCK     # 导入到模块级
# FUCK.def1()    # 使用自定义函
# # 或者
# sys.path.append(r'此处填c1.py路径')    # 直接将.py添加到环境
# import c1
# c1.def1()     #使用自定义函数

# 2.3 重新加载模块 & 获取路径
# 导入自定义函数后，如果更改函数代码，需要重新加载该自定义模块；
# python为了节省内存，不会二次加载模已导入的模块，因此为了导入修改后的模块，需要我们使用另外的指令
import importlib
importlib.reload('修改后的模块')      # 二次加载了更改的模块
import os
os.chdir('需要转换到的新路径')       # 将当前路径转换到新文件夹下


# 3 nd-array相关

import numpy as np

# 3.1 nd-array & matrix
# nd-array为python特有数组，具有更强通用性，允许多层嵌套；基本覆盖matrix全部功能
A = [1, 2, 3, 4, 5, 6]
B = np.array(A).reshape(3, 2)
C = B.T     # 数组转置
# matrix更类似matlab中的矩阵，必须为二维，支持通过 ";" 分行
A = np.matrix('1, 2; 3, 4; 5, 6')   # 必须加引号
A = np.mat('1; 2; 3')

# 3.2 nd-array的创建
np.random.seed(1)    # 每次生成固定随机序列；单次有效； 相同随机数函数有效
np.random.rand(2, 4)  # 生成2*4的[0, 1]间随机数
np.random.seed(1)  # 在相同的seed下，生成的随机数序列恒定；每次生成随机数前均需要运行，否则无效
np.random.rand(8).reshape(2, 4)
np.random.randn(2, 4)   # 生成标准正态随机分布
np.array([1,2,3])   # 普通转化list为nd-array
np.empty((4,2))     # 生成4*2的全0数组
np.zeros((4,2))     # 生成4*2列全0数组
np.ones((4,2))      # 生成全1数组
np.arange(0,25,5).reshape(1,5)      # 生成以5为步长的，从0到25(不含25)的数组，并重组成1行5列
np.linspace(1, 10, 10)      # 生成1-10总数为10的等差序列
A = np.arange(1, 10)
np.random.shuffle(A)  # 按一定随机方式打乱序列，A可以是nd_array或list
print('A = ' + str(A))

# 3.3 nd-array基本操作
np.sum(np.random.rand(2, 4), axis=0)  # axis=0表示沿着横轴执行操作
np.mean(np.random.rand(2, 4), axis=1)  # axis=1表示沿着纵轴执行操作；具体操作视函数功能而定
A = np.random.randn(5, 2)
B = np.arange(1, 11).reshape(5, 2)
print(np.r_[A, B])  # 按行拼接
print(np.c_[A, B])  # 按列拼接
print(np.vstack((A, B, A)))  # 垂直操作(vertical)
print(np.hstack((A, B, A)))  # 水平操作(horizontal)
np.where(B > 5, B, 100)  # np.where(condition, A, B), 对于条件condition。满足的执行A, 不满足的执行B
np.where(B > 5)  # np.where(condition)，返回满足条件的索引
B_1 = B[B[:, 1] > 5]  # 分为两部操作：首先内层找出B的第3列向量中>5的行数；将符合标准的行从B取出来
B_2 = B[:, B[1, :] == 3]  # 将B矩阵中第2行==3的列取出
# 双向筛选
C = np.logical_and(B[:, 1] > 5, B[:, 1] < 10)  # 必须使用logical_and(condition_1, condition_2)来筛选双向数据，否则报错
B_3 = B[C, :]  # 得到最终的筛选结果
# 条件筛选与反向选择
A = np.arange(0, 20).reshape(-1, 2)
A_new = A[~(A[:, 0] == 2)]  # ~[True, False, False] = [False, True, True], '~'意味着取反集
A_new = A[A[:, 0] != 2]  # 与上者相同

# 3.5 探索数组中某数是否存在
A = np.arange(0, 10).reshape(-1, 1)
Bool = 3 in A  # 如果3存在，则返回True；不存在则返回False

# 3.6 list/nd_array添加&删除
# list添加
list1 = ['a', 'b', 'c'] 
list1.append('d')  # 直接在尾部添加
list1.insert(0, 'first')  # 在指定位置添加
list2 = ['e', 'f']
list3 = list1 + list2  # 拼接两个list
# list删除
list1 = ['a', 'b', 'c', 'd']
del list1[0]  # 按照索引删除list元素
list1.remove('b')  # 按照元素值删除list元素
list1.pop()  # 删除最后一个元素
# 消除list中的重复项
List_union = list(set(List))
# array删除
A = np.arange(0, 20).reshape(-1, 4) # 一个5行4列数组
np.delete(arr=A, obj=1, axis=1)  # 对数组A, 删除第二列

# 3.7 array随机抽样与list交、并、补集计算
# 随机抽样
A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
B = np.random.choice(A, 4, replace=False)  # 在A中不放回的抽取4个元素
# 交集运算
set(A) & set(B)
set(A).intersection(B)
# 并集运算
set(A) | set(B)
set(A).union(B)
# 差集运算
set(A) - set(B)
& set(A).difference(B)

	

# 4 数据类型与数据规模

import numpy as np
import pandas as pd

# 4.1 数据规模获取
A_1 = [[1, 2, 3, 4], [2, 3, 3]]  # list
print('len(A_1) = ' + str(int(len(A_1))))  # 获取list长度，如果list为多层嵌套，之获取最高一级元素个数
print('np.shape(A_1) = ' + str(np.shape(A_1)))  # np.shape(list)可以，list.shape不可行
A_2 = ((2, 3), (4, 5), (6, 7))  # tuple
print('len(A_2) = ' + str(int(len(A_2))))  # 同list类似仅获取tuple最外层个数
print('np.shape(A_2) = ' + str(np.shape(A_2)))  # tuple.shape同样不可行
A_3 = {'a': 1, 'b': 2, 'c': 3}  # dict
print('len(A_3) = ' + str(int(len(A_3))))  # len()可以获取字典key个数，np.shape()和.shape均不可
A_4 = np.arange(1, 13).reshape(4, 3)  # nd_array
print('len(A_4) = ' + str(int(len(A_4))))  # len(nd_array)仅能获取行数，不推荐
print('np.shape(A_4) = ' + str(np.shape(A_4)))  # 可行
print('A_4.shape = ' + str(A_4.shape))  # 可行
A_5 = pd.DataFrame(A_4)  # DataFrame
print('len(A_5) = ' + str(int(len(A_5))))  # dataFrame在数据规模完全和Nd_array保持一致
print('np.shape(A_5) = ' + str(np.shape(A_5)))  # 可行
print('A_5.shape = ' + str(A_5.shape))  # 可行

# 4.2 数据类型获取
type(A_1)  # type()获取数据类型(list\tuple)，list-dataFrame均适用
A_4.astype('int')  # 转换内部元素格式(float\int)，仅适用于nd_array和DataFrame
A_5 = A_5.astype('float64')
print(A_5.dtypes)  # .dtypes输出每列元素的数据格式，仅适用于pandas数据类型！！
A_6 = np.array([[1., 2, 3, 4], [5, 6, 7, 2.], [8, 9, 10, 11]]).reshape(3, 4)
A_6 = pd.DataFrame(A_6)
print(A_6.dtypes)  # 混合str非数类型和float/int数据类型，dytpes结果为object

# 4.3 pandas中的数据类型转换
# 由于pandas中经常包含int/str/float等各类数据格式，不胫骨处理可能影响运算
# 如果处理数据类型为series单列数据
S = pd.Series([1, 2, 3, 'd', 5])
S_1 = pd.to_numeric(S)  # 直接进行数据转换会报错，因为str类型无法转为数据类型
# errors默认为'raise'，即提示报错；设置errors值为'coerce'，对于s不能转为数据的类型会转为NAN
S_2 = pd.to_numeric(S, errors='coerce')  # 可以通过此方法消除非数的行/列
S_3 = pd.to_numeric(S, errors='ignore')  # 设置为'ignore'类型，则忽视该元素
# 在nd_array变换中使用数据类型转换
# 对于某些DataFrame,其value中的开头存在str，整个value的数据格式会被统一为object格式，这样取其余数据进行函数计算会出现非数无法计算的情况
DF = pd.DataFrame(data=['time', 2, 3, 4, 5, 7, 8, 9])
X = np.array(DF.iloc[1:, :],dtype='float').reshape(-1, 1)  #这样就保证了剩余数据还原为'float'数据格式
# https://www.cnblogs.com/xitingxie/p/8426340.html pandas数据格式转换参考博客


# 5 Pandas数据类型基本操作

import pandas as pd
import numpy as np

# 5.1 读取和写入文件
df = pd.read_excel('input_path/file.xlsx', sheetname='sheet_x')  # 将path.xlsx文件下sheet_x表格中的数据读取进来
# 多次写入文件
Writer = pd.ExcelWriter('output_path/file.xlsx')  # 建立ExcelWriter文件缓存
df1.to_excel(Writer, sheetname='sheet_1')  # 写入sheet_1表格
df2.to_excel(Writer, sheetname='sheet_2')  # 写入sheet_2表格
Writer.save()  # 最终保存
# 保存DataFrame中包含中文，如何不乱码？
df.to_csv(path_file, encoding='utf_8_sig')
# 读取文件夹下所有文件
import os
os.chdir(path_folder)  # path_folder是文件夹绝对路径, 结尾不包含'/'
file_list = os.listdir()  # 将该文件夹下的所有文件名存入一个列表

# 5.2 生成基本数据类型

# Series: 可以视为一个以index为键值，内容为value的长字典
s_1 = pd.Series({'a': 1, 'b': 2, 'c': 3})  # 由字典生成Series
s_2 = pd.Series(np.arange(1, 10), index=np.arange(2, 11))  # 由numpy_array生成Series
# dataframe：可以视为多个Series的组合，内容包含了多种数据类型
df_1 = pd.DataFrame({'state': ['tom', 'peter', 'nina'],
                  'kimmy': [1, 2, 3],
                  'tinna': [1, 2, 2]})  # 由字典生成dataframe
df_2 = pd.DataFrame(np.random.randn(3, 4), index=['a', 'b', 'c'], columns=['d', 'e', 'f', 'g'])  # 由nd_array生成
# 转化数据类型为ndarray
X = np.array(df_1, dtype='float64')  # 使用dtype属性控制nd_array数据格式
# 按照时间=刻首末值生成某一范围的时间序列
time = pd.date_range(start, end, freq, periods)
# start/end格式很宽松 '2010-10-11'/'10/11/2010'均可识别
# 但需要注意，periods不可和end同时使用，因为periods设定了从start开始持续的周期长度，会与end冲突
# 详细说明https://blog.csdn.net/bqw18744018044/article/details/80920356

# 5.3 元素查找
# 使用行/列索引查找
print(df_2['d'])  # 只能用于查找列名，index索引无效；一此仅能返回单列
print(df_2[0: 2])  # 只能用来搜索行范围；列索引、单独行索引无效 --> df_2[2]报错
# 使用loc行/列索引返回多列
print(df_2.loc['a'])  # df.loc[行， 列]；可以仅有行名，不能仅有列名
print(df_2.loc[['a', 'b'], ['d', 'e']])  # 返回index为a, b; columns为d, e的dataframe
print(df_2.loc[:, 'd'])  # index可以使用:代全体，不可以使用1:3具体切片
# 使用iloc区块定位
print(df_2.iloc[0:2, 1:3])  # 返回0、1行；1、2列的dataframe
print(df_2.iloc[:, [1, 2]])  # 返回1、2列的dataframe
# 条件筛选
# df中包含['A', 'B', 'C', 'D']4列
vector = df['A'] > 22  # 返回一个布尔(True, False)向量，用于筛选A列数据>22的行数
df['B'][vector]  # 按A的筛选条件取B列的元素
df['B'][df['A'] > 22]  # 与上个语句等效
df[vector]  # 按A的而筛选条件选取整个列表
df[df.iloc[:, 1] > 22]  # 诸如.iloc/.loc等语句也完全适用
# 注意条件行数和被筛选行数保持一致
df.iloc[10:, :][df.iloc[10:, 0] > 22]  # 条件判断从10行开始，被筛选对象也要从10行开始
# 双边筛选，不同于numpy的np.logic_and()函数，DataFrame支持用&进行双边筛选
DF = pd.DataFrame(data=np.arange(0, 20).reshape(-1, 2))
DF_taget = DF[(DF.iloc[:, 0] > 4) & (DF.iloc[:, 1]< 15)]  # 不同于nd_array，其返回结果是一串bool值，因此不同利用.iloc对某列再切片
# 判断DataFrame是否为空
df.empty  # 绝对不能写empty()！, 否则报错


# 5.4 缺失值处理
# 当pandas数据中出现np.NAN数据类型，即为缺失值
df_3 = pd.DataFrame(np.array([[1, 2, 3], [4, 'r', np.nan]]), index=['a', 'b'], columns=['c', 'd', 'e'])
df_3.fillna(2, inplace=True)  # 使用2来填充df_3中的nan, 并且更新df_3
df_3.dropna(axis=1, how='any')  # 消除任何位置带有nan的纵轴，不更新df_3
df_3.drop('a')  # 删除a行
df_3.drop(df_3.columns[[1, 2]], axis=1)  # 删除1、2列, 必须指明axis=1
print('columns_name =' + str(df_3.columns))  # 获取列名
print('index_name = ' + str(df_3.index))  # 获取index名
# 批量替换元素
A = [1, 2, 3, 4, 'a', 3, 4, 4]
B = A.replace('a', 2) # 即可完成批量替换

# 5.5 合并
# Series和DataFrame是可以进行合并的；对于Series来说，列名为name；对于DataFrame来说，列名为columns
# 合并可以采用两种方式，一是直接赋值；二是使用pd.concat()
# 5.5.1 直接赋值方法
DF = pd.DataFrame(data=np.arange(0, 10).reshape(5, 2), columns=['a', 'b'])
S = DF.iloc[:, 0]
DF['c'] = S  # 通过赋值的方法给DataFrame添加一列
# 5.5.2 通过concat方法
DF = pd.DataFrame(data=np.arange(0, 10).reshape(5, 2), columns=['a', 'b'])
S = DF.iloc[:, 0]
S = S.name('c')  # 拼接dataFrame时候必须对所有列进行命名
DF_new = pd.concat([DF, S], axis=1)  # axis=1代表columns间拼接

# 来源： https://blog.csdn.net/tz_zs/article/details/81238085

# 5.6 模糊首值拼接DataFame
# 需要拼接一系列DataFrame，需要面临的问题是对首值的处理，可以选择循环判断的方式，也可以采用模糊首值的方式
# 模糊首值，只有DataFrame可以这样操作，Series不能
DF = pd.DataFrame()
df = pd.DataFrame(data=np.arange(0, 20).reshape(-1, 2), index=pd.date_range('2018', periods=10, freq='D'))
DF = pd.concat((DF, df))  # 由此解决了拼接首值的问题

# 5.7 对Series和DataFrame的重命名
# Series对象
S = pd.Series(data=np.arange(0, 10).reshape(10, 1), index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name='New')  # 赋予列名'New'
S.name = 'A_new_name'  # 进行重命名
# DataFrame对象
DF = pd.DataFrame(data=np.arange(0, 10).reshape(5, 2), index=pd.date_range('2018', periods=5, freq='D', columns=['a', 'b']))
DF.rename({'a': 'aa'}, inplace=True)  # 对于DataFrame的某一列进行变量名替换的方法  
DF.columns = ['c', 'd']  # 对全体列名进行替换

# 5.8 DataFrame数据统计基本操作合集
# DataFrame基本数据属性统计
DataFrame.describe()  # 对DataFrame各列数据点数、最大值、最小值等属性进行统计
# DataFrame乱序抽取数据
DataFrame_localshuffle = DataFrame.sample(frac=0.2, random_state=1)  # 从原始DataFrame中随机抽取20%数据，random_state是随机种子代号


# 6 pandas中的时间序列处理

# 6.1 个体对象3类基本时间格式
# 对于个体数据而言，pandas提供了3种数据格式, Timestamp, Period, Timedelta
Timestamp_1 = pd.Timestamp('2018-10-11 23:00:12')  # Timestamp表示时刻点，这种数据格式最为常见
Timestamp_2 = pd.Timestamp('2018-10-11 24:00:23') 
Datetime = pd.datetime(2018, 3, 12, 12, 00, 00)  # Datetime也表示时刻点，它拥有格式化字符串到某个时间上的能力
Datetime = pd.datetime.strptime('2018-3-12 12:00:00', '%Y-%m-%d %H:%M:%S')  # 将字符串格式化为制定datetime格式 
String = pd.datetime.strftime(Datetime, '%Y-%m-%d')  # 将datetime数据格式转化成制定的字符串格式, 本例转化到天(day)为止
Period = Timestamp_1.to_period('D')  # 将Timestamp转换到天为止, ‘2018-10-11'
Timedelta = Timestamp_2 - Timestamp_1  # 两个时刻相减，格式自动转化为Timedalta数据格式， 用来表示时间间隔
TImedelta_day = Timedelta.days  # 通过weeks、days、seconds、milliseconds等属性可以将间隔筛选到某一个间隔尺度上

# 6.2 序列对象基本数据格式
# 上述个体3类数据格式，其序列格式为DatetimeIndex, PeriodtimeIndex, TimedeltaIndex
# 以'2018-10-11 23:00'为开头，'2019-11-22 23:00'的开区间为结尾，以分钟('M')为频率生成一组datetimeIndex时间序列
DatetimeIndex = pd.date_range('2018-10-11 23:00:00', '2019-11-22 23:00:00', freq='M')  # 序列可以作为DataFrame、Series索引
# PeriodtimeIndex数据格式可以用于时刻相减，换算到某单位尺度(1day\24hours\), 比DatetimeIndex.day更强大，推荐使用
PeriodtimeIndex = DatetimeIndex.to_period('D')  # 将序列转化到'天'来显示

# 6.3 常见的时间数据操作
DatetimeIndex_2018 = DatetimeIndex['2018']  # DatetimeIndex格式支持使用'2018'\'2018-10'等时间尺度来进行切片筛选
Data = np.arange(0, 20).reshape(-1, 2)  # 生成10*2数据
Index = pd.date_range('2018-10-11', period=10, freq='D')  # 生成10个对应时间索引
Index_list = Index.tolist()  # 将DatetimeIndex转化成list也可以DataFrame赋索引；但不能转化成nd_array，会破坏其时间格式！
Name = ['fuck', 'you']
DF = pd.DataFrame(data=Data, index=Index, columns=Name)  # 生成了一个新的dataFrame
DF_resample = DF.resample('1s').mean()  # 按秒进行降采样，采用mean求均值的方式

# 6.4 DataFrame和csv的交互
# python与.csv交互可以获得更快的读取速度
f = open(File_input)  # 开启csv文件的标准格式
Data = pd.read_csv(f, index_col=0)  # 将excel第0列作为DataFrame索引
f.close()
Data.to_csv(File_output, index=True)  # 将dataFrame存为csv，索引也要输出


# 7 二次加载模块与路径转换

# 7.1 二次加载

# 在python中，import一些自定义模块；如果导入后修改了自定义模块，再次import，这是python为了节省内存设置的规则
# 一次import，永久存入内存
import importlib
importlib.reload(module)  # 执行该语句才可以实现模块二次加载
# 如果当前模块通过from model import *的形式导入，重导入需要先import module, 再imortlib.reload(module)，再from module import * 方可成功

# 7.2 转换当前路径

import os
os.chdir(Path)  # 将路径Path添加当前路径，系统再执行相应操作，即保存在路径下

# 7.3 添加模块路径

import sys
sys.path.append(self_module_path)  # 将自定义模块路径位置加入，才可以正常导入自定义模块

# 7.4 相对路径文件
# 读取本地文件时，如果不设置绝对路径，所有文件读入、保存都必须在.py同一个文件夹下面进行


# 8 pytorch相关
import torch
import torch.nn as nn
import numpy as np

# 8.1 创建pytorch
# 使用函数创建--基本和numpy语法一致
a = torch.eye(10)
a = torch.arange(0, 10).reshape(-1, 2)
a = torch.linspace(0, 10, 10).reshape(-1, 2)
a = torch.ones((1, 10, 10))  # 维度非常灵活
# 使用list&类型指令建立
b = torch.LongTensor([1, 2, 3])  # float32
bb = torch.IntTensor(2, 4).zero_()  # Int32
# 注意在使用pytorch深度学习层的时候，要保证层参数的数据格式和输入数据格式保持一致。以下提供一个统一模型、数据格式的方法
# https://stackoverflow.com/questions/49407303/runtimeerror-expected-object-of-type-torch-doubletensor-but-found-type-torch-fl

# 8.2 数据类型&数据规整
# 参考网址: https://ptorch.com/news/71.html
# 数据类型转换
c = torch.arange(0, 20)
c = c.int()  # 转化为int32
cc = c.type(torch.LongTensor)  # 转化为Int64
ccc = c.type(torch.DoubleTensor)  # 转化为float64
# 数据size规整
c_1 = c.reshape(1, 2, 10)  # .reshape()指令和.view()具有完全相同的作用
c_2 = c.view(1, 2, 10)
print(c_1.shape)
print(c_2.shape)
print(c_1.size)

# 8.3 数据切分&数据拼接&数据索引
# 数据拼接
d_1 = torch.arange(0, 10).reshape(-1, 2)
d_2 = torch.arange(0, 10).reshape(-1, 2)
d_3 = torch.cat((d_1, d_2, d_1), dim=0)  # .cat()指令和.stack()指令效果基本一致
d_4 = torch.stack((d_1, d_2, d_1), dim=1)  # numpy中的.r_[]不能再使用
# 数据切分
d_1 = torch.arange(0, 20).reshape(-1, 4)
d_2 = torch.split(d_1, 1, dim=0)  # 沿0维每1行切一次
d_3 = torch.chunk(d_1, chunks=5, dim=0)  # 沿0维一共切成chunks块
# 数据索引
# torch.index_select(input, dim, index)
d_1 = torch.randn(5, 4)
indices = torch.LongTensor([0, 2])  # index必须是torch形式，而不能是list格式
d_2 = torch.index_select(d_1, dim=0, index=indices)  # 按0维(行)取数，取出0\2列矩阵
# data[row, column]直接索引
d_1 = torch.arange(0, 20).reshape(-1, 2)
d_2 = d_1[:, [0, 1]]
# 查找索引【利用函数】     
tensor = torch.arange(0, 20).reshape(-1, 4)
number = 3  # 查找索引时，目标数值可以是float/int等元素类型，也可以是torch.tensor(3)等张量类型
indices = torch.gt(tensor, number)  # 获取tensor中大于number的位置，注意indices不直接定位位置，而是给出和tensor等维的0/1值
indices = torch.lt(tensor, number)  # 获取tensor中小于number的位置
indices = torch.eq(tensor, number)  # 获取tensor中等于number的位置
indices = torch.nonzero(tensor)  # 获取tensor中非0的位置  
indices = torch.ne(tensor, number)  # 获取tensor中不等于number的位置
number = tensor[indices]  # 利用得到的索引可以反向获得torch中的目标值
# 查找索引【直接法】
# 上述利用函数的查找索引方法理论上都可以通过直接查找来实现
indices = tensor > number 
indices != number  # 类似如此不再一一列举
number = tensor[indices]
indices = tensor[:, 0] < number  # 指定某些维度进行筛选也是完全可以

# 8.4 小技巧
# 上/下三角矩阵
down_3 = np.triu(np.ones((10, 10)), k=1)  # 生成一个下三角矩阵
up_3 = np.tril(np.ones((10, 10)), k=1)  # 生成一个上三角矩阵
up_3 = torch.from_numpy(down_3) == 0  # 生成相反矩阵(上三角矩阵)

# 数据压缩
xx.squeeze()  # 不指定维度，会自动将变量xx中元素为1的维度压缩

# 8.5 GPU相关
# 1)查看GPU状态
print(torch.cuda.is_available())  # GPU是否可用
print(torch.cuda.device_count())  # GPU个数
print(torch.cuda.get_device_name(0))  # 第0个GPU名称
print(torch.cuda.current_device())  # 当前GPU序号
# 2)查看GPU使用情况
# 打开windows控制台，输入nvidia-smi，可以查看GPU状态
# 3)转移张量到指定设备
x = torch.ones(10, 1)
print(x.device)  # cpu, 默认存储在cpu
y = x.cuda(device=0)  # 转移到制定序号的gpu上，本案例gpu只有1个，因此设定device=0
print(y.device)  # cuda:0
# 4)创建张量到指令设备
device_t = torch.device('cuda:0')  # 用专有格式定义设备
y = torch.ones((10, 1), device=device_t)
print(y)  # device='cuda:0'
## 5)转移模型到指定设备
net = torch.nn.Sequential(torch.nn.Linear(10, 1))
net = net.to(device_t)
print(next(net.parameters()).device)  # cuda:0
## 6)查看显存占用情况
torch.cuda.memory_allocated(0) / 1048576)  # 字节换算为MB

# 8.6 查看网络模型参数
for name, parameter in model.named_parameters():
	print(str(name) + ': ' + str(parameter.size()))
# 查看模型参数总量
print(sum(params.numel() for params in model.parameters()))

# 8.7 Dataset、DataLoader、Sampler
# Dataset、DataLoader、Sampler是pytorch3种典型的pytorch官方设定，帮助用户按照一定的需求，从原始数据中生成batch数据

# (1) Dataset：有3种官方函数,  def __init__(self, args)、def __getitem__(self, idx)、def __len__(self)
class MyDataset(Dataset):
    # 创建一个可迭代的Dataset类
    def __init__(self, datax, datay):
        self.data = datax
        self.label = datay
    # 定义可在DataLoader中被迭代执行的函数, 作用主要是对原始数据进行必要的预处理, 非必选
    # 这个函数容易让人迷惑，因为用户不知道idx这个变量是哪里来的。
    # 实际上，这个函数需要和DataLoader配合使用，这个函数会在DataLoader中被迭代执行，这里只要保证其预设格式(如下所示)，就可以和DataLoaderw无缝衔接
    def __getitem__(self, idx):
        data_temp = self.data[idx]
	label = self.label[idx]
        # 预处理
	data_process = f_process(data_temp)  # f_process代指所有用户所需的预处理操作
	return data_process, label
    # 与DataLoader配合的官方内置函数, 决定了在DataLoader内被迭代调用的最大次数
    def __len__(self):
        return len(self.data)

# (2) DataLoader：内部运行主要分为3个步骤
# 1、按照Sampler定义的规则生成取数的序号顺序, 2、按照序号迭代地执行MyDataset中的getitem函数，并保存结果，3、如果设置了batchsize，则将结果按照batchsize数量一堆一堆打包
dataloader = DataLoader(dataset=dataset, batch_size=batchsize, sampler=sampler)  # 返回的dataloader是一个可迭代的容器, 其迭代次数取决于dataset样本总数和预设的batchsize
# 获取dataloader内容的途径1：按批获取特征数据和标签
for batchdata in dataloader:
    features, labels = batchdata[0], batchdata[1]
# 获取dataloader内容的途径2：直接合并取出所有样本
feature_all, label_all = dataloader.__iter__().next()

# (3) Sampler: 生成样本序号次序的方法。本质上，其支持自定义和官方设定两种方式。
# 1、官方设定
SequentialSampler, RandomSampler, WeightedSampler, SubsetRandomSampler
sampler = torch.utils.data.sampler.SequentialSampler(dataset)
_ = DataLoader(dataset, sampler=sampler)
# 2、自定义
from torch.utils.data.sampler import Sampler
class MySampler(Sampler):
    # 初始化
    def __init__(self, dataset):
        self.dataset = dataset
    # 生成一个可迭代的待采样序号全集--注意！是全集，不是一个数
    def __iter__(self):
        index = range(len(self.dataset))  # index需要是单层list，是DataLoader迭代采样的全集参照
        return iter(index)

    def __len__(self):
        return len(self.dataset)  # 决定了迭代的最大次数
# 采样的实际次数，由len(dataset)真实值、def__len__(self)设定值、以及def__iter__(self)的iter(index)长度，三者的最小值决定！！


# 9 matplotlib小技巧

# 9.1 水平、垂直线
import matplotlib.pyplot as plt
plt.vlines(value, ymin, ymax)  # 绘制垂直线
plt.hlines(value, xmin, xmax)  # 绘制水平线

# 9.2 为图例增加标签
x = [i for i in range(10)]  
y = np.arange(0, 10).tolist()  # 实质上，array格式也可以进行后续操作
for a, b in zip(x, y):
	plt.text(a, b + 0.001, '%.2f' % b, ha = 'center', va = 'bottom', fontsize=10)  # 必须对坐标a, b逐点绘制；绘制字符串为2位有效数字的b

# 9.3 替换字符标注
plt.xticks([0, 1, 2], ['循环泵', '冷备泵', '过滤器'], fontsize=20)  # 将数字标签替换为字符标签

# 9.4 坐标轴相关设置
# (1) 坐标轴线宽设置
ax = plt.gca()
ax.spines['bottom'].set_linewidth(5)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(5)  # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(5)  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(5)  # 设置上部坐标轴的粗细

# (2) 坐标轴刻度长度、宽度设置
plt.tick_params(length=10, width=4)  # 坐标轴刻度长度、宽度设置
# 取消坐标轴刻度值
plt.xticks([])
plt.yticks([])

# (3) 图例字体设置
plt.rc('font', family='Times New Roman')
# 粗体设置
font = {'fontsize': 50, 'weight': 'bold'}
plt.ylabel('RUL', fontdict=font)

# (4) 设置线型标志
# 设置标志类型，并设置标志大小
plt.plot(index, pre, lw=8, label='Raw Prediction', marker='o', markersize=18)


# 10 深度学习相关--基于pytorch库


# 10.1 常用代码整理网址
https://cloud.tencent.com/developer/article/1438866

# 10.1 分类问题-onehot编码
# pytorch中onehot编码思路：
# 1) 给定待编码矩阵Label, 其包含N个样本, 标签所有种类为n_classes个。
# 2) 生成一个N*n_classes的全零矩阵, 将代编码矩阵Label中各元素值视为index, 以此把全零矩阵标号为1
# 3) 输出则为onehot编码矩阵
Tensor = torch.LongTensor([1, 2, 3, 1, 1, 1, 2]).reshape(-1, 1)  # 给定的待编码矩阵
num_classes = 10  # 假设共包含10类标签
N = Tensor.size(0)  # 提取待编码矩阵当前包含的样本个数
onehot = torch.zeros(N, num_classes).long()  # 生成全零矩阵
# dim指，在全零矩阵dim维上填1; index指, 填1的具体位置; src指生成1, 实际上根据需求使用scr生成2、3、4...都可以
onehot.scatter(dim=1, index=Tensor, src=torch.ones(N, num_classes).long())  

# 10.2 固定参数初始化
# pytorch中使用torch.manual_seed(seed_value), 进行随机数初始化:
# 运行一次torch.manual_seed(seed_value), 其后生成的随机数变量集合[a1, b1, c2, d2]彼此都不同；
# 再次运行torch.manual_seed(seed_value), 再次生成的集合[a2, b2, c2, d2]和上一次呈一一对应关系, 即a1=a2,..,d1=d2; 但a1 != b1 !=...d1.


# 11 log日志相关
https://www.cnblogs.com/yyds/p/6901864.html

# 11.1 日志的作用
# 仅讨论直接相关功能：一方面，日志可以和print一样实时在console显示出来; 另一方面，日志可以保存到本地，不用单独用excel进行程序信息记录，便于回溯。

# 11.2 本地保存日志
import logging
# 配置日志输出形式
logging.basicConfig(format='%(asctime)s - : %(message)s',
                    level=logging.INFO,
                    filename='XXX.log')  # format决定一条日志内容, level决定日志在console的显示级别，filename决定日志本地文件的名称
# 记录一条log
logging.info('xxx')  # 以INFO级别，在'XXX.log'本地文件中记录一条log('xxx')
# 如果设置了本地保存log（在logging.basicConfig配置了filename属性），那么仅通过logging命令无法将日志打印到console

# 11.3 实时打印日志
# 在11.2基础上，既想保存日志到本地又想实时打印日志到console，需要增加如下代码
logger = logging.getLogger('test')  # 'test'名称无所谓，任意即可
handler = logging.StreamHandler()  # 类似一个抽取器, 从本地文件中读取日志信息
formatter = logging.Formatter('%(asctime)s - : %(message)s')  # 设置抽取出的重整日志格式, 相当于二次过滤
handler.setFormatter(formatter)  # 为抽取器添加格式
handler.setLevel(logging.INFO)  # 为抽取器添加显示级别
logger.addHandler(handler)  # 添加抽取器

# 11.4 案例--本地保存+实时打印
import logging
logging.basicConfig(format='%(asctime)s - : %(message)s',
                    level=logging.INFO,
                    filename='XXX.log')
logger = logging.getLogger('test')
handler = logging.StreamHandler() 
formatter = logging.Formatter('%(asctime)s - : %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
# 完成上述配置后, 仅保存本地、不打印到console，用如下指令:
logging.info('仅保存本地')  # .info是指该条日志生成的级别为INFO，类似的级别还可以是.debug()/.warning()/.error()/.critical()
# 既保存本地又打印到console
logger.info('保存+打印')

# 11.5 注意
# (1) 如果要重新定义一个新的.log文件，需要重启一次console，在已有console中无法创建新.log文件
# (2) 如果不该写.log命名，多次运行console，记录会被叠加续写。




