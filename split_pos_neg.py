# coding:utf-8
# coding=gbk
import numpy as np
import pandas as pd
df = pd.read_excel('./Platform_information/Douban_information/Douban_reviews.xlsx')

def xlsx_long():
    df1 = pd.read_excel('./Platform_information/Douban_information/Douban_reviews.xlsx', sheet_name='Sheet1', usecols=[0])
    data = df1.values
    n = len(data)
    return n

with open('./Platform_information/Pos_Neg/Pos.txt','w',encoding='utf-8') as f:
    for i in range(xlsx_long()):
        if df.loc[i, 'label'] == 1:
            content = df.loc[i, 'Reviews']
            f.write(content+'\n')
        else:
            continue

with open('./Platform_information/Pos_Neg/Neg.txt','w',encoding='utf-8') as d:
    for i in range(xlsx_long()):
        if df.loc[i, 'label'] == 0:
            content1 = df.loc[i, 'Reviews']
            d.write(content1 + '\n')
        else:
            continue
# list1 =[]
list2 =[]
# with open("Pos.txt", "r",encoding='utf-8') as c:
#     for line1 in c.readlines():
#         # print(line1)
#         line = line1.strip('\n')  # Remove the line break from each element in the list
#         list1.append(line)
with open("./Platform_information/Pos_Neg/Neg.txt", "r",encoding='utf-8') as c:
    for line2 in c.readlines():
        # print(line1)
        line = line2.strip('\n')  # Remove the line break from each element in the list
        list2.append(line)


ratio_train = 0.8 # training sets
ratio_val = 0.2 # Test set ratio
assert(ratio_train + ratio_val) == 1.0
np.random.shuffle(list2)
cnt_val = round(len(list2) * ratio_val ,0)
cnt_train = len(list2) - cnt_val
print("val Sample:" + str(cnt_val))
print("train Sample:" + str(cnt_train))
np.random.shuffle(list2) # Disrupting the list of documents

train_list = []
trainval_list = []
val_list = []

for i in range(int(cnt_train)):
    train_list.append(list2[i])

for i in range(int(cnt_train),int(cnt_train + cnt_val)):
    val_list.append(list2[i])

file = open('Model/LSTM/data/A/Neg-train.txt', 'w', encoding='utf-8')
for i in range(len(train_list)):
    name = str( train_list[i])
    index = name.rfind('.')
    name = name[:index]
    file.write(name+'\n')
file.close()
#
file = open('Model/LSTM/data/A/Neg-test.txt', 'w', encoding='utf-8')
for i in range(len(val_list)):
    name = str(val_list[i])
    index = name.rfind('.')
    name = name[:index]
    file.write(name+'\n')
file.close()
