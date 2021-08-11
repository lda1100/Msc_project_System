import pandas as pd

def xlsx_long():
    df1 = pd.read_excel('./Platform_information/Douban_information/Douban_reviews.xlsx', sheet_name='Sheet1', usecols=[0])
    data = df1.values
    n = len(data)
    return n

def xlxs_number():
    df = pd.read_excel('./Platform_information/Douban_information/Douban_reviews.xlsx', sheet_name='Sheet1', nrows=xlsx_long())
    data1 = df.values
    return data1

def label_score():
    df = pd.read_excel('./Platform_information/Douban_information/Douban_reviews.xlsx')
    df['label'] = '1'
    for i in range(xlsx_long()):
        a = xlxs_number()[i][4]
        b = a.split('r',2)[1].split('0', 1)[0]
        if b == 'None' or b == '':
            df.loc[i, 'label'] ='0'
        elif int(b)>=4:
            df.loc[i, 'label'] ='1'
        else:
            df.loc[i, 'label'] ='0'
    df.to_excel('./Platform_information/Douban_information/Douban_reviews.xlsx', index=True)

label_score()



