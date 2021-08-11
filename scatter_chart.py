import datetime

import pandas as pd
from matplotlib import pyplot as plt


def show_first(df):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False      # Set normal display characters

    Y = df['price'] # Y-value for each point
    X = df['Release Date']# X-value for each point
    plt.style.use('seaborn')# Canvas style
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']# Fonts
    plt.figure(figsize=(20, 5))#大小
    # The scatter size here is the inverse of the hot ranking, i.e. the hotter the game, the bigger the dot
    # The colour depends on the popularity rating, the colourbar is also the cmap choice 'RdYlBu' style
    plt.scatter(X,Y, s=15000/(df.index+200), c=df['Favorable rating rate'], alpha=.9,cmap=plt.get_cmap('RdYlBu'))
    plt.colorbar().set_label('Favorable rating rate',fontsize=20)
    '''
    #Partial zoom
    datenow = datetime.datetime(2021, 1, 1)
    dstart = datetime.datetime(2010, 1, 1)
    plt.xlim(dstart, datenow)
    plt.ylim(0, 500)
    '''
    plt.xlabel('Year',fontsize=20)
    plt.ylabel('Price',fontsize=20)
    plt.title('the positive rating of this type of game')
    plt.show()

show_first(pd.read_excel('./Platform_information/Steam_information/Scatter_cahrt_data.xlsx'))