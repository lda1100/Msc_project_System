import re
import pandas as pd

def deal_year(x):
    mon = {'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6','Jul':'7',
           'Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'}
    year, month, day= ' ',' ',' '
    if x['Release Date'] == ' ':
        year = '2000/1/1'
    else:
        year = x['Release Date'][-4:]

        if(x['Release Date'][0] >= '0'and x['Release Date'][0] <= '9' ):
            day = re.sub(' ', '', x['Release Date'][:2])
            month = mon[re.sub(r' |,', '', x['Release Date'][2:6])]
        else:
            day = '1'
            month = mon[x['Release Date'][:3]]
        year = year + '/' + month + '/' + day
    return year
def deal_price(x):
    price = ' '
    if(x['Current price'][:1] != '£' or x['Current price'] == ' '):
        price = '0'
    else:
        price = re.sub(r'£| |,','',x['Current price'])
    return price
def deal_rate(x):
    rate = ' '
    if(x['Favorable rating rate'] == 'Need' or x['Favorable rating rate'] == ' '):
        rate = '0'
    else:
        rate = re.sub(r'%| |o', '', str(x['Favorable rating rate']))
    return rate
def deal_num(x):
    num = ' '
    if x['Number of people evaluated'] == ' Need more user reviews to generate a scor' or x['Number of people evaluated'] == ' ':
        num = '0'
    else:
        num = re.sub(',','',str(x['Number of people evaluated']))
    return num

if __name__ == '__main__':

    df = pd.read_excel('./Platform_information/Steam_information/Full_info.xlsx')
    df['price'] = df.apply(lambda x : deal_price(x), axis=1)
    df['price'] = pd.to_numeric(df['price'])
    df['Favorable rating rate'] = df.apply(lambda x : deal_rate(x), axis=1)
    df['Favorable rating rate'] = pd.to_numeric(df['Favorable rating rate'])
    df['Number of people evaluated'] = df.apply(lambda x : deal_num(x), axis=1)
    df['Number of people evaluated'] = pd.to_numeric(df['Number of people evaluated'])

    df['Release Date'] = df.apply(lambda x: deal_year(x), axis=1)
    result = df.dropna(axis=0)
    result.to_excel('./Platform_information/Steam_information/Scatter_cahrt_data.xlsx')