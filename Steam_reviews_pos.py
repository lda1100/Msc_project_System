import time
import numpy as np
from selenium import webdriver
from time import sleep
import pandas as pd
from lxml import etree
import re
headers = {
'Referer': 'https://movie.douban.com/subject/33420285/comments?status=P',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'
}

a = int(input('Enter the number you want to query:'))
df = pd.read_excel('./Platform_information/Steam_information/Full_info.xlsx')
Steam_ID = df.loc[a-1, 'ID']

def xlsx_long():
    df1 = pd.read_excel('./Platform_information/Steam_information/Full_info.xlsx', sheet_name='Sheet1', usecols=[0])
    data = df1.values
    n = len(data)
    return n

def xlxs_number():
    df = pd.read_excel('./Platform_information/Steam_information/Full_info.xlsx', sheet_name='Sheet1', nrows=xlsx_long())
    data1 = df.values
    return data1

def game_name_url():
    name_list = []
    for i in range(xlsx_long()):
        name_list.append(xlxs_number()[i][3])
    return name_list
df = pd.read_excel('./Platform_information/Steam_information/Full_info.xlsx')

def search_game_name_url(ID):
    a =  "https://steamcommunity.com/app/%s/positivereviews/?browsefilter=toprated&snr=1_5_100010_&filterLanguage=english" %(ID)
    get_steam_review(a)
def remove_punctuation(line):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub(' ',line)
    return line
def get_steam_review(url):
    reviewers =[]
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('headless')
    chrome_options.add_argument('lang=en')
    driver = webdriver.Chrome(r'./chromedriver.exe', chrome_options=chrome_options)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.get(url)
    js = "return action=document.body.scrollHeight"
    height = driver.execute_script(js)
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
    t1 = int(time.time())
    # Define the loop identifier to terminate the while loop
    status = True
    # Number of retries
    num = 0
    while status:
        # Get the current timestamp (seconds)
        t2 = int(time.time())
        # Determine if the difference between the initial timestamp and the
        # current timestamp is greater than 30 seconds, less than 30 seconds then drop down the scroll bar
        if t2 - t1 < 30:
            new_height = driver.execute_script(js)
            # print(new_height)
            if int(new_height) > height:
                sleep(2)
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
                # Reset initial page height
                height = new_height
                # Resetting the initial timestamp and retiming
                t1 = int(time.time())
            elif num < 20:  # When the page height has not been updated for more than 20 seconds, the retry logic enters, retrying 3 times, waiting 20 seconds each time
                sleep(3)
                num = num + 1
            else:  # Timeout and retry count exceeded, the program ends the loop and the page is considered loaded
                print("The scroll bar is already at the bottom of the page!")
                status = False
                # Scroll bar adjusted to top of page
                driver.execute_script('window.scrollTo(0, 0)')
                # sleep(50)

    r1 = driver.page_source
    xml = etree.HTML(r1)
    for apphub_CardRow in xml.xpath('//div[@class="apphub_Card modalContentLink interactable"]') :
        reviewers_frist = apphub_CardRow.xpath('div[1]/div[1]/div[3]/text()')
        for c in reviewers_frist:
            b = remove_punctuation(c)
            d = ' '.join(b.split())
            reviewers.append(d.replace('\n', '').replace('\r', ''))
    test = [b for b in reviewers if b != '']
    b = np.array(test)
    Steam_reviews = []
    label = []
    label_pos = '1'
    for item in (b) :
        if len(item) > 4:
            Steam_reviews.append(item)
        else :
            continue
    len_steam = len(Steam_reviews)
    for d in range(len_steam):
        label.append(label_pos)

    short = pd.DataFrame({
        'Reviews' : Steam_reviews, 'lebel' : label
    })
    short.to_excel('./Platform_information/Steam_information/Steam_reviews/Steam_pos_final.xlsx', index=False)


search_game_name_url(Steam_ID)
