import openpyxl
import urllib
from selenium import webdriver
from bs4 import BeautifulSoup
from lxml import etree
from tqdm import tqdm
import requests
import re
import pandas as pd
import random
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

# User-Agent
headers = [
    {'User-Agent': 'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)'},
    {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'},
    {'User-Agent': 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'},
    {'User-Agent': 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0'},
    {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0'}
]

class Crawl(object):
    def __init__(self, name, announce):
        # game_type
        self.url = "https://store.steampowered.com/tags/zh-cn/" + name
        self.headers = random.choice(headers)
        self.announce = announce

    def get_page(self):
        # self.url
        response = requests.get(self.url, self.headers)
        # <span id="TopSellers_total">(.*?)</span>
        str = '<span id="' + self.announce + "_total" + '">(.*?)</span>'
        # <span id="TopSellers_total">20,702</span>
        total_number = re.compile(str)
        # first one contained in the regular url
        page_number = re.findall(total_number, response.text)[0]
        # 20,702 => 20702
        final_page_number = re.sub(r',', '', page_number)
        # start a multiple of 15
        # max page
        max_page = int(int(final_page_number) / 15) + 1
        page_num = input("Enter the number of pages you want to crawl, the maximum"
                         " number of pages in the current product column is {}:  ".format(max_page))
        return int(page_num)

class steam_spider_request():
    # Game type,Sales,total number of pages
    def __init__(self,name,announce,page):
        self.headers = random.choice(headers)
        self.name = name
        self.announce = announce
        self.page = page

    def get_spider(self):
        srclist = []
        IDlist = []
        # total number of pages
        for page in tqdm(range(self.page)):
            url = 'https://store.steampowered.com/contenthub/querypaginated/tags/{0}/render' \
                  '/?query=&start={1}&count=15&cc=CN&l=english&v=4&tag={2}' \
                .format(self.announce, page * 15, self.name)
            html = requests.get(url, self.headers).text
            com = re.compile('https://store.steampowered.com/app/(.*?)/')
            com1 = re.compile('href="(.*?)"')
            result = re.sub(r'\\', '', html)
            result = re.findall(com1, result)
            for dat in result:
                srclist.append(str(dat))
                IDlist.append(re.findall(com, str(dat))[0])
        return srclist, IDlist

    def save(self):
        srclist, IDlist = self.get_spider()
        df = pd.DataFrame(list(zip(srclist, IDlist)),
                          columns=['url', 'ID'])
        df.info()
        return df

def get_type(html):
    str = ' '
    final_cost = html.xpath('//div[@class="glance_tags popular_tags"]/a')
    for i in final_cost[0:1]:   # Output the first 6 labels
        str = str + i.text + ' '
    return str

def clear(str):
    com = re.sub(r'\t|\r\n','',str)
    return com
def getreviews(ID): # Get reviews
    r1 = requests.get(
            # schinese
            'https://store.steampowered.com/appreviews/%s?cursor=*&day_range=30&start_date=-1&end_date=-1&date_range_type=all&filter=summary&language=english&l=schinese&review_type=all&purchase_type=all&playtime_filter_min=0&playtime_filter_max=0&filter_offtopic_activity=1'%str(ID),headers=random.choice(headers),timeout=10)
    # rrr = 'https://store.steampowered.com/appreviews/%s?cursor=*&day_range=30&start_date=-1&end_date=-1&date_range_type=all&filter=summary&language=english&l=schinese&review_type=all&purchase_type=all&playtime_filter_min=0&playtime_filter_max=0&filter_offtopic_activity=1'%str(ID)
    # print(rrr)

    soup = BeautifulSoup(r1.json()['html'], 'lxml')
    a = soup.findAll(class_="content")

    list1 = []
    for i in a:
        list1.append(i.text.replace('	', '').replace('\n', '').replace('\r', ''))
    k = str('\n'.join(list1))
    return k
def getdetail(x):
    name, original_cost ,final_cost,now_evaluate, all_evaluate, rate, people, des, type, time, deve, review='', '', '', '', '', '', '', '', '', '','',''
    header = random.choice(headers)
    global count
    try:
        html = requests.get(x['url'], headers=header, timeout=10).text
        xml = etree.HTML(html)
        a1 = xml.xpath('//div[@class="agegate_text_container btns"]/a[1]/span/text()')
        if len(a1) == 1:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('lang=en')
            chrome_options.add_argument('headless')
            driver = webdriver.Chrome(r'./chromedriver.exe',options=chrome_options)
            driver.get(x['url'])
            selectTag = Select(driver.find_element_by_name('ageYear'))
            selectTag.select_by_index(97)
            driver.find_element_by_xpath('//*[@id="app_agegate"]/div[1]/div[4]/a[1]').send_keys(Keys.ENTER)
            r1 = driver.page_source
            xml = etree.HTML(r1)
        else:
            html = requests.get(x['url'], headers=header, timeout=10).text
            xml = etree.HTML(html)
    except:
        print('Server not responding, trying a new request')
        try:
            html = requests.get(x['url'], headers = header, timeout=10).text
            xml = etree.HTML(html)
        except:
            print('Server not responding, trying a new request')
            try:
                html = requests.get(x['url'], headers=header, timeout=10).text
                xml = etree.HTML(html)
            except:
                print('Server not responding, trying a new request')
                #html = requests.get(x['url'], headers=header, timeout=10).text
    try:
        try:
            name = xml.xpath('//*[@id="appHubAppName"]')[0].text
        except :
            print('Incomplete:')
        try:
            original_cost = xml.xpath('//div[@class="discount_prices"]/div[1]')[0].text
            final_cost = xml.xpath('//div[@class="discount_prices"]/div[2]')[0].text
        except:
            original_cost = xml.xpath('//div[@class="game_purchase_price price"]')[0].text
            original_cost = clear(original_cost)
            final_cost = xml.xpath('//div[@class="game_purchase_price price"]')[0].text
            final_cost = clear(final_cost)
        try:
            now_evaluate = xml.xpath('//div[@class="summary column"]/span[1]')[0].text
            all_evaluate = xml.xpath('//div[@class="summary column"]/span[1]')[1].text
            rate = xml.xpath('//div[@class="user_reviews_summary_row"]/@data-tooltip-html')[0][:4]
            people = clear(xml.xpath('//div[@class="summary column"]/span[2]')[0].text)[1:-1]
        except:
            now_evaluate = "None"
            all_evaluate = xml.xpath('//div[@class="summary column"]/span[1]')[0].text
            rate = xml.xpath('//div[@class="user_reviews_summary_row"]/@data-tooltip-html')[0][:4]
            people = clear(xml.xpath('//div[@class="summary column"]/span[2]')[0].text)[1:-1]
        des = clear(xml.xpath('//div[@class="game_description_snippet"]')[0].text)
        type = get_type(xml)
        time = xml.xpath('//div[@class="date"]')[0].text
        deve = xml.xpath('//div[@class="dev_row"]/div[2]/a[1]')[0].text
        review = getreviews(str(x['ID']))

    except:
        print('Unfinished search game number: {}'.format(count))
        # print("")

    count += 1
    return name, original_cost, final_cost, now_evaluate, all_evaluate, rate, people, des, type, time, deve, review

def info(path):
    df = pd.read_excel('./Platform_information/Steam_information/Basic_info.xlsx')

    df['Details']     = df.apply(lambda x: getdetail(x), axis=1)
    df['Name']     = df.apply(lambda x : x['Details'][0], axis=1)
    df['Original Price']     = df.apply(lambda x:x['Details'][1], axis=1)
    df['Current price']     = df.apply(lambda x:x['Details'][2], axis=1)
    df['Recent Comments'] = df.apply(lambda x:x['Details'][3], axis=1)
    df['All Reviews'] = df.apply(lambda x:x['Details'][4], axis=1)
    df['Favorable rating rate']   = df.apply(lambda x:x['Details'][5], axis=1)
    df['Number of people evaluated'] = df.apply(lambda x:x['Details'][6], axis=1)
    df['Game Description'] = df.apply(lambda x:x['Details'][7], axis=1)
    df['Type']     = df.apply(lambda x:x['Details'][8], axis=1)
    df['Release Date'] = df.apply(lambda x:x['Details'][9], axis=1)
    df['Developers']   = df.apply(lambda x:x['Details'][10], axis=1)
    df['Reviews'] = df.apply(lambda x : x['Details'][11], axis=1)
    #df = df.dropna()
    # df = df.drop('Unnamed: 0', axis=1)   # Delete useless data
    df.to_excel(path,index=False)
    df.info()
    print('---Finish---')

def show_detail(path):
    wb = openpyxl.load_workbook(path)
    sheet = wb.active
    max_row = sheet.max_row
    columnB = sheet['D2':'D%d' % max_row]
    for column_cells in columnB :
        for cell in column_cells :
            a = int(str(cell.coordinate).split('D',1)[1])-1
            print(a,cell.value)
if __name__ == '__main__':
    count = 1
    # rep_thread = ['rep1','rep2','rep3','rep4']
    num = ['NewReleases', 'TopSellers', 'ConcurrentUsers', 'TopRated', 'ComingSoon']
    game_type = input('Enter the type of game you want to crawl  ')

    # In Chinese character encoding
    game_type = urllib.parse.quote(game_type)

    game_anno = int(input('Enter the column number of the query: 1,New and Trending 2,Top Sellers 3,What is Poloular 4,Top Rated 5,Upcoming  '))

    # Get the maximum number of pages and return the number of pages you want to crawl 0
    # num[] starts at 0 so minus one
    steam = Crawl(game_type, num[game_anno - 1])

    page = steam.get_page()

    # Basic information about the game for crawling a specified number of pages
    path1 = './Platform_information/Steam_information/Basic_info.xlsx'

    # Game genre Popular/hot selling Corresponding total number of pages
    spider = steam_spider_request(game_type, num[game_anno - 1], page)
    # spider.get_spider(page)
    save = spider.save()
    save.to_excel(path1,index=False)

    path2 = './Platform_information/Steam_information/Full_info.xlsx'
    info(path2)

    df = pd.read_excel('./Platform_information/Steam_information/Full_info.xlsx')
    result = df.dropna(axis=0)
    result.to_excel('./Platform_information/Steam_information/Full_info.xlsx',index=False)

    show_detail('Platform_information/Steam_information/Full_info.xlsx')

