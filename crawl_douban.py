import requests
from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
import pandas as pd
from lxml import etree


headers = {
'Referer': 'https://movie.douban.com/subject/33420285/comments?status=P',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'
}

def search_game_name_url(Douban_Name):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(r'./chromedriver.exe',options=options)
    driver.get("https://www.douban.com/")
    driver.find_element_by_name('q').send_keys(Douban_Name)
    sleep(2)
    driver.find_element_by_name('q').send_keys(Keys.RETURN)
    driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div[2]/ul/li[11]/a').send_keys(Keys.ENTER)
    driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div[3]/div[2]/div[1]/div[2]/div/h3/a').send_keys(Keys.ENTER)
    get_douban_review(driver.current_url)
def get_douban_review(url): # total_score_final
    reviewers = []
    dates = []
    shot_comments = []
    votes = []
    star = []
    a = 0
    b = ["allstar00"]
    for i in range(0, 500, 20):
        # Reviewers Date reviews Vote
        url_last = f'comments?start={i}&sort=score'
        r1 = requests.get(url + url_last, headers=headers).text
        xml = etree.HTML(r1)
        for comment_list in xml.xpath('//li[@class="comment-item"]') :
            reviewer_first = comment_list.xpath('div[@class="info"]/div[@class="user-info"]/a/text()')
            reviewers = reviewers + reviewer_first
            dates_first = comment_list.xpath('div[@class="info"]/div[@class="user-info"]/span[1]/text()')
            dates = dates + dates_first
            shot_comments_frist = comment_list.xpath('div[@class="info"]/p/span/text()')
            shot_comments = shot_comments + shot_comments_frist
            votes_frist = comment_list.xpath('div[@class="info"]/span[@class="digg"]/span/text()')
            votes = votes + votes_frist
            star_first = comment_list.xpath('div[@class="info"]/div[@class="user-info"]/span[2]/@class')
            star = star + star_first
            a = a+1
            if len(star) != a:
                star = star + b
    short = pd.DataFrame({
         'Reviewers':reviewers,'dates':dates,'Reviews':shot_comments,'votes':votes,'star': star
    })
    short.to_excel('./Platform_information/Douban_information/Douban_reviews.xlsx',index=False)
    short.info()
if __name__ == '__main__':
    a = int(input('Enter the number you want to query:'))
    df = pd.read_excel('./Platform_information/Steam_information/Full_info.xlsx')
    Douban_Name = df.loc[a - 1, 'Name']
    search_game_name_url(Douban_Name)
    # Msc_project.label.label_score()
