import warnings
warnings.filterwarnings("ignore")
import jieba  # Split word package
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
from wordcloud import WordCloud

# Length of document
def xlsx_long():
    df1 = pd.read_excel('./Platform_information/Steam_information/Steam_reviews/Steam_pos_final.xlsx', sheet_name='Sheet1', usecols=[0])
    data = df1.values
    n = len(data)
    return n

# Read the file
df = pd.read_excel('./Platform_information/Steam_information/Steam_reviews/Steam_pos_final.xlsx')['Reviews']
m = xlsx_long()

# participles
segment = []
for line in range(m):
    try:
        segs = jieba.lcut(df[line])  # jiaba.lcut()
        for seg in segs :
            if len(seg) > 1 and seg != '\r\n' :
                segment.append(seg)
    except:
        print(line)
        continue


# stopwords
words_df = pd.DataFrame({'segment':segment})
# 1:Read the stopwords.txt file
stopwords = pd.read_csv("./stopwords.txt"
                      ,index_col=False
                      ,quoting=3
                      ,sep="\t"
                      ,names=['stopword']
                      ,encoding='utf-8')
# 2:Delete the stopwords
words_df = words_df[~words_df.segment.isin(stopwords.stopword)]

# Statistical word frequency
words_stat = words_df.groupby(by=['segment'])['segment'].agg([("count", "count")])
words_stat = words_stat.reset_index().sort_values(by=["count"], ascending=False)

# words clouds
wordcloud = WordCloud(font_path="./simhei.ttf",background_color="white",max_font_size=80)

word_frequence = {x[0]: x[1] for x in words_stat.head(100).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
pylab.show()
