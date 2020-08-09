#球队名 读取写入成功
import requests
import json
import os
import sys
import re
from bs4 import BeautifulSoup
from urllib import request
import xlrd
import xlwt
from xlutils.copy import copy
from datetime import datetime
from bs4 import BeautifulSoup
from bs4 import UnicodeDammit
import sqlite3
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}



def site_web(url):    #网页提取
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}
        res = requests.get(url,headers=headers).text
        res2 = json.loads(res)
        res3 = json.dumps(res2, ensure_ascii=False)
        print(res3)
        data_clean(res3)
    except Exception:
        print('error.site')

def data_clean(x):     #数据清洗 正则表达式
    try:
        zzname = '"球队": "(.*?)".*?'  # 球队名 正则表达式
        zzcc = '"场次": "(.*?)".*?'
        zzw = '"胜": "(.*?)".*?'
        zzp = ', "平": "(.*?)".*?'
        zzl = '"负": "(.*?)".*?'
        teamname = re.findall(zzname, x)
        teamcc = re.findall(zzcc, x)
        teamw = re.findall(zzw, x)
        teamp = re.findall(zzp, x)
        teaml = re.findall(zzl, x)
        y = teamname,teamcc,teamw,teamp,teaml

        write_file(y)
        print('data')
    except Exception:
        print('error.data')
def write_file(text):   #写入文件
    try:
        with open('teamname.txt', 'a+') as f:  # a+ 写入方式为 追加
            for x in range(20):
                f.write(text[x] + ' ')
                f.write(teamcc[x] + ' ')
                f.write(teamw[x] + ' ')
                f.write(teamp[x] + ' ')
                f.write(teaml[x] + '\n')
            print('写入成功')
    except Exception:
        print('writeerror')

    #with open('indexdata.txt','wb') as f :
     #   f.write(text)
      #  f.close()




#try:
#except IndexError:print(error)

if __name__ == '__main__':
    url1 = 'https://dc.qiumibao.com/shuju/public/index.php?_url=/data/index&league=%E8%8B%B1%E8%B6%85&tab=%E7%A7%AF%E5%88%86%E6%A6%9C&year=[year]&_platform=web&_env=pc'
    #输入url网址
    site_web(url1)
    #调用site_web函数  url=url1
    #site_web(input())



#if:else:

#url = 'https://dc.qiumibao.com/shuju/public/index.php?_url=/data/index&league=%E8%8B%B1%E8%B6%85&tab=%E7%A7%AF%E5%88%86%E6%A6%9C&year=[year]&_platform=web&_env=pc'

#JSON到字典转化：
#res2 = json.dumps(res2)
#print(type(res2))
#print(res2)
#字典到JSON转化：
#json_str = json.dumps(dict)

#with open('indexdata.txt','wb+') as f:
 #   f.write(res2)
 #   f.close()