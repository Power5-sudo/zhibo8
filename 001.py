from bs4 import BeautifulSoup
import requests
headers = {}
headers = {'user-agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}
url = 'https://data.zhibo8.cc/html/match.html?match=%E8%8B%B1%E8%B6%85&saishi=24'
res = requests.get(url,headers=headers).text
soup = BeautifulSoup(res,"lxml")
print(res)