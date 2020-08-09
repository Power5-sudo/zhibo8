# -*- coding: utf-8 -*-
import json
import re
from multiprocessing import Pool
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import sys,io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="gb18030") #改变标准输出的默认编码

 # 创建正则表达式规则对象，匹配每页里的段子内容，re.S 表示匹配全部字符串内容
 #pattern = re.compile('<div\sclass="f18 mb20">(.*?)</div>', re.S)
 # 将正则匹配对象应用到html源码字符串里，返回这个页面里的所有段子的列表
 #content_list = pattern.findall(html.decode('gbk'))

#sys.stdout的形式就是print的一种默认输出格式
#这里为什么不用print呢：https://www.cnblogs.com/sundahua/p/10206801.html


#获取一个页面
def get_one_page(url):
    try:
        response = requests.get(url) #请求目标网站
        #response.encoding = "utf-8"
        if response.status_code ==200: #获取状态码，判断是否请求成功(200)  HTTP状态码：https://www.runoob.com/http/http-status-codes.html
            return response.text  #以文本形式打印网页源码
        return None
    except RequestException:
        return None

def parse_one_page(html):
    # 构建beautifulsoup实例
    soup = BeautifulSoup(html,'html.parser')  # 第一个参数是要匹配的内容 第二个参数是beautifulsoup要采用的模块，即规则，HTMLParser是Python自带的网页解析库
    host = "http://www.52jingsai.com/"
    inf_list = soup.find_all("dl")# find_all返回所有匹配到的结果，返回所有di标签，所有竞赛信息都在dl标签中
    for inf in inf_list:
        cover = host + inf.img["src"] # 封面，我爱竞赛网网页源码中的图片是没有域名的，无法直接访问，所以在此要拼接上域名
        name = inf.dt.text.strip() # 名称  strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        details = inf.dd.text.split("\n")[1].strip() # 介绍 split()通过指定分隔符对字符串进行切片,返回字符串数组
        type = inf.label.text.strip() # 类型
        date = inf.label.next_sibling.next_sibling.next_sibling.strip() #发布时间 返回某个元素之后紧跟的元素（处于同一树层级中）
        yield {# 迭代给字典中的键赋值
            'cover':cover,
            'name':name,
            'details':details,
            'type':type,
            'date':date
        }

#将信息存储到文件中
def write_to_file(content):
    with open('Contest.txt', 'a', encoding='gb18030') as f:  #模式a:打开一个文件用于追加
        #将字典内容转换为Json字符串
        f.write(json.dumps(content, ensure_ascii=False)+'\n')
        f.close()


def main(page):
    url = 'http://www.52jingsai.com/bisai/index.php?page=' + str(page)
    html = get_one_page(url)
    for item in parse_one_page(html):
        print(item)
        write_to_file(item)


#Python是脚本语言，他并不是将main函数作为入口函数，从脚本第一行开始运行，没有统一的入口
#当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
if __name__ == '__main__':
    pool = Pool() # 创建进程池
    pool.map(main, [i for i in range(1,178)]) ## 第一个参数是函数，第二个参数是一个迭代器，将迭代器中的数字作为参数依次传入函数中