import requests
import json
import re
#解决了Json编译问题
###
###

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}

url = 'https://dc.qiumibao.com/shuju/public/index.php?_url=/data/index&league=%E8%8B%B1%E8%B6%85&tab=%E7%A7%AF%E5%88%86%E6%A6%9C&year=[year]&_platform=web&_env=pc'
res = requests.get(url,headers=headers).text
res2 = json.loads(res)
print(res2)
res3 = json.dumps(res2,ensure_ascii=False)
print(res3.strip())


zzname = '"球队": "(.*?)".*?'                           #球队名 正则表达式
zzcc = '"场次": "(.*?)".*?'
zzw = '"胜": "(.*?)".*?'
zzp = ', "平": "(.*?)".*?'
zzl = '"负": "(.*?)".*?'
teamname = re.findall(zzname,res3)
teamcc = re.findall(zzcc,res3)
teamw = re.findall(zzw,res3)
teamp = re.findall(zzp,res3)
teaml = re.findall(zzl,res3)

with open('teamname.txt','a+') as f:  #a+ 写入方式为 追加
    for x in range(20):
        f.write(teamname[x] + ' ')
        f.write(teamcc[x] + ' ')
        f.write(teamw[x] + ' ')
        f.write(teamp[x] + ' ')
        f.write(teaml[x] + '\n')

    f.close()







 #球队name
#tgname = '"场次": "(.*?)", "胜": "(.*?)", "平": "(.*?)", "负": "(.*?)",'                           #球队名 正则表达式
#teamgame = re.findall(tgname,res3)
#print(teamgame)

#try:
 #   with open('teamname.txt','w') as f:
  #      for line in teamname:
   #         for x in range(5):
   #             f.writelines(line[x] + '\t')
    #    f.close()
     #   print('写入成功')
#except Exception:
 #   print('写入失败02')

