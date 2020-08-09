Python 学习笔记
Python 学习笔记 1
1、利用matplotlib绘制散点图：    2
2、利用matplotlib绘制正弦曲线：   3
3、绘制一组幂函数   4
4、绘制直方图：    5
5、极坐标绘图：    7
6、柱形图：  9
7、scipy 10
8、ipython用法：    10
（1）保存第2行到第22行的输入命令：%save 2-22   10
（2）显示ipython命令的帮助信息：?%save  11
（3）windows命令行里面，左键选择后单击右键为复制所选内容。   11
（4）在启动Ipython时可以使用--pylab启动，它相当于导入了如下库： 11
（5）IPython的Magic函数  11
（6）执行shell命令时，在前面加上感叹号。 12
（7）参考文章：    12
（8）在其他程序中使用ipython时，包的导入：   12
（9）颜色，tab的使用，需要安装pyreadline库    12
（10）跳到指定的历史行    13
9、模拟掷色子：    13
10、IPython科学计算环境的配置：    13
11、numpy向量运算：   13
12、复数   14
（1）建立复数：（虚数单位用j表示，虚部和j之间没有乘号）   14
（2）复数作图 14
（3）复数乘法的几何意义：   15
13、三维作图 21
14、jupyter（原来的ipython notebook） 22
（1）安装：  22
（2）运行：  22
（3）开启pylab模式：导入必要的包 22
（4）直接在网页上显示matplotlib图像：    22
（5）显示数学公式：  23
15、numpy的网格函数：  23
（1）np.meshgrid  23
(2)np.mgrid 24
16、SymPy：Python语言符号计算   24
17、IPython测试代码执行时间  24
（1）%time：测试语句或表达式的执行时间。 24
(2)%timeit:语句或表达式的执行时间  25
18、ipython基本性能分析：   25
（1）%prun    25
(2)%run –p  25
19、numpy数组合并    26
20、pandas合并两个DataFrame  26
21、pandas索引 27
（1）MultiIndex   27
reset_index()层次化索引的级别会被移到列里面    28
（2）DataFrame索引  28
obj[val] 选取DataFrame的单个列或一组列    28
obj.ix[val] 选取DataFrame的单个行或一组行 29
obj.ix[:,val] 选取单个列或列子集 29
obj.ix[val1, val2] 同时选取行和列  30
（3）层次化索引    30
22、flask微框架 31
（1）安装   31
（2）一个最小的应用  31
（3）构造url    31
23、ipython命令    32
（1）%autoindent自动缩进  32
（2）%automagic   32
(3) %paste  33




1、利用matplotlib绘制散点图：

import numpy as np
import matplotlib.pyplot as plt

x=np.random.randint(1000,size=10) #生成10个0~1000的随机整数
y=np.random.randint(1000,size=10)
plt.scatter(x,y)
plt.show()

运行结果：



2、利用matplotlib绘制正弦曲线：

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-2*np.pi,2*np.pi,0.1)
y=np.sin(x)
plt.plot(x,y)
plt.show()

运行结果：



3、绘制一组幂函数

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-4,4,0.1)
f1=10**x
f2=np.e**x
f3=2**x
plt.axis([-4,4,-0.5,8]) #设置坐标轴刻度范围
plt.plot(x,f1,'r',x,f2,'b',x,f3,'g') #绘制一组函数，设置为不同颜色
plt.text(1,7.5,r'$10^x$') #设置图像的标题
plt.text(2.2,7.5,r'$e^x$')
plt.text(3.2,7.5,r'$2^x$')
plt.show()

运行结果：



4、绘制直方图：
方式一：
n=np.random.randn(10000)
plt.hist(n,50) #50为分组数
plt.show()

运行结果：



方式二：
import numpy as np
import matplotlib.pyplot as plt

mu,sigma=100,15
x=mu+sigma*np.random.randn(10000)
n,bins,patches=plt.hist(x,50,normed=1,facecolor='g',alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60,.025, r'$\mu=100,\ \sigma=15$')
plt.axis([40,160,0,0.03])
plt.show()

运行结果：

5、极坐标绘图：
方式一：
import numpy as np
import matplotlib.pyplot as plt

theta=np.arange(0,2*np.pi,0.01)
r=2*theta
plt.polar(theta,r)
plt.show()

运行结果：



方式二：
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
r = np.arange(0,1,0.001)
theta = 2*2*np.pi*r
line, = ax.plot(theta, r, color='#ee8d18', lw=3)
plt.show()

运行结果：


6、柱形图：

import numpy as np
import matplotlib.pyplot as plt
x=[1,2,3]
plt.bar(x,x)
plt.show()

运行结果：




7、scipy

import scipy
a=scipy.zeros(1000) #建立以1000列的全零向量
b=scipy.fft(a) #快速傅里叶变换

import pylab
pylab.plot(b)
pylab.show()




8、ipython用法：

（1）保存第2行到第22行的输入命令：%save 2-22

In [23]: %save t 2-22

（2）显示ipython命令的帮助信息：?%save

In [21]: ?%save
Docstring:
Save a set of lines or a macro to a given filename.

Usage:
  %save [options] filename n1-n2 n3-n4 ... n5 .. n6 ...

Options:
……

（3）windows命令行里面，左键选择后单击右键为复制所选内容。

（4）在启动Ipython时可以使用--pylab启动，它相当于导入了如下库：
import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot
from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs
from pylab import *
from numpy import *

（5）IPython的Magic函数
显示当前目录：
%pwd

Magic命令后加问号，显示命令的帮助信息。

列出所有Magic函数：

In [7]: %lsmagic
Out[7]:
Available line magics:
%alias  %alias_magic  %autocall  %autoindent  %automagic  %bookmark  %cd  %cls  %colors  %config  %copy  %cpaste  %ddir  %debug  %dhist  %dirs  %doctest_mode  %echo  %ed  %edit  %env  %gui  %hist  %history  %install_default_config  %install_ext  %install_profiles  %killbgscripts  %ldir  %load  %load_ext  %loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %macro  %magic  %matplotlib  %mkdir  %notebook  %page  %paste  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %popd  %pprint  %precision  %profile  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %quickref  %recall  %rehashx  %reload_ext  %ren  %rep  %rerun  %reset  %reset_selective  %rmdir  %run  %save  %sc  %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  %xdel  %xmode

Available cell magics:
%%!  %%HTML  %%SVG  %%bash  %%capture  %%cmd  %%debug  %%file  %%html  %%javascript  %%latex  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  %%sx  %%system  %%time  %%timeit  %%writefile

Automagic is ON, % prefix IS NOT needed for line magics.

（6）执行shell命令时，在前面加上感叹号。
In [14]: !dir

（7）参考文章：
笔记:IPython基础 - 个人笔记共同分享 - 博客频道 - CSDN.NET
http://blog.csdn.net/tiany524/article/details/6230029

（8）在其他程序中使用ipython时，包的导入：
import IPython

（9）颜色，tab的使用，需要安装pyreadline库

（10）跳到指定的历史行
例如：跳到已经输入的第1行
%hist 1
9、模拟掷色子：
a=np.random.random(1000)
a[a>0.5]=1
a[a<=0.5]=0
print(np.sum(a))

（IPython）
In [16]: a=np.random.random(1000)
In [18]: a[a>0.5]=1
In [19]: a[a<=0.5]=0
In [21]: np.sum(a)
Out[21]: 527.0 #出现正面的次数

10、IPython科学计算环境的配置：
（1）安装Python；
（2）利用pip安装：ipython, pyreadline
（3）从下面的网址下载whl文件：numpy+mkl, matplotlib, pandas, scipy
Python Extension Packages for Windows - Christoph Gohlke
http://www.lfd.uci.edu/~gohlke/pythonlibs/

11、numpy向量运算：
例：  ，求  的坐标。
In [1]: p1=np.array([2,-1])
In [2]: p2=np.array([0,5])
In [3]: p=(p1+2*p2)/3
In [4]: p
Out[4]: array([ 0.66666667,  3.        ])

12、复数
（1）建立复数：（虚数单位用j表示，虚部和j之间没有乘号）
In [31]: a=3+4j
In [32]: a
Out[32]: (3+4j)

验证  ：
In [35]: 1j*1j
Out[35]: (-1+0j)

四则运算：
In [70]: (3+4j)+(1+2j)
Out[70]: (4+6j)
In [71]: (3+4j)-(1+2j)
Out[71]: (2+2j)
In [72]: (3+4j)*(1+2j)
Out[72]: (-5+10j)
In [73]: (3+4j)/(1+2j)
Out[73]: (2.2-0.4j)
In [74]: (3+4j)**2
Out[74]: (-7+24j)
In [75]: sqrt(3+4j)
Out[75]: (2+1j)

复数的模：
abs(3+4j)

共轭复数:
conjugate()

（2）复数作图
In [40]:  a=np.array([1+2j,3+4j])
In [41]: plot(a.real, a.imag)

注意：如果直接plot(a)，则去掉a的虚部后再绘图。

（3）复数乘法的几何意义：
复数的三种表示形式：①代数形式  ；②三角形式：  ；③指数形式：  。

 乘以z2后，相当于把z1的模伸缩 倍，同时旋转 角度。
例如： 的图像如下，复数 后，相当于逆时针旋转 。
In [93]: x=np.linspace(-2*pi,2*pi,num=100)
In [94]: y=sin(x)
In [95]: plot(x,y)

In [99]: d=(x+1j*y)*e**(1j*pi/4)
In [100]: plot(d.real,d.imag)



例2、先画出四条线，
In [128]: x=linspace(0,5,num=100)
In [129]: y=[1j,2j,3j,4j]
In [130]: d=np.array([i+j for i in x for j in y])
In [131]: scatter(d.real,d.imag)

画出 的图像
In [135]: d2=d**2
In [136]: scatter(d2.real,d2.imag) #画散点图

再画出 的图像
In [138]: d3=d**3
In [139]: scatter(d3.real,d3.imag)


例3、作 的图像
x,y=mgrid[-2:2:20j,-2:2:20j]
z=x+1j*y
plot(z.real,z.imag,z.imag,z.real)

w1=e**(x+1j*y)
w2=e**(y+1j*x)
plot(w1.real,w1.imag,w2.real,w2.imag)


连续的几个例子


13、三维作图
例如：作图像 。
import mpl_toolkits.mplot3d
x,y=np.mgrid[-2:2:20j,-2:2:20j]
z=x*np.exp(-x**2-y**2)
ax=plt.subplot(111,projection='3d')
ax.plot_surface(x,y,z)


14、jupyter（原来的ipython notebook）
（1）安装：
pip install jupyter

安装使用jupyter（原来的notebook） - 我思故我在 - 博客频道 - CSDN.NET
http://blog.csdn.net/superdont/article/details/46468781
（2）运行：
Jupyter notebook
（3）开启pylab模式：导入必要的包
%pylab
（4）直接在网页上显示matplotlib图像：
%matplotlib inline

（5）显示数学公式：


15、numpy的网格函数：
（1）np.meshgrid

(2)np.mgrid

说明：步长为虚数时，虚部的整数部分表示的是元素的个数。
>>> np.mgrid[-1:1:5j]
array([-1. , -0.5,  0. ,  0.5,  1. ])

16、SymPy：Python语言符号计算


17、IPython测试代码执行时间
（1）%time：测试语句或表达式的执行时间。
例如：
In [1]: %time print('hello world')
hello world
Wall time: 0 ns
In [2]: n=1000000

In [3]: %time sum(range(n))
Wall time: 118 ms
Out[3]: 499999500000
(2)%timeit:语句或表达式的执行时间
行模式用法：
  %timeit [-n<N> -r<R> [-t|-c] -q -p<P> -o] statement
或者单元格模式用法：
  %%timeit [-n<N> -r<R> [-t|-c] -q -p<P> -o] setup_code
  code
  code...
18、ipython基本性能分析：
（1）%prun
例如：
In [13]: def f(x):
   ....:     return x**2
   ....:

In [14]: %prun f(2)
         4 function calls in 0.000 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 <ipython-input-13-05261bd2898a>:1(f)
        1    0.000    0.000    0.000    0.000 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
(2)%run –p



19、numpy数组合并
In [14]: a=array([[1,2,3],[4,5,6]])
In [15]: a
Out[15]:
array([[1, 2, 3],
       [4, 5, 6]])
In [16]: b=array([[7],[8]])
In [17]: b
Out[17]:
array([[7],
       [8]])
In [26]: c=np.concatenate((a,b),axis=1)
In [27]: c
Out[27]:
array([[1, 2, 3, 7],
       [4, 5, 6, 8]])

In [33]: d=array([[7,8,9]])

In [34]: f=np.concatenate((a,d))

In [35]: f
Out[35]:
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
20、pandas合并两个DataFrame

import pandas as pd
from pandas import DataFrame,Series

In [41]: df1=DataFrame(array([[1,2,3],[4,5,6]]),columns=['a','b','c'])

In [42]: df1
Out[42]:
   a  b  c
0  1  2  3
1  4  5  6

In [43]: df2=DataFrame(array([[7,8,9],[10,11,12]]),columns=['a','b','c'])

In [44]: df2
Out[44]:
    a   b   c
0   7   8   9
1  10  11  12
In [48]: df1.merge(df2,how='outer')
Out[48]:
      a     b     c
0   1.0   2.0   3.0
1   4.0   5.0   6.0
2   7.0   8.0   9.0
3  10.0  11.0  12.0
21、pandas索引
（1）MultiIndex
In [53]: df1=pd.DataFrame(arange(20).reshape(5,4),columns=['a','b','c','d'])

In [54]: df1
Out[54]:
    a   b   c   d
0   0   1   2   3
1   4   5   6   7
2   8   9  10  11
3  12  13  14  15
4  16  17  18  19

In [55]: d=df1.groupby(['a','b']).count()

In [56]: d
Out[56]:
       c  d
a  b
0  1   1  1
4  5   1  1
8  9   1  1
12 13  1  1
16 17  1  1

In [57]: d.iloc[d.index.get_level_values('a')<=10]
Out[57]:
     c  d
a b
0 1  1  1
4 5  1  1
8 9  1  1
In [58]: d.index
Out[58]:
MultiIndex(levels=[[0, 4, 8, 12, 16], [1, 5, 9, 13, 17]],
           labels=[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
           names=['a', 'b'])

reset_index()层次化索引的级别会被移到列里面
In [77]: d=d.reset_index()

In [78]: d
Out[78]:
    a   b  c  d
0   0   1  1  1
1   4   5  1  1
2   8   9  1  1
3  12  13  1  1
4  16  17  1  1

In [79]: d[d['a']<10]
Out[79]:
   a  b  c  d
0  0  1  1  1
1  4  5  1  1
2  8  9  1  1
（2）DataFrame索引
obj[val] 选取DataFrame的单个列或一组列
In [60]: df1
Out[60]:
    a   b   c   d
0   0   1   2   3
1   4   5   6   7
2   8   9  10  11
3  12  13  14  15
4  16  17  18  19

In [61]: df1['a']
Out[61]:
0     0
1     4
2     8
3    12
4    16
Name: a, dtype: int32

In [62]: df1[['a','c']]
Out[62]:
    a   c
0   0   2
1   4   6
2   8  10
3  12  14
4  16  18
obj.ix[val] 选取DataFrame的单个行或一组行
In [63]: df1.ix[[0,2]]
Out[63]:
   a  b   c   d
0  0  1   2   3
2  8  9  10  11
obj.ix[:,val] 选取单个列或列子集
In [64]: df1.ix[:,['b','d']]
Out[64]:
    b   d
0   1   3
1   5   7
2   9  11
3  13  15
4  17  19
obj.ix[val1, val2] 同时选取行和列
In [65]: df1.ix[:2,['b','d']]
Out[65]:
   b   d
0  1   3
1  5   7
2  9  11

（3）层次化索引
In [4]:
data=pd.Series(np.random.randn(10),index=[['a','a','a','b','b','b','c','c','d','d'],[1,2,3,1,2,3,1,2,2,3]])

In [5]: data
Out[5]:
a  1    0.665760
   2    0.144131
   3    0.266128
b  1    1.313978
   2   -0.521947
   3   -0.895899
c  1    0.834148
   2   -1.148669
d  2    0.941651
   3    0.626735
dtype: float64

In [6]: data.index
Out[6]:
MultiIndex(levels=[['a', 'b', 'c', 'd'], [1, 2, 3]],
           labels=[[0, 0, 0, 1, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 1, 2, 0, 1, 1, 2]])

In [7]: data['b']
Out[7]:
1    1.313978
2   -0.521947
3   -0.895899
dtype: float64

In [8]: data[:,2]
Out[8]:
a    0.144131
b   -0.521947
c   -1.148669
d    0.941651
dtype: float64

In [9]: data.ix[['b','d']]
Out[9]:
b  1    1.313978
   2   -0.521947
   3   -0.895899
d  2    0.941651
   3    0.626735
dtype: float64

22、flask微框架
（1）安装
pip install flask
（2）一个最小的应用
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()

运行后：
* Running on http://127.0.0.1:5000/
（3）构造url
>>> from flask import Flask, url_for
>>> app = Flask(__name__)
>>> @app.route('/')
... def index(): pass
...
>>> @app.route('/login')
... def login(): pass
...
>>> @app.route('/user/<username>')
... def profile(username): pass
...
>>> with app.test_request_context():
...  print url_for('index')
...  print url_for('login')
...  print url_for('login', next='/')
...  print url_for('profile', username='John Doe')
...
/
/login
/login?next=/
/user/John%20Doe

23、ipython命令
（1）%autoindent自动缩进
（2）%automagic
Make magic functions callable without having to type the initial %.

Without argumentsl toggles on/off (when off, you must call it as
%automagic, of course).  With arguments it sets the value, and you can
use any of (case insensitive):

 - on, 1, True: to activate

 - off, 0, False: to deactivate.

Note that magic functions have lowest priority, so if there's a
variable whose name collides with that of a magic fn, automagic won't
work for that function (you get the variable instead). However, if you
delete the variable (del var), the previously shadowed magic function
becomes visible to automagic again.
(3) %paste
Paste & execute a pre-formatted code block from clipboard.
