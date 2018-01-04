import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

data_train = pd.read_csv("/home/yyj/Learn_Machine_Learning/kaggle_test/titanic/train.csv")
# print(data_train.info())
# print(data_train.describe())


#------------------------------------------
fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图
plt.title(u"survived (1 for survived)") # 标题
plt.ylabel(u"num")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"num")
plt.title(u"passenger class")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age) #散点图
plt.ylabel(u"age")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')  # y方向网格
plt.title(u"age (1 for survived)")


plt.subplot2grid((2,3),(1,0), colspan=2) # colspan = 2 表示2列
# plots a kernel density estimate（核密度估计） of the subset of the 1st class passanges's age
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")# plots an axis lable
plt.ylabel(u"density")
plt.title(u"age density")
plt.legend((u'1 class', u'2 class',u'3 class'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"embark num")
plt.ylabel(u"num")
plt.show()


#------------------------------------------
#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'survived':Survived_1, u'unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"survived passenger in class")
plt.xlabel(u"class")
plt.ylabel(u"num")
plt.show()



#-----------------------------
fig = plt.figure()
fig.set(alpha=0.2)

Sex_Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
Sex_Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'survived':Sex_Survived_1, u'unsurvived':Sex_Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("survived passengers in sex")
plt.xlabel("sex")
plt.ylabel("num")
plt.show()

#----------------------------
 #然后我们再来看看各种舱级别情况下各性别的获救情况
fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"class & sex")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"Y", u"N"], rotation=0)
ax1.legend([u"female/high class"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"Y", u"N"], rotation=0)
plt.legend([u"female/low class"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"N", u"Y"], rotation=0)
plt.legend([u"male/high class"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"N", u"Y"], rotation=0)
plt.legend([u"male/low class"], loc='best')

plt.show()


#--------------------------
#我们看看各登船港口的获救情况。

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'survived':Survived_1, u'unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"embark survived")
plt.xlabel(u"embark")
plt.ylabel(u"num")

plt.show()


#------------------------
#下面我们来看看 堂兄弟/妹，孩子/父母有几人，对是否获救的影响。

g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)

g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)


#---------------
#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
#cabin只有204个乘客有值，我们先看看它的一个分布
data_train.Cabin.value_counts()

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'has':Survived_cabin, u'has not':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"has cabin")
plt.xlabel(u"Cabin")
plt.ylabel(u"num")
plt.show()

#咳咳，有Cabin记录的似乎获救概率稍高一些，先这么着放一放吧。

