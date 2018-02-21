# DataCastle_HBC
DataCasle精品旅行服务成单预测B榜Rank23解决方案

### Background

皇包车（HI GUIDES）是一个为中国出境游用户提供全球中文包车游服务的平台。拥有境外10万名华人司机兼导游（司导），覆盖全球90多个国家，1600多个城市，300多个国际机场。截止2017年6月，已累计服务400万中国出境游用户。 由于消费者消费能力逐渐增强、 旅游信息不透明程度的下降，游客的行为逐渐变得难以预测，传统旅行社的旅游路线模式已经不能满足游客需求。如何为用户提供更受欢迎、更合适的包车游路线，就需要借助大数据的力量。结合用户个人喜好、景点受欢迎度、天气交通等维度，制定多套旅游信息化解决方案和产品。

### Introduction
队名：不知道  
队员：沙茶海鲜锅、asdss1029、铺铺自摸九只马、NEALC、玛丽  
成绩：单模型（LightGBM）A榜：0.97002（rank28）/B榜：0.97119（rank23）  

利用业余时间玩一玩这类的数据挖掘比赛挺有意思的，能学到不少东西。作为萌新，分享下本次比赛我的比赛代码，就当做是学习过程的一个记录吧，最后用的是我的单模型，没有用啥stacking，成绩不算高，大家看看就好哈。

### How to run
1 在根目录创建data、cache、submission三个文件夹，data用于存放数据文件、cache用于存放缓存文件、submission用于存放提交文件  
2 设置根目录（os.chdir()）  
3 运行higuides.py

### Feature Engineering
userProfile：  
1 缺失信息特征：是否有gender\province\age信息，缺失值个数，信息完整度  
2 各性别类型历史购买精品服务比例、各年龄段历史购买精品服务比例、各省份历史购买精品服务比例  
3 省份做LabelEncoder  
4 gender\province\age做哑变量  
action：  
1 用户历史各actionType的计数\比率  
2 用户活跃天数、各actionType发生天数计数\比率  
3 用户最初的actionType、actionTime、用户最近的actionType、actionTime以及相关组合特征  
4 用户actionTime时间差统计特征，最近K个actionTime时间差  
5 用户action5\6\7\8\9时间差统计特征  
6 用户距最近的actionType1-9的行为个数  
7 用户行为序列56\567\5678\56789计数\比率  
8 用户actionType1-9最近发行为生时间、到最近一次行为时间的时间差  
9 用户actionType序列傅里叶变换，取实部，取前3个分量  
10 用户actionType序列TF-IDF  
11 actionType1-5的时间差小于阈值（100）的数量、actionType5-6的时间差小于阈值（100）的数量、actionType6-7的时间差小于阈值（100）的数量、actionType7-8的时间差小于阈值（100）的数量、actionType8-9的时间差小于阈值（100）的数量  
12 用户最近1天actionType计数\比率，56789连续程度  
13 用户最后一次下单后actionType计数\比率  
14 用户actionTime时间差（2阶差分）统计特征，最近K个actionTime时间差（2阶差分）  
15 用户行为序列小波变换  
orderHistory：  
1 用户历史各orderType计数/比率  
2 用户历史订单的city个数、country个数、continent个数  
3 最近一次orderType、最近一次orderTime、最近一次orderType=1的orderTime  
4 下单时间差  
5 用户历史去各continent计数/比率  
6 用户历史下单city/country/continent精品服务概率  
userComment：  
1 用户tags字数、词数  
2 用户commentsKeyWords词数  
3 rating  
action & orderHistory：  
1 用户下单前各actionType次数\占比  
2 用户每个动作到下单时间的时间差  
3 用户每一单各actionType的时间差  
4 用户最近一单各actionType的时间差  
lastest date Action：  
1 最近1天各actionType次数及比率  
