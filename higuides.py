import pandas as pd
import os
import numpy as np
import gc
import re
import pickle
from scipy.stats import skew
from scipy.fftpack import fft
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pywt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
import lightgbm as lgb
os.chdir('D:/data/DataCastle/huangbaoche/')

###################################### constant variables ###############################
num_province = 10
eta = 0.00000001

###################################### dump path ########################################
userProfile_dump_path = 'cache/userProfile_feats.pkl'
Action_dump_path = 'cache/action_feats.pkl'
orderHistory_dump_path = 'cache/orderHistory_feats.pkl'
userComment_dump_path = 'cache/userComment_feats.pkl'
order_action_dump_path = 'cache/order_action_feats.pkl'
latestdate_action_dump_path = 'cache/latestdate_action_feats.pkl'

###################################### function #########################################
def is_not_nan(x):
    if pd.isnull(x) == True:
        return 0
    else:
        return 1

def cutting(df):
    pop_province = df['province'].value_counts().index[:num_province]
    df.loc[~df['province'].isin(pop_province), 'province'] = u'其他'
    return df

def timestamp_transform(unix):
    time_local = time.localtime(unix)
    tm = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return tm

def timespan_thredthold(x):
    y = []
    for i in x:
        if i < 86400:
            y.append(i)
    return y

def get_tm_diff_max(diff):
    try:
        return np.max(diff)
    except:
        return np.nan

def get_tm_diff_min(diff):
    try:
        return np.min(diff)
    except:
        return np.nan

def get_k_index(x, k):
    try:
        return x[k]
    except:
        return np.nan

def getActionTimeSpan(df_action_of_userid, actiontypeA, actiontypeB, timethred):
    timespan_list = []
    i = 0
    while i < (len(df_action_of_userid) - 1):
        if df_action_of_userid['actionType'].iat[i] == actiontypeA:
            timeA = df_action_of_userid['actionTime'].iat[i]
            for j in range(i + 1, len(df_action_of_userid)):
                if df_action_of_userid['actionType'].iat[j] == actiontypeA:
                    timeA = df_action_of_userid['actionTime'].iat[j]
                if df_action_of_userid['actionType'].iat[j] == actiontypeB:
                    timeB = df_action_of_userid['actionTime'].iat[j]
                    timespan_list.append(timeB - timeA)
                    i = j
                    break
        i += 1
    return np.sum(np.array(timespan_list) <= timethred) / (np.sum(np.array(timespan_list)) + 1.0)

def get_actionType_first_actionTime(df, actionType):
    df = df[df.actionType == actionType]
    tm = df.groupby('userid').actionTime.nsmallest(1).reset_index()
    del tm['level_1']
    tm.columns = ['userid', 'first_actionType' + str(actionType) + '_tm']
    return tm

def get_actionType_latest_actionTime(df, actionType):
    df = df[df.actionType == actionType]
    tm = df.groupby('userid').actionTime.nlargest(1).reset_index()
    del tm['level_1']
    tm.columns = ['userid', 'latest_actionType' + str(actionType) + '_tm']
    return tm

def get_last_wave_feature_dwt_cd(i, j, x):
    x = list(x)
    if len(x) <= 4:
        return np.nan
    else:
        return pywt.dwt(x[-4:], 'db2')[i][j-1]

def gen_actionTypek_tm_diff_feats(actionType):
    # 读取数据
    action_train = pd.read_csv('trainingset/action_train.csv')
    action_test = pd.read_csv('test/action_test.csv')
    df = pd.concat([action_train, action_test], axis=0)  # type:pd.DataFrame
    # 找到每个用户最近的actionTypek时间戳
    tm = df[df.actionType == actionType].groupby('userid').last()['actionTime'].reset_index()
    tm.columns = ['userid', 'latest_actionTime']
    # 到最近的actionTypek的时间间隔
    df = pd.merge(df, tm, on = 'userid', how = 'left')
    df['diff'] = df.latest_actionTime - df.actionTime
    # 到最近的actionTypek的时间间隔均值
    diff_mean = df.groupby('userid').mean()['diff'].reset_index()
    diff_mean.columns = ['userid', 'to_latest_actionType_' + str(actionType) + '_diff_mean']
    # 到最近的actionTypek的时间间隔最大值
    diff_max = df.groupby('userid').max()['diff'].reset_index()
    diff_max.columns = ['userid', 'to_latest_actionType_' + str(actionType) + '_diff_max']
    # 到最近的actionTypek的时间间隔最小值
    diff_min = df.groupby('userid').min()['diff'].reset_index()
    diff_min.columns = ['userid', 'to_latest_actionType_' + str(actionType) + '_diff_min']
    # 到最近的actionTypek的时间间隔方差
    diff_var = df.groupby('userid').var()['diff'].reset_index()
    diff_var.columns = ['userid', 'to_latest_actionType_' + str(actionType) + '_diff_var']
    # 到最近的actionTypek的时间间隔中位数
    diff_median = df.groupby('userid').median()['diff'].reset_index()
    diff_median.columns = ['userid', 'to_latest_actionType_' + str(actionType) + '_diff_median']
    # 到最近的actionTypek的时间间隔峰度
    diff_skew = df.groupby('userid').skew()['diff'].reset_index()
    diff_skew.columns = ['userid', 'to_latest_actionType_' + str(actionType) + '_diff_skew']
    # 到最近的actionTypek的时间间隔总和
    diff_sum = df.groupby('userid').sum()['diff'].reset_index()
    diff_sum.columns = ['userid', 'to_latest_actionType_' + str(actionType) + '_diff_sum']
    actionTypek_tm_diff_feats = pd.merge(diff_mean, diff_max, on = 'userid', how = 'left')
    actionTypek_tm_diff_feats = pd.merge(actionTypek_tm_diff_feats, diff_min, on = 'userid', how = 'left')
    actionTypek_tm_diff_feats = pd.merge(actionTypek_tm_diff_feats, diff_var, on = 'userid', how = 'left')
    actionTypek_tm_diff_feats = pd.merge(actionTypek_tm_diff_feats, diff_median, on = 'userid', how = 'left')
    actionTypek_tm_diff_feats = pd.merge(actionTypek_tm_diff_feats, diff_skew, on = 'userid', how = 'left')
    actionTypek_tm_diff_feats = pd.merge(actionTypek_tm_diff_feats, diff_sum, on = 'userid', how = 'left')
    gc.collect()
    return actionTypek_tm_diff_feats

def tags_words_count(comment):
    if comment == u'无':
        return 0
    else:
        comment = re.sub('\|', '', comment)
        return len(comment)

def tag_phrase_count(comment):
    if comment == u'无':
        return 0
    if len(comment.split('|')) > 1:
        return len(comment.split('|'))
    else:
        return 1

def commentsKeyWords_phrase_count(comment):
    if comment == u'无':
        return 0
    if len(comment.split(',')) > 1:
        return len(comment.split(','))
    else:
        return 1

def calc_seqentialratio(df_action_of_userid):
    pos_5 = -1
    result = 0
    df_len = len(df_action_of_userid)
    for i in range(0, df_len):
        if df_action_of_userid['actionType'].iat[i] == 5:
            pos_5 = i
    if pos_5 != -1:
        result += 1
        if pos_5 + 1 < df_len:
            if df_action_of_userid['actionType'].iat[pos_5 + 1] == 6:
                result += 1
                if pos_5 + 2 < df_len:
                    if df_action_of_userid['actionType'].iat[pos_5 + 2] == 7:
                        result += 1
                        if pos_5 + 3 < df_len:
                            if df_action_of_userid['actionType'].iat[pos_5 + 3] == 8:
                                result += 1
    return result

def gen_action_feats_mid(df, i):
    latest_actionDt = df.groupby('userid').last()['actionDt'].reset_index()
    latest_actionDt.columns = ['userid', 'latest_actionTm']
    df_cl = pd.merge(df, latest_actionDt, on = 'userid', how = 'left')
    df_cl['tmDiff'] = df_cl.latest_actionTm - df_cl.actionDt
    df_cl['tmDiff'] = df_cl.tmDiff.dt.days
    df_cl['weight'] = df_cl.tmDiff.apply(lambda x: np.exp(-x))
    df_cl = df_cl[df_cl.tmDiff <= i]
    # 各actionType计数、比率
    user_actionType_cnt = pd.get_dummies(df_cl.actionType, prefix = 'windows %s actionType' % str(i))
    user_actionType_cnt = pd.concat([df_cl[['userid', 'weight']], user_actionType_cnt], axis = 1)  # type:pd.DataFrame
    user_actionType_cnt['windows %s actionType_1' % str(i)] = user_actionType_cnt['windows %s actionType_1' % str(i)] * user_actionType_cnt.weight
    user_actionType_cnt['windows %s actionType_2' % str(i)] = user_actionType_cnt['windows %s actionType_2' % str(i)] * user_actionType_cnt.weight
    user_actionType_cnt['windows %s actionType_3' % str(i)] = user_actionType_cnt['windows %s actionType_3' % str(i)] * user_actionType_cnt.weight
    user_actionType_cnt['windows %s actionType_4' % str(i)] = user_actionType_cnt['windows %s actionType_4' % str(i)] * user_actionType_cnt.weight
    user_actionType_cnt['windows %s actionType_5' % str(i)] = user_actionType_cnt['windows %s actionType_5' % str(i)] * user_actionType_cnt.weight
    user_actionType_cnt['windows %s actionType_6' % str(i)] = user_actionType_cnt['windows %s actionType_6' % str(i)] * user_actionType_cnt.weight
    user_actionType_cnt['windows %s actionType_7' % str(i)] = user_actionType_cnt['windows %s actionType_7' % str(i)] * user_actionType_cnt.weight
    user_actionType_cnt['windows %s actionType_8' % str(i)] = user_actionType_cnt['windows %s actionType_8' % str(i)] * user_actionType_cnt.weight
    user_actionType_cnt['windows %s actionType_9' % str(i)] = user_actionType_cnt['windows %s actionType_9' % str(i)] * user_actionType_cnt.weight
    user_actionType_cnt = user_actionType_cnt.groupby('userid').sum()
    user_actionType_cnt['windows %s action cnt' % str(i)] = user_actionType_cnt.sum(axis = 1)
    for col in user_actionType_cnt.columns:
        if col != 'windows %s action cnt' % str(i):
            user_actionType_cnt[col + ' ratio'] = user_actionType_cnt[col] / user_actionType_cnt['windows %s action cnt' % str(i)]
    user_actionType_cnt['windows %s view cnt' % str(i)] = user_actionType_cnt['windows %s actionType_2' % str(i)] + \
                                                         user_actionType_cnt['windows %s actionType_3' % str(i)] + \
                                                         user_actionType_cnt['windows %s actionType_4' % str(i)]
    user_actionType_cnt['windows %s order cnt' % str(i)] = user_actionType_cnt['windows %s actionType_5' % str(i)] + \
                                                          user_actionType_cnt['windows %s actionType_6' % str(i)] + \
                                                          user_actionType_cnt['windows %s actionType_7' % str(i)] + \
                                                          user_actionType_cnt['windows %s actionType_8' % str(i)] + \
                                                          user_actionType_cnt['windows %s actionType_9' % str(i)]
    user_actionType_cnt['windows %s view ratio' % str(i)] = user_actionType_cnt['windows %s view cnt' % str(i)] / user_actionType_cnt['windows %s action cnt' % str(i)]
    user_actionType_cnt['windows %s order ratio' % str(i)] = user_actionType_cnt['windows %s order cnt' % str(i)] / user_actionType_cnt['windows %s action cnt' % str(i)]
    user_actionType_cnt['windows %s actionType_1 order ratio' % str(i)] = user_actionType_cnt['windows %s actionType_1' % str(i)] / user_actionType_cnt['windows %s order cnt' % str(i)]
    user_actionType_cnt.loc[user_actionType_cnt['windows %s order cnt' % str(i)] == 0, 'windows %s actionType_1 order ratio' % str(i)] = 0
    user_actionType_cnt['windows %s actionType_2 order ratio' % str(i)] = user_actionType_cnt['windows %s actionType_2' % str(i)] / user_actionType_cnt['windows %s order cnt' % str(i)]
    user_actionType_cnt.loc[user_actionType_cnt['windows %s order cnt' % str(i)] == 0, 'windows %s actionType_2 order ratio' % str(i)] = 0
    user_actionType_cnt['windows %s actionType_3 order ratio' % str(i)] = user_actionType_cnt['windows %s actionType_3' % str(i)] / user_actionType_cnt['windows %s order cnt' % str(i)]
    user_actionType_cnt.loc[user_actionType_cnt['windows %s order cnt' % str(i)] == 0, 'windows %s actionType_3 order ratio' % str(i)] = 0
    user_actionType_cnt['windows %s actionType_4 order ratio' % str(i)] = user_actionType_cnt['windows %s actionType_4' % str(i)] / user_actionType_cnt['windows %s order cnt' % str(i)]
    user_actionType_cnt.loc[user_actionType_cnt['windows %s order cnt' % str(i)] == 0, 'windows %s actionType_4 order ratio' % str(i)] = 0
    user_actionType_cnt['windows %s actionType_5 order ratio' % str(i)] = user_actionType_cnt['windows %s actionType_5' % str(i)] / user_actionType_cnt['windows %s order cnt' % str(i)]
    user_actionType_cnt.loc[user_actionType_cnt['windows %s order cnt' % str(i)] == 0, 'windows %s actionType_5 order ratio' % str(i)] = 0
    user_actionType_cnt['windows %s actionType_6 order ratio' % str(i)] = user_actionType_cnt['windows %s actionType_6' % str(i)] / user_actionType_cnt['windows %s order cnt' % str(i)]
    user_actionType_cnt.loc[user_actionType_cnt['windows %s order cnt' % str(i)] == 0, 'windows %s actionType_6 order ratio' % str(i)] = 0
    user_actionType_cnt['windows %s actionType_7 order ratio' % str(i)] = user_actionType_cnt['windows %s actionType_7' % str(i)] / user_actionType_cnt['windows %s order cnt' % str(i)]
    user_actionType_cnt.loc[user_actionType_cnt['windows %s order cnt' % str(i)] == 0, 'windows %s actionType_7 order ratio' % str(i)] = 0
    user_actionType_cnt['windows %s actionType_8 order ratio' % str(i)] = user_actionType_cnt['windows %s actionType_8' % str(i)] / user_actionType_cnt['windows %s order cnt' % str(i)]
    user_actionType_cnt.loc[user_actionType_cnt['windows %s order cnt' % str(i)] == 0, 'windows %s actionType_8 order ratio' % str(i)] = 0
    user_actionType_cnt['windows %s actionType_9 order ratio' % str(i)] = user_actionType_cnt['windows %s actionType_9' % str(i)] / user_actionType_cnt['windows %s order cnt' % str(i)]
    user_actionType_cnt.loc[user_actionType_cnt['windows %s order cnt' % str(i)] == 0, 'windows %s actionType_9 order ratio' % str(i)] = 0
    user_actionType_cnt.reset_index(inplace = True)
    ### 最后一天序列连续程度 ###
    userid = df_cl.userid.unique()
    seqentialratio_dict = {'userid': [],
                           '59_seqential_cnt': []}
    for uid in userid:
        action_df = df[df.userid == uid]
        action59seqentialratio = calc_seqentialratio(action_df)
        seqentialratio_dict['userid'].append(uid)
        seqentialratio_dict['59_seqential_cnt'].append(action59seqentialratio)
    seqentialratio_dict = pd.DataFrame(seqentialratio_dict)
    action_feats = pd.merge(user_actionType_cnt, seqentialratio_dict, on = 'userid', how = 'left')
    return action_feats

def is_actionType(x, k):
    if x == k:
        return 1
    else:
        return 0

###################################### feature engineering ###############################
# UserProfile
def gen_userProfile_feats():
    # 读取数据
    userProfile_train = pd.read_csv('trainingset/userProfile_train.csv')
    userProfile_test = pd.read_csv('test/userProfile_test.csv')
    df = pd.concat([userProfile_train, userProfile_test])  # type:pd.DataFrame
    ### 缺失值信息特征 ###
    # 是否有gender\province\age的信息
    df['has_gender_info'] = df.gender.apply(lambda x: is_not_nan(x))
    df['has_province_info'] = df.gender.apply(lambda x: is_not_nan(x))
    df['has_age_info'] = df.gender.apply(lambda x: is_not_nan(x))
    # 缺失信息个数
    nan_cnt = df.isnull().sum(axis = 1)
    df = pd.concat([df, nan_cnt], axis = 1) #type:pd.DataFrame
    df = df.rename(columns = {0: 'nan_cnt'})
    df['info_complete_ratio'] = 1 - df.nan_cnt / 3
    # 填补缺失值
    df.gender.fillna(u'未知', inplace = True)
    df.province.fillna(u'未知', inplace = True)
    df.age.fillna(u'未知', inplace = True)
    # 各性别类型历史购买精品服务比例、各年龄段历史购买精品服务比例、各省份历史购买精品服务比例
    orderHistory_train = pd.read_csv('trainingset/orderHistory_train.csv')
    orderHistory_test = pd.read_csv('test/orderHistory_test.csv')
    order_df = pd.concat([orderHistory_train, orderHistory_test], axis = 0)  # type:pd.DataFrame
    order_cnt = order_df.groupby(['userid', 'orderType']).size().unstack()
    order_cnt['order_cnt'] = order_cnt.sum(axis = 1)
    order_cnt.columns = ['normal_cnt', 'special_cnt', 'order_cnt']
    order_cnt.fillna(0, inplace = True)
    order_cnt.reset_index(inplace = True)
    order_cnt = pd.merge(order_cnt, df, on = 'userid')
    province_pos_ratio = order_cnt.groupby('province').sum()[['special_cnt', 'order_cnt']].reset_index()
    province_pos_ratio['province_pos_ratio'] = province_pos_ratio.special_cnt / province_pos_ratio.order_cnt
    province_pos_ratio = province_pos_ratio[['province', 'province_pos_ratio']]
    gender_pos_ratio = order_cnt.groupby('gender').sum()[['special_cnt', 'order_cnt']].reset_index()
    gender_pos_ratio['gender_pos_ratio'] = gender_pos_ratio.special_cnt / gender_pos_ratio.order_cnt
    gender_pos_ratio = gender_pos_ratio[['gender', 'gender_pos_ratio']]
    age_pos_ratio = order_cnt.groupby('age').sum()[['special_cnt', 'order_cnt']].reset_index()
    age_pos_ratio['age_pos_ratio'] = age_pos_ratio.special_cnt / age_pos_ratio.order_cnt
    age_pos_ratio = age_pos_ratio[['age', 'age_pos_ratio']]
    df = pd.merge(df, province_pos_ratio, on = 'province', how = 'left')
    df = pd.merge(df, gender_pos_ratio, on = 'gender', how = 'left')
    df = pd.merge(df, age_pos_ratio, on = 'age', how = 'left')
    # 对省份做LabelEncoder
    le = LabelEncoder()
    le.fit(df.province)
    df['province_labeled'] = le.fit_transform(df.province)
    # 取topN的省份
    df = cutting(df)
    # 哑变量
    for col in ['gender', 'province', 'age']:
        dummy = pd.get_dummies(df[col], prefix = col)
        df = pd.concat([df, dummy], axis = 1)  # type:pd.DataFrame
        del df[col]
    # 缓存
    pickle.dump(df, open(userProfile_dump_path, 'wb'))
    gc.collect()
    return df
# Action
def gen_action_feats():
    ## 读取数据 ###
    action_train = pd.read_csv('trainingset/action_train.csv')
    action_test = pd.read_csv('test/action_test.csv')
    df = pd.concat([action_train, action_test], axis = 0)  # type:pd.DataFrame
    df['actionTm'] = df.actionTime.apply(lambda x: timestamp_transform(x))
    df['actionTm'] = pd.to_datetime(df.actionTm)
    df['actionDt'] = df['actionTm'].dt.date
    ### 各actionType计数\比率 ###
    #  各actionType计数
    type_cnt = pd.get_dummies(df.actionType, prefix = 'actionType')
    type_cnt = pd.concat([df.userid, type_cnt], axis = 1)  # type:pd.DataFrame
    type_cnt = type_cnt.groupby('userid').sum()
    type_cnt['action_cnt'] = type_cnt.sum(axis = 1)
    type_cnt.reset_index(inplace = True)
    type_cnt['view'] = type_cnt['actionType_2'] + type_cnt['actionType_3'] + type_cnt['actionType_4']
    type_cnt['order'] = type_cnt['actionType_5'] + type_cnt['actionType_6'] + type_cnt['actionType_7'] + type_cnt['actionType_8'] + type_cnt['actionType_9']
    #  各actionType比率
    for col in type_cnt.columns:
        if col not in ['userid', 'action_cnt']:
            type_cnt[col + '_ctr'] = type_cnt[col] / type_cnt.action_cnt
    type_cnt['actionType1_order_ratio'] = type_cnt.actionType_1 / (type_cnt.order + eta)
    type_cnt['actionType5_order_ratio'] = type_cnt.actionType_5 / (type_cnt.order + eta)
    type_cnt['actionType6_order_ratio'] = type_cnt.actionType_6 / (type_cnt.order + eta)
    type_cnt['actionType7_order_ratio'] = type_cnt.actionType_7 / (type_cnt.order + eta)
    type_cnt['actionType8_order_ratio'] = type_cnt.actionType_8 / (type_cnt.order + eta)
    type_cnt['actionType9_order_ratio'] = type_cnt.actionType_9 / (type_cnt.order + eta)
    # 组合特征
    type_cnt['actionType5_actionType6'] = type_cnt.actionType_5 - type_cnt.actionType_6
    type_cnt['actionType6_actionType7'] = type_cnt.actionType_6 - type_cnt.actionType_7
    type_cnt['actionType7_actionType8'] = type_cnt.actionType_7 - type_cnt.actionType_8
    type_cnt['actionType8_actionType9'] = type_cnt.actionType_8 - type_cnt.actionType_9
    ### 用户活跃天数 ###
    # 用户活跃天数
    active_days = df.groupby('userid').nunique()['actionDt'].reset_index()
    active_days = active_days.rename(columns = {'actionDt': 'active_days'})
    action_feats = pd.merge(type_cnt, active_days, on = 'userid', how = 'left')
    del active_days
    del type_cnt
    # 用户有2\3\4\5\6\7\8的天数\比率
    actionType_days = df[df.actionType.isin([2, 3, 4, 5, 6, 7, 8])].groupby(['userid', 'actionType']).nunique()['actionDt'].unstack()
    actionType_days.fillna(0, inplace = True)
    actionType_days.columns = ['actionType2_days', 'actionType3_days', 'actionType4_days', 'actionType5_days',
                               'actionType6_days', 'actionType7_days', 'actionType8_days']
    actionType_days.reset_index(inplace = True)
    actionType_days = pd.merge(actionType_days, action_feats[['userid', 'active_days']], on = 'userid', how = 'left')
    for col in actionType_days.columns:
        if col not in ['userid', 'active_days']:
            actionType_days[col + '_ratio'] = actionType_days[col] / actionType_days.active_days
    del actionType_days['active_days']
    action_feats = pd.merge(action_feats, actionType_days, on = 'userid', how = 'left')
    action_feats['action_per_date'] = action_feats.action_cnt / action_feats.active_days
    del actionType_days
    ### 用户最近的actionType和actionTime ###
    # 用户第一个actionType,actionTime
    first_actionType = df.groupby('userid').actionTime.nsmallest(1).reset_index()
    first_actionType = pd.merge(first_actionType,
                                df[['userid', 'actionTime', 'actionType', 'actionTm']],
                                on = ['userid', 'actionTime'], how = 'left')
    del first_actionType['level_1']
    first_actionType.columns = ['userid', 'first_actionTime_1', 'first_actionType_1', 'first_actionTm_1']
    action_feats = pd.merge(action_feats, first_actionType, on = 'userid', how = 'left')
    del first_actionType
    # 用户最近K个actionType,actionTime
    latest_actionType = df.groupby('userid').actionTime.nlargest(7).reset_index()
    latest_actionType = pd.merge(latest_actionType, df[['userid', 'actionTime', 'actionType', 'actionTm']],
                                 on = ['userid', 'actionTime'], how = 'left')
    del latest_actionType['level_1']
    latest_actionType_1 = latest_actionType.groupby('userid').nth(0)[['actionType', 'actionTime', 'actionTm']].reset_index()
    latest_actionType_1.columns = ['userid', 'latest_actionType_1', 'latest_actionType_1_tm', 'latest_actionTm_1']
    latest_actionType_2 = latest_actionType.groupby('userid').nth(1)[['actionType', 'actionTime', 'actionTm']].reset_index()
    latest_actionType_2.columns = ['userid', 'latest_actionType_2', 'latest_actionType_2_tm', 'latest_actionTm_2']
    latest_actionType_3 = latest_actionType.groupby('userid').nth(2)[['actionType', 'actionTime', 'actionTm']].reset_index()
    latest_actionType_3.columns = ['userid', 'latest_actionType_3', 'latest_actionType_3_tm', 'latest_actionTm_3']
    latest_actionType_4 = latest_actionType.groupby('userid').nth(3)[['actionType', 'actionTime', 'actionTm']].reset_index()
    latest_actionType_4.columns = ['userid', 'latest_actionType_4', 'latest_actionType_4_tm', 'latest_actionTm_4']
    latest_actionType_5 = latest_actionType.groupby('userid').nth(4)[['actionType', 'actionTime', 'actionTm']].reset_index()
    latest_actionType_5.columns = ['userid', 'latest_actionType_5', 'latest_actionType_5_tm', 'latest_actionTm_5']
    latest_actionType_6 = latest_actionType.groupby('userid').nth(5)[['actionType', 'actionTime', 'actionTm']].reset_index()
    latest_actionType_6.columns = ['userid', 'latest_actionType_6', 'latest_actionType_6_tm', 'latest_actionTm_6']
    latest_actionType_7 = latest_actionType.groupby('userid').nth(6)[['actionType', 'actionTime', 'actionTm']].reset_index()
    latest_actionType_7.columns = ['userid', 'latest_actionType_7', 'latest_actionType_7_tm', 'latest_actionTm_7']
    action_feats = pd.merge(action_feats, latest_actionType_1, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_2, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_3, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_4, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_5, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_6, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_7, on = 'userid', how = 'left')
    del latest_actionType_1
    del latest_actionType_2
    del latest_actionType_3
    del latest_actionType_4
    del latest_actionType_5
    del latest_actionType_6
    del latest_actionType_7
    action_feats['action_tm_range'] = action_feats.latest_actionType_1_tm - action_feats.first_actionTime_1
    action_feats['action_tm_days'] = action_feats.latest_actionTm_1 - action_feats.first_actionTm_1
    action_feats['action_tm_days'] = action_feats['action_tm_days'].dt.days
    action_feats['action_tm_range_2'] = action_feats.latest_actionType_2_tm - action_feats.first_actionTime_1
    action_feats['action_tm_days_2'] = action_feats.latest_actionTm_2 - action_feats.first_actionTm_1
    action_feats['action_tm_days_2'] = action_feats['action_tm_days_2'].dt.days
    action_feats['action_tm_range_3'] = action_feats.latest_actionType_3_tm - action_feats.first_actionTime_1
    action_feats['action_tm_days_3'] = action_feats.latest_actionTm_3 - action_feats.first_actionTm_1
    action_feats['action_tm_days_3'] = action_feats['action_tm_days_3'].dt.days
    action_feats['action_tm_range_4'] = action_feats.latest_actionType_4_tm - action_feats.first_actionTime_1
    action_feats['action_tm_range_5'] = action_feats.latest_actionType_5_tm - action_feats.first_actionTime_1
    action_feats['action_tm_range_6'] = action_feats.latest_actionType_6_tm - action_feats.first_actionTime_1
    action_feats['action_tm_range_7'] = action_feats.latest_actionType_7_tm - action_feats.first_actionTime_1
    action_feats['latest_actionType_mutiply'] = action_feats.latest_actionType_1 * action_feats.latest_actionType_2 * action_feats.latest_actionType_3
    action_feats['latest_actionType_mean'] = (action_feats.latest_actionType_1 + action_feats.latest_actionType_2 + action_feats.latest_actionType_3) / 3
    action_feats.drop(['first_actionTm_1', 'latest_actionTm_1', 'latest_actionTm_2', 'latest_actionTm_3',
                       'latest_actionTm_4', 'latest_actionTm_5', 'latest_actionTm_6', 'latest_actionTm_7'],
                      axis = 1,
                      inplace = True)
    gc.collect()
    ## 用户actionType时间差均值、方差、最大值、最小值、中位数、偏度 ###
    # 用户所有的actionType(timespan_thredthold用来限制时间间隔在1天以内)
    user_actionTime_seq = df.groupby('userid').apply(lambda x: x['actionTime'].tolist()).reset_index()
    user_actionTime_seq.columns = ['userid', 'actionTime_seq']
    user_actionTime_seq['actionTime_seq'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.diff(x))
    user_actionTime_seq['actionTime_seq'] = user_actionTime_seq.actionTime_seq.apply(lambda x: timespan_thredthold(x))
    user_actionTime_seq['actionTime_diff_mean'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.mean(x))
    user_actionTime_seq['actionTime_diff_var'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.var(x))
    user_actionTime_seq['actionTime_diff_max'] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_tm_diff_max(x))
    user_actionTime_seq['actionTime_diff_min'] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_tm_diff_min(x))
    user_actionTime_seq['actionTime_diff_median'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.median(x))
    user_actionTime_seq['actionTime_diff_skew'] = user_actionTime_seq.actionTime_seq.apply(lambda x: skew(x))
    user_actionTime_seq['actionTime_diff_sum'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.sum(x))
    user_actionTime_seq['latest_actionTime_diff_1'] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_k_index(x, -1))
    user_actionTime_seq['latest_actionTime_diff_2'] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_k_index(x, -2))
    user_actionTime_seq['latest_actionTime_diff_3'] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_k_index(x, -3))
    user_actionTime_seq['latest_actionTime_diff_4'] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_k_index(x, -4))
    user_actionTime_seq['latest_actionTime_diff_5'] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_k_index(x, -5))
    user_actionTime_seq['latest_actionTime_diff_mean'] = (user_actionTime_seq.latest_actionTime_diff_1 + user_actionTime_seq.latest_actionTime_diff_2 + user_actionTime_seq.latest_actionTime_diff_3)/3
    user_actionTime_seq['latest_actionTime_diff_multiply'] = user_actionTime_seq.latest_actionTime_diff_1 * user_actionTime_seq.latest_actionTime_diff_2 * user_actionTime_seq.latest_actionTime_diff_3
    print(user_actionTime_seq.head())
    del user_actionTime_seq['actionTime_seq']
    action_feats = pd.merge(action_feats, user_actionTime_seq, on = 'userid', how = 'left')
    del user_actionTime_seq
    # 用户单独actionType5\6\7\8\9
    for i in [1, 5, 6, 7, 8, 9]:
        user_actionTime_seq = df[df.actionType == i].groupby('userid').apply(lambda x: x['actionTime'].tolist()).reset_index()
        user_actionTime_seq.columns = ['userid', 'actionTime_seq']
        user_actionTime_seq['actionTime_seq'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.diff(x))
        user_actionTime_seq['actionType%s_actionTime_diff_mean' % str(i)] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.mean(x))
        user_actionTime_seq['actionType%s_actionTime_diff_var' % str(i)] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.var(x))
        user_actionTime_seq['actionType%s_actionTime_diff_max' % str(i)] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_tm_diff_max(x))
        user_actionTime_seq['actionType%s_actionTime_diff_min' % str(i)] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_tm_diff_min(x))
        user_actionTime_seq['actionType%s_actionTime_diff_median' % str(i)] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.median(x))
        user_actionTime_seq['actionType%s_actionTime_diff_skew' % str(i)] = user_actionTime_seq.actionTime_seq.apply(lambda x: skew(x))
        user_actionTime_seq['latest_actionType%s_actionTime_diff_1' % str(i)] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_k_index(x, -1))
        user_actionTime_seq['latest_actionType%s_actionTime_diff_2' % str(i)] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_k_index(x, -2))
        del user_actionTime_seq['actionTime_seq']
        action_feats = pd.merge(action_feats, user_actionTime_seq, on = 'userid', how = 'left')
    ## actionType5和6之间的时间差 ###
    user_actionTime_seq = df[df.actionType.isin([5, 6])].groupby('userid').apply(lambda x: x['actionTime'].tolist()).reset_index()
    user_actionTime_seq.columns = ['userid', 'actionTime_seq']
    user_actionTime_seq['actionTime_seq'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.diff(x))
    user_actionTime_seq['actionTime_seq'] = user_actionTime_seq.actionTime_seq.apply(lambda x: timespan_thredthold(x))
    user_actionTime_seq['actionType5to6_actionTime_diff_mean'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.mean(x))
    user_actionTime_seq['actionType5to6_actionTime_diff_var'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.var(x))
    user_actionTime_seq['actionType5to6_actionTime_diff_max'] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_tm_diff_max(x))
    user_actionTime_seq['actionType5to6_actionTime_diff_min'] = user_actionTime_seq.actionTime_seq.apply(lambda x: get_tm_diff_min(x))
    user_actionTime_seq['actionType5to6_actionTime_diff_median'] = user_actionTime_seq.actionTime_seq.apply(lambda x: np.median(x))
    user_actionTime_seq['actionType5to6_actionTime_diff_skew'] = user_actionTime_seq.actionTime_seq.apply(lambda x: skew(x))
    del user_actionTime_seq['actionTime_seq']
    action_feats = pd.merge(action_feats, user_actionTime_seq, on = 'userid', how = 'left')
    del user_actionTime_seq
    gc.collect()
    # 行为距离特征、行为序列特征
    userid_list = df.userid.unique()
    act_range_dict = {'userid': [],
                      'latest_actionType9_act_range': [],
                      'latest_actionType8_act_range': [],
                      'latest_actionType7_act_range': [],
                      'latest_actionType6_act_range': [],
                      'latest_actionType5_act_range': [],
                      'latest_actionType4_act_range': [],
                      'latest_actionType3_act_range': [],
                      'latest_actionType2_act_range': [],
                      'latest_actionType1_act_range': [],
                      '56_seq_cnt': [],
                      '567_seq_cnt': [],
                      '5678_seq_cnt': [],
                      '56789_seq_cnt': [],
                      '56_seq_ratio': [],
                      '567_seq_ratio': [],
                      '5678_seq_ratio': [],
                      '56789_seq_ratio': [],
					  '678_seq_cnt': [],
					  '678_seq_ratio': [],
					  '789_seq_cnt': [],
					  '789_seq_ratio': []
                      }
    for id in userid_list:
        user_action_record = df[df.userid == id]
        user_act_list = []
        act_range_dict['userid'].append(id)
        for index, row in user_action_record.iterrows():
            user_act_list.append(row['actionType'])
        user_act_list = pd.Series(user_act_list)
        n = len(user_act_list)
        try:
            act_range_dict['latest_actionType9_act_range'].append(n - max(user_act_list[user_act_list == 9].index))
        except:
            act_range_dict['latest_actionType9_act_range'].append(np.nan)
        try:
            act_range_dict['latest_actionType8_act_range'].append(n - max(user_act_list[user_act_list == 8].index))
        except:
            act_range_dict['latest_actionType8_act_range'].append(np.nan)
        try:
            act_range_dict['latest_actionType7_act_range'].append(n - max(user_act_list[user_act_list == 7].index))
        except:
            act_range_dict['latest_actionType7_act_range'].append(np.nan)
        try:
            act_range_dict['latest_actionType6_act_range'].append(n - max(user_act_list[user_act_list == 6].index))
        except:
            act_range_dict['latest_actionType6_act_range'].append(np.nan)
        try:
            act_range_dict['latest_actionType5_act_range'].append(n - max(user_act_list[user_act_list == 5].index))
        except:
            act_range_dict['latest_actionType5_act_range'].append(np.nan)
        try:
            act_range_dict['latest_actionType4_act_range'].append(n - max(user_act_list[user_act_list == 4].index))
        except:
            act_range_dict['latest_actionType4_act_range'].append(np.nan)
        try:
            act_range_dict['latest_actionType3_act_range'].append(n - max(user_act_list[user_act_list == 3].index))
        except:
            act_range_dict['latest_actionType3_act_range'].append(np.nan)
        try:
            act_range_dict['latest_actionType2_act_range'].append(n - max(user_act_list[user_act_list == 2].index))
        except:
            act_range_dict['latest_actionType2_act_range'].append(np.nan)
        try:
            act_range_dict['latest_actionType1_act_range'].append(n - max(user_act_list[user_act_list == 1].index))
        except:
            act_range_dict['latest_actionType1_act_range'].append(np.nan)
        user_act_seq = ''
        n_seq = len(user_act_list)
        for act in user_act_list:
            user_act_seq = user_act_seq + str(act) + '-'
        try:
            act_range_dict['56_seq_cnt'].append(len(re.findall('5-6', user_act_seq)))
            act_range_dict['56_seq_ratio'].append(len(re.findall('5-6', user_act_seq)) / n_seq)
        except:
            act_range_dict['56_seq_cnt'].append(0)
            act_range_dict['56_seq_ratio'].append(0)
        try:
            act_range_dict['567_seq_cnt'].append(len(re.findall('5-6-7', user_act_seq)))
            act_range_dict['567_seq_ratio'].append(len(re.findall('5-6-7', user_act_seq)) / n_seq)
        except:
            act_range_dict['567_seq_cnt'].append(0)
            act_range_dict['567_seq_ratio'].append(0)
        try:
            act_range_dict['5678_seq_cnt'].append(len(re.findall('5-6-7-8', user_act_seq)))
            act_range_dict['5678_seq_ratio'].append(len(re.findall('5-6-7-8-9', user_act_seq)) / n_seq)
        except:
            act_range_dict['5678_seq_cnt'].append(0)
            act_range_dict['5678_seq_ratio'].append(0)
        try:
            act_range_dict['56789_seq_cnt'].append(len(re.findall('5-6-7-8-9', user_act_seq)))
            act_range_dict['56789_seq_ratio'].append(len(re.findall('5-6-7-8-9', user_act_seq)) / n_seq)
        except:
            act_range_dict['56789_seq_cnt'].append(0)
            act_range_dict['56789_seq_ratio'].append(0)
        try:
            act_range_dict['678_seq_cnt'].append(len(re.findall('6-7-8', user_act_seq)))
            act_range_dict['678_seq_ratio'].append(len(re.findall('6-7-8', user_act_seq)) / n_seq)
        except:
            act_range_dict['678_seq_cnt'].append(0)
            act_range_dict['678_seq_ratio'].append(0)
        try:
            act_range_dict['789_seq_cnt'].append(len(re.findall('7-8-9', user_act_seq)))
            act_range_dict['789_seq_ratio'].append(len(re.findall('7-8-9', user_act_seq)) / n_seq)
        except:
            act_range_dict['789_seq_cnt'].append(0)
            act_range_dict['789_seq_ratio'].append(0)
    act_range_dict = pd.DataFrame(act_range_dict)
    action_feats = pd.merge(action_feats, act_range_dict, on = 'userid', how = 'left')
    del act_range_dict
    gc.collect()
    # 用户每个actionType最初出现的时间
    first_actionType_tm_1 = get_actionType_first_actionTime(df, 1)
    first_actionType_tm_2 = get_actionType_first_actionTime(df, 2)
    first_actionType_tm_3 = get_actionType_first_actionTime(df, 3)
    first_actionType_tm_4 = get_actionType_first_actionTime(df, 4)
    first_actionType_tm_5 = get_actionType_first_actionTime(df, 5)
    first_actionType_tm_6 = get_actionType_first_actionTime(df, 6)
    first_actionType_tm_7 = get_actionType_first_actionTime(df, 7)
    first_actionType_tm_8 = get_actionType_first_actionTime(df, 8)
    first_actionType_tm_9 = get_actionType_first_actionTime(df, 9)
    # 用户每个actionType最近出现的时间
    latest_actionType_tm_1 = get_actionType_latest_actionTime(df, 1)
    latest_actionType_tm_2 = get_actionType_latest_actionTime(df, 2)
    latest_actionType_tm_3 = get_actionType_latest_actionTime(df, 3)
    latest_actionType_tm_4 = get_actionType_latest_actionTime(df, 4)
    latest_actionType_tm_5 = get_actionType_latest_actionTime(df, 5)
    latest_actionType_tm_6 = get_actionType_latest_actionTime(df, 6)
    latest_actionType_tm_7 = get_actionType_latest_actionTime(df, 7)
    latest_actionType_tm_8 = get_actionType_latest_actionTime(df, 8)
    latest_actionType_tm_9 = get_actionType_latest_actionTime(df, 9)
    # actionA到actionB之间时间间隔小于threthold的个数
    userid = df.userid.unique()
    timespancount_dict = {'userid': [],
                          'actiontimespancount_1_5': [],
                          'actiontimespancount_5_6': [],
                          'actiontimespancount_6_7': [],
                          'actiontimespancount_7_8': [],
                          'actiontimespancount_8_9': []}
    for uid in userid:
        action_df = df[df.userid == uid]
        actiontimespancount_1_5 = getActionTimeSpan(action_df, 1, 5, timethred = 100)
        actiontimespancount_5_6 = getActionTimeSpan(action_df, 5, 6, timethred = 100)
        actiontimespancount_6_7 = getActionTimeSpan(action_df, 6, 7, timethred = 100)
        actiontimespancount_7_8 = getActionTimeSpan(action_df, 7, 8, timethred = 100)
        actiontimespancount_8_9 = getActionTimeSpan(action_df, 8, 9, timethred = 100)
        timespancount_dict['userid'].append(uid)
        timespancount_dict['actiontimespancount_1_5'].append(actiontimespancount_1_5)
        timespancount_dict['actiontimespancount_5_6'].append(actiontimespancount_5_6)
        timespancount_dict['actiontimespancount_6_7'].append(actiontimespancount_6_7)
        timespancount_dict['actiontimespancount_7_8'].append(actiontimespancount_7_8)
        timespancount_dict['actiontimespancount_8_9'].append(actiontimespancount_8_9)
    timespancount_dict = pd.DataFrame(timespancount_dict)
   # action的tfidf
    df['actionType_trans'] = df.actionType.replace({1: 'one',
                                                    2: 'two',
                                                    3: 'three',
                                                    4: 'four',
                                                    5: 'five',
                                                    6: 'six',
                                                    7: 'seven',
                                                    8: 'eight',
                                                    9: 'nine'})
    user_actionType_seq = df.groupby('userid').apply(lambda x: x['actionType_trans'].tolist()).reset_index()
    user_actionType_seq.columns = ['userid', 'act_seq']
    user_actionType_seq['act_seq'] = user_actionType_seq.act_seq.astype(str)
    tv = TfidfVectorizer(ngram_range = (1, 2))
    act_tfidf = tv.fit_transform(user_actionType_seq['act_seq'])
    act_tfidf = pd.DataFrame(act_tfidf.toarray())
    act_tfidf.columns = tv.get_feature_names()
    user_act_tfidf = pd.concat([user_actionType_seq, act_tfidf], axis = 1)  # type:pd.DataFrame
    del user_act_tfidf['act_seq']
    print(user_act_tfidf.head())
    # 快速傅里叶变换、小波变换
    user_actionType_seq['act_seq_fft'] = user_actionType_seq.act_seq.apply(lambda x: fft(x).real)
    user_actionType_seq['fft_1'] = user_actionType_seq.act_seq_fft.apply(lambda x: get_k_index(x, 0))
    user_actionType_seq['fft_2'] = user_actionType_seq.act_seq_fft.apply(lambda x: get_k_index(x, 1))
    user_actionType_seq['fft_3'] = user_actionType_seq.act_seq_fft.apply(lambda x: get_k_index(x, 2))
    user_actionType_seq['act_seq_wavelet_1'] = user_actionType_seq.act_seq.apply(lambda x: get_last_wave_feature_dwt_cd(0, 1, x))
    user_actionType_seq['act_seq_wavelet_2'] = user_actionType_seq.act_seq.apply(lambda x: get_last_wave_feature_dwt_cd(0, 2, x))
    user_actionType_seq['act_seq_wavelet_3'] = user_actionType_seq.act_seq.apply(lambda x: get_last_wave_feature_dwt_cd(0, 3, x))
    user_actionType_seq['act_seq_wavelet_4'] = user_actionType_seq.act_seq.apply(lambda x: get_last_wave_feature_dwt_cd(1, 1, x))
    user_actionType_seq['act_seq_wavelet_5'] = user_actionType_seq.act_seq.apply(lambda x: get_last_wave_feature_dwt_cd(1, 2, x))
    user_actionType_seq['act_seq_wavelet_6'] = user_actionType_seq.act_seq.apply(lambda x: get_last_wave_feature_dwt_cd(1, 3, x))
    del user_actionType_seq['act_seq']
    # 合并特征
    action_feats = pd.merge(action_feats, latest_actionType_tm_1, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_tm_2, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_tm_3, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_tm_4, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_tm_5, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_tm_6, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_tm_7, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_tm_8, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, latest_actionType_tm_9, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, first_actionType_tm_1, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, first_actionType_tm_2, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, first_actionType_tm_3, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, first_actionType_tm_4, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, first_actionType_tm_5, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, first_actionType_tm_6, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, first_actionType_tm_7, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, first_actionType_tm_8, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, first_actionType_tm_9, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, timespancount_dict, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, user_act_tfidf, on = 'userid', how = 'left')
    action_feats = pd.merge(action_feats, user_actionType_seq, on = 'userid', how = 'left')
    # 用户到最近一次actionTime的时间差特征(每个actionType)
    for act in [2, 3, 4, 5, 6, 7, 8, 9]:
        action_feats = pd.merge(action_feats, gen_actionTypek_tm_diff_feats(act), on = 'userid', how = 'left')
    # 组合特征
    action_feats['action_tm_range'] = action_feats.latest_actionType_1_tm - action_feats.first_actionTime_1
    action_feats['latest_actionType5_6_tm_range'] = action_feats.latest_actionType6_tm - action_feats.latest_actionType5_tm
    action_feats['latest_actionType5_7_tm_range'] = action_feats.latest_actionType7_tm - action_feats.latest_actionType5_tm
    action_feats['latest_actionType5_8_tm_range'] = action_feats.latest_actionType8_tm - action_feats.latest_actionType5_tm
    action_feats['latest_actionType5_9_tm_range'] = action_feats.latest_actionType9_tm - action_feats.latest_actionType5_tm
    action_feats['latest_actionType6_7_tm_range'] = action_feats.latest_actionType7_tm - action_feats.latest_actionType6_tm
    action_feats['latest_actionType7_8_tm_range'] = action_feats.latest_actionType8_tm - action_feats.latest_actionType7_tm
    action_feats['latest_actionType8_9_tm_range'] = action_feats.latest_actionType9_tm - action_feats.latest_actionType8_tm
    ### 最近的actionType1\2\3\4\5\6\7\8\9到最近一个actioTyep的时间间隔 ###
    action_feats['latest_actionType1_to_maxtime'] = action_feats.latest_actionType_1_tm - action_feats.latest_actionType1_tm
    action_feats['latest_actionType2_to_maxtime'] = action_feats.latest_actionType_1_tm - action_feats.latest_actionType2_tm
    action_feats['latest_actionType3_to_maxtime'] = action_feats.latest_actionType_1_tm - action_feats.latest_actionType3_tm
    action_feats['latest_actionType4_to_maxtime'] = action_feats.latest_actionType_1_tm - action_feats.latest_actionType4_tm
    action_feats['latest_actionType5_to_maxtime'] = action_feats.latest_actionType_1_tm - action_feats.latest_actionType5_tm
    action_feats['latest_actionType6_to_maxtime'] = action_feats.latest_actionType_1_tm - action_feats.latest_actionType6_tm
    action_feats['latest_actionType7_to_maxtime'] = action_feats.latest_actionType_1_tm - action_feats.latest_actionType7_tm
    action_feats['latest_actionType8_to_maxtime'] = action_feats.latest_actionType_1_tm - action_feats.latest_actionType8_tm
    action_feats['latest_actionType9_to_maxtime'] = action_feats.latest_actionType_1_tm - action_feats.latest_actionType9_tm
    ### 最近的actionType1\2\3\4\5\6\7\8\9到倒数第二个actioTyep的时间间隔 ###
    action_feats['latest_actionType1_to_maxtime2'] = action_feats.latest_actionType_2_tm - action_feats.latest_actionType1_tm
    action_feats['latest_actionType2_to_maxtime2'] = action_feats.latest_actionType_2_tm - action_feats.latest_actionType2_tm
    action_feats['latest_actionType3_to_maxtime2'] = action_feats.latest_actionType_2_tm - action_feats.latest_actionType3_tm
    action_feats['latest_actionType4_to_maxtime2'] = action_feats.latest_actionType_2_tm - action_feats.latest_actionType4_tm
    action_feats['latest_actionType5_to_maxtime2'] = action_feats.latest_actionType_2_tm - action_feats.latest_actionType5_tm
    action_feats['latest_actionType6_to_maxtime2'] = action_feats.latest_actionType_2_tm - action_feats.latest_actionType6_tm
    action_feats['latest_actionType7_to_maxtime2'] = action_feats.latest_actionType_2_tm - action_feats.latest_actionType7_tm
    action_feats['latest_actionType8_to_maxtime2'] = action_feats.latest_actionType_2_tm - action_feats.latest_actionType8_tm
    action_feats['latest_actionType9_to_maxtime2'] = action_feats.latest_actionType_2_tm - action_feats.latest_actionType9_tm
    ### 最近的actionType1\2\3\4\5\6\7\8\9到倒数第三个actioTyep的时间间隔 ###
    action_feats['latest_actionType1_to_maxtime3'] = action_feats.latest_actionType_3_tm - action_feats.latest_actionType1_tm
    action_feats['latest_actionType2_to_maxtime3'] = action_feats.latest_actionType_3_tm - action_feats.latest_actionType2_tm
    action_feats['latest_actionType3_to_maxtime3'] = action_feats.latest_actionType_3_tm - action_feats.latest_actionType3_tm
    action_feats['latest_actionType4_to_maxtime3'] = action_feats.latest_actionType_3_tm - action_feats.latest_actionType4_tm
    action_feats['latest_actionType5_to_maxtime3'] = action_feats.latest_actionType_3_tm - action_feats.latest_actionType5_tm
    action_feats['latest_actionType6_to_maxtime3'] = action_feats.latest_actionType_3_tm - action_feats.latest_actionType6_tm
    action_feats['latest_actionType7_to_maxtime3'] = action_feats.latest_actionType_3_tm - action_feats.latest_actionType7_tm
    action_feats['latest_actionType8_to_maxtime3'] = action_feats.latest_actionType_3_tm - action_feats.latest_actionType8_tm
    action_feats['latest_actionType9_to_maxtime3'] = action_feats.latest_actionType_3_tm - action_feats.latest_actionType9_tm
    ### 最近的actionTypeK到最初的actionTypeK的时间间隔 ###
    action_feats['latest_actionType1_to_first_actionType1'] = action_feats.latest_actionType1_tm - action_feats.first_actionType1_tm
    action_feats['latest_actionType2_to_first_actionType2'] = action_feats.latest_actionType2_tm - action_feats.first_actionType2_tm
    action_feats['latest_actionType3_to_first_actionType3'] = action_feats.latest_actionType3_tm - action_feats.first_actionType3_tm
    action_feats['latest_actionType4_to_first_actionType4'] = action_feats.latest_actionType4_tm - action_feats.first_actionType4_tm
    action_feats['latest_actionType5_to_first_actionType5'] = action_feats.latest_actionType5_tm - action_feats.first_actionType5_tm
    action_feats['latest_actionType6_to_first_actionType6'] = action_feats.latest_actionType6_tm - action_feats.first_actionType6_tm
    action_feats['latest_actionType7_to_first_actionType7'] = action_feats.latest_actionType7_tm - action_feats.first_actionType7_tm
    action_feats['latest_actionType8_to_first_actionType8'] = action_feats.latest_actionType8_tm - action_feats.first_actionType8_tm
    action_feats['latest_actionType9_to_first_actionType9'] = action_feats.latest_actionType9_tm - action_feats.first_actionType9_tm
    # 缓存
    pickle.dump(action_feats, open(Action_dump_path, 'wb'))
    gc.collect()
    return action_feats
# OrderHistory
def gen_orderHistory_feats():
    # 读取数据
    orderHistory_train = pd.read_csv('trainingset/orderHistory_train.csv')
    orderHistory_test = pd.read_csv('test/orderHistory_test.csv')
    df = pd.concat([orderHistory_train, orderHistory_test], axis = 0) #type:pd.DataFrame
    # 用户历史普通/精品订单数及其比例
    order_cnt = df.groupby(['userid', 'orderType']).size().unstack()
    order_cnt['order_cnt'] = order_cnt.sum(axis = 1)
    order_cnt.columns = ['normal_cnt', 'special_cnt', 'order_cnt']
    order_cnt.fillna(0, inplace = True)
    order_cnt.reset_index(inplace = True)
    order_cnt['normal_ratio'] = order_cnt.normal_cnt / order_cnt.order_cnt
    order_cnt['special_ratio'] = order_cnt.special_cnt / order_cnt.order_cnt
    # 用户购买过的城市个数、国家个数、大洲个数
    city_cnt = df.groupby('userid').nunique()[['city', 'country', 'continent']]
    city_cnt.columns = ['city_cnt', 'country_cnt', 'continent_cnt']
    city_cnt.reset_index(inplace = True)
    # 最近一次购买时间、最近一次购买服务种类、最近一次购买精品服务时间
    order_tm = df.sort_values(by = ['userid', 'orderTime', 'orderType'])
    latest_order_tm = order_tm.groupby('userid').last()[['orderTime', 'orderType']].reset_index()
    latest_order_tm.columns = ['userid', 'latest_orderTime', 'latest_orderType']
    latest_special_order_tm = order_tm[order_tm.orderType == 1]
    latest_special_order_tm = latest_special_order_tm.groupby('userid').last()['orderTime'].reset_index()
    latest_special_order_tm.columns = ['userid', 'latest_special_orderTime']
    # 每个城市精品服务比例
    city_spe_ratio = df.groupby(['city', 'orderType']).size().unstack()
    city_spe_ratio.fillna(0, inplace = True)
    city_spe_ratio.columns = ['normal', 'special']
    city_spe_ratio['city_special_ratio'] = city_spe_ratio.special / (city_spe_ratio.normal + city_spe_ratio.special)
    city_spe_ratio.reset_index(inplace = True)
    # 每个国家精品服务比例
    country_spe_ratio = df.groupby(['country', 'orderType']).size().unstack()
    country_spe_ratio.fillna(0, inplace = True)
    country_spe_ratio.columns = ['normal', 'special']
    country_spe_ratio['country_special_ratio'] = country_spe_ratio.special / (country_spe_ratio.normal + country_spe_ratio.special)
    country_spe_ratio.reset_index(inplace = True)
    # 每个大洲精品服务比例
    continent_spe_ratio = df.groupby(['continent', 'orderType']).size().unstack()
    continent_spe_ratio.fillna(0, inplace = True)
    continent_spe_ratio.columns = ['normal', 'special']
    continent_spe_ratio['continent_special_ratio'] = continent_spe_ratio.special / (continent_spe_ratio.normal + continent_spe_ratio.special)
    continent_spe_ratio.reset_index(inplace = True)
    # 用户历史去过的城市、国家、大洲精品服务比例和
    user_spe_degree = pd.merge(df, city_spe_ratio[['city', 'city_special_ratio']], on = 'city', how = 'left')
    user_spe_degree = pd.merge(user_spe_degree, country_spe_ratio[['country', 'country_special_ratio']], on = 'country', how = 'left')
    user_spe_degree = pd.merge(user_spe_degree, continent_spe_ratio[['continent', 'continent_special_ratio']], on = 'continent', how = 'left')
    user_spe_degree['user_spe_degree'] = user_spe_degree.city_special_ratio * user_spe_degree.country_special_ratio * user_spe_degree.continent_special_ratio
    user_spe_degree_sum = user_spe_degree.groupby('userid').sum()[['city_special_ratio', 'country_special_ratio', 'continent_special_ratio', 'user_spe_degree']].reset_index()
    user_spe_degree_sum.columns = ['userid', 'city_special_ratio_sum', 'country_special_ratio_sum', 'continent_special_ratio_sum', 'user_spe_degree_sum']
    user_spe_degree_mean = user_spe_degree.groupby('userid').mean()[['city_special_ratio', 'country_special_ratio', 'continent_special_ratio', 'user_spe_degree']].reset_index()
    user_spe_degree_mean.columns = ['userid', 'city_special_ratio_mean', 'country_special_ratio_mean', 'continent_special_ratio_mean', 'user_spe_degree_mean']
    user_spe_degree_max = user_spe_degree.groupby('userid').max()[['city_special_ratio', 'country_special_ratio', 'continent_special_ratio', 'user_spe_degree']].reset_index()
    user_spe_degree_max.columns = ['userid', 'city_special_ratio_max', 'country_special_ratio_max', 'continent_special_ratio_max', 'user_spe_degree_max']
    user_spe_degree_min = user_spe_degree.groupby('userid').min()[['city_special_ratio', 'country_special_ratio', 'continent_special_ratio', 'user_spe_degree']].reset_index()
    user_spe_degree_min.columns = ['userid', 'city_special_ratio_min', 'country_special_ratio_min', 'continent_special_ratio_min', 'user_spe_degree_min']
    user_spe_degree_var = user_spe_degree.groupby('userid').var()[['city_special_ratio', 'country_special_ratio', 'continent_special_ratio', 'user_spe_degree']].reset_index()
    user_spe_degree_var.columns = ['userid', 'city_special_ratio_var', 'country_special_ratio_var', 'continent_special_ratio_var', 'user_spe_degree_var']
    user_spe_degree_skew = user_spe_degree.groupby('userid').skew()[['city_special_ratio', 'country_special_ratio', 'continent_special_ratio', 'user_spe_degree']].reset_index()
    user_spe_degree_skew.columns = ['userid', 'city_special_ratio_skew', 'country_special_ratio_skew', 'continent_special_ratio_skew', 'user_spe_degree_skew']
    user_spe_degree_prod = user_spe_degree.groupby('userid').cumprod()[['city_special_ratio', 'country_special_ratio', 'continent_special_ratio', 'user_spe_degree']]
    user_spe_degree_prod = pd.concat([user_spe_degree.userid, user_spe_degree_prod], axis = 1) #type:pd.DataFrame
    user_spe_degree_prod = user_spe_degree_prod.groupby('userid').last().reset_index()
    user_spe_degree_prod.columns = ['userid', 'city_special_multi', 'country_special_multi', 'continent_special_multi', 'user_spe_degree_multi']
    # 下单时间差特征
    orderTime_df = df[['userid', 'orderTime']].drop_duplicates()
    orderTime_df = orderTime_df.sort_values(by = ['userid', 'orderTime'])
    userid = orderTime_df.userid.unique()
    tm_feats_dict = {'userid': [],
                     'order_diff_mean': [],
                     'order_diff_var': [],
                     'order_diff_max': [],
                     'order_diff_min': [],
                     'order_diff_skew': [],
                     'order_diff_median': [],
                     'latest_order_diff_1': [],
                     'latest_order_diff_2': [],
                     'latest_order_diff_3': []}
    for id in userid:
        tm_df = orderTime_df[orderTime_df.userid == id]
        tm_df = tm_df[['userid', 'orderTime']]
        tm_df.set_index('userid', inplace = True)
        tm_df = tm_df.diff()
        tm_feats_dict['userid'].append(id)
        tm_feats_dict['order_diff_mean'].append(tm_df.orderTime.mean())
        tm_feats_dict['order_diff_min'].append(tm_df.orderTime.min())
        tm_feats_dict['order_diff_max'].append(tm_df.orderTime.max())
        tm_feats_dict['order_diff_var'].append(tm_df.orderTime.var())
        tm_feats_dict['order_diff_median'].append(tm_df.orderTime.median())
        tm_feats_dict['order_diff_skew'].append(tm_df.orderTime.skew())
        tm_feats_dict['latest_order_diff_1'].append(list(tm_df.orderTime)[-1])
        try:
            tm_feats_dict['latest_order_diff_2'].append(list(tm_df.orderTime)[-2])
        except:
            tm_feats_dict['latest_order_diff_2'].append(np.nan)
        try:
            tm_feats_dict['latest_order_diff_3'].append(list(tm_df.orderTime)[-3])
        except:
            tm_feats_dict['latest_order_diff_3'].append(np.nan)
    tm_feats_dict = pd.DataFrame(tm_feats_dict)
    # 用户历史去各大洲的次数及频率
    user_continent_ratio = df.groupby(['userid', 'continent']).size().unstack()
    user_continent_ratio.fillna(0, inplace = True)
    user_continent_ratio['cnt'] = user_continent_ratio.sum(axis = 1)
    for col in user_continent_ratio.columns:
        if col != 'cnt':
            user_continent_ratio[col + '_ratio'] = user_continent_ratio[col] / user_continent_ratio.cnt
    user_continent_ratio.reset_index(inplace = True)
    del user_continent_ratio['cnt']
    # 合并特征
    order_feats = pd.merge(order_cnt, city_cnt, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, latest_order_tm, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, latest_special_order_tm, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, user_spe_degree_sum, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, user_spe_degree_mean, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, user_spe_degree_max, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, user_spe_degree_min, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, user_spe_degree_var, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, user_spe_degree_skew, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, user_spe_degree_prod, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, tm_feats_dict, on = 'userid', how = 'left')
    order_feats = pd.merge(order_feats, user_continent_ratio, on = 'userid', how = 'left')
    # 缓存
    pickle.dump(order_feats, open(orderHistory_dump_path, 'wb'))
    gc.collect()
    return order_feats
# UserComment
def gen_userComment_feats():
    # 读取数据
    userComment_train = pd.read_csv('trainingset/userComment_train.csv')
    userComment_test = pd.read_csv('test/userComment_test.csv')
    df = pd.concat([userComment_train, userComment_test], axis = 0) #type:pd.DataFrame
    # 数据预处理
    df.tags.fillna(u'无', inplace = True)
    df.commentsKeyWords.fillna(u'无', inplace = True)
    # 用户tags字数、词数
    df['tag_words_cnt'] = df.tags.apply(lambda x: tags_words_count(x))
    df['tag_phrase_cnt'] = df.tags.apply(lambda x: tag_phrase_count(x))
    # 用户commentsKeyWords词数
    df['commenKey_phrase_cnt'] = df.commentsKeyWords.apply(lambda x: commentsKeyWords_phrase_count(x))
    # 组合特征
    df['rating_tag_words_cnt'] = df.rating * df.tag_words_cnt
    df['rating_tag_phrase_cnt'] = df.rating * df.tag_phrase_cnt
    df['rating_commenKey_phrase_cnt'] = df.rating * df.commenKey_phrase_cnt
    # 剔除不用的特征
    df.drop(['orderid', 'tags', 'commentsKeyWords'],
            axis = 1,
            inplace = True)
    # 缓存
    pickle.dump(df, open(userComment_dump_path, 'wb'))
    gc.collect()
    return df
# Action & OrderHistory
def gen_order_action_feats():
    ### 读取数据 ###
    orderHistory_train = pd.read_csv('trainingset/orderHistory_train.csv')
    orderHistory_test = pd.read_csv('test/orderHistory_test.csv')
    order_df = pd.concat([orderHistory_train, orderHistory_test], axis = 0)  # type:pd.DataFrame
    order_df.drop(['orderType', 'orderid', 'city', 'country', 'continent'],
                  axis = 1,
                  inplace = True)
    order_df = order_df.drop_duplicates()
    action_train = pd.read_csv('trainingset/action_train.csv')
    action_test = pd.read_csv('test/action_test.csv')
    action_df = pd.concat([action_train, action_test], axis = 0)  # type:pd.DataFrame
    ### 下单前的action ###
    order_df = pd.merge(order_df, action_df, on = 'userid', how = 'left')
    order_df = order_df[order_df.actionTime <= order_df.orderTime]
    ### 用户下单前各actionType次数\占比 ###
    user_act_cnt_before_order = order_df.groupby(['userid', 'actionType']).size().unstack()
    user_act_cnt_before_order.fillna(0, inplace = True)
    user_act_cnt_before_order['action_cnt_before_order'] = user_act_cnt_before_order.sum(axis = 1)
    for col in user_act_cnt_before_order.columns:
        if col != 'action_cnt_before_order':
            user_act_cnt_before_order['actionType' + str(col) + '_ratio_before_order'] = user_act_cnt_before_order[col] / user_act_cnt_before_order.action_cnt_before_order
    user_act_cnt_before_order = user_act_cnt_before_order[['actionType1_ratio_before_order', 'actionType2_ratio_before_order',
                                                           'actionType3_ratio_before_order', 'actionType4_ratio_before_order',
                                                           'actionType5_ratio_before_order', 'actionType6_ratio_before_order',
                                                           'actionType7_ratio_before_order', 'actionType8_ratio_before_order',
                                                           'actionType9_ratio_before_order', 'action_cnt_before_order']]
    user_act_cnt_before_order.reset_index(inplace = True)
    user_act_mean = user_act_cnt_before_order.groupby('userid').mean().reset_index()
    user_act_max = user_act_cnt_before_order.groupby('userid').max().reset_index()
    user_act_min = user_act_cnt_before_order.groupby('userid').min().reset_index()
    user_act_var = user_act_cnt_before_order.groupby('userid').var().reset_index()
    user_act_median = user_act_cnt_before_order.groupby('userid').median().reset_index()
    user_act_skew = user_act_cnt_before_order.groupby('userid').skew().reset_index()
    order_action_feats = pd.merge(user_act_mean, user_act_max, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, user_act_min, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, user_act_var, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, user_act_median, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, user_act_skew, on = 'userid', how = 'left')
    ### 用户每个动作到下单时间的时间差 ###
    order_df['tm_diff'] = order_df.orderTime - order_df.actionTime
    tm_diff_mean = order_df.groupby('userid').mean()['tm_diff'].reset_index()
    tm_diff_mean.columns = ['userid', 'tm_diff_mean']
    tm_diff_var = order_df.groupby('userid').var()['tm_diff'].reset_index()
    tm_diff_var.columns = ['userid', 'tm_diff_var']
    tm_diff_median = order_df.groupby('userid').median()['tm_diff'].reset_index()
    tm_diff_median.columns = ['userid', 'tm_diff_median']
    tm_diff_max = order_df.groupby('userid').max()['tm_diff'].reset_index()
    tm_diff_max.columns = ['userid', 'tm_diff_max']
    tm_diff_min = order_df.groupby('userid').min()['tm_diff'].reset_index()
    tm_diff_min.columns = ['userid', 'tm_diff_min']
    tm_diff_skew = order_df.groupby('userid').skew()['tm_diff'].reset_index()
    tm_diff_skew.columns = ['userid', 'tm_diff_skew']
    order_action_feats = pd.merge(order_action_feats, tm_diff_mean, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, tm_diff_var, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, tm_diff_median, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, tm_diff_max, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, tm_diff_min, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, tm_diff_skew, on = 'userid', how = 'left')
    ### 用户每一单各actionType的时间差特征 ###
    tm_seq = order_df.groupby(['userid', 'orderTime']).apply(lambda x: x['actionTime'].tolist()).reset_index()
    tm_seq.columns = ['userid', 'orderTime', 'actionTime_seq']
    tm_seq['actionTime_diff_mean_before_order'] = tm_seq.actionTime_seq.apply(lambda x: np.mean(np.diff(x)))
    tm_seq['actionTime_diff_var_before_order'] = tm_seq.actionTime_seq.apply(lambda x: np.var(np.diff(x)))
    tm_seq['actionTime_diff_max_before_order'] = tm_seq.actionTime_seq.apply(lambda x: get_tm_diff_max(np.diff(x)))
    tm_seq['actionTime_diff_min_before_order'] = tm_seq.actionTime_seq.apply(lambda x: get_tm_diff_min(np.diff(x)))
    tm_seq['actionTime_diff_skew_before_order'] = tm_seq.actionTime_seq.apply(lambda x: skew(np.diff(x)))
    tm_seq['latest_actionTime_diff_1_before_order'] = tm_seq.actionTime_seq.apply(lambda x: get_k_index(np.diff(x), -1))
    tm_seq['latest_actionTime_diff_2_before_order'] = tm_seq.actionTime_seq.apply(lambda x: get_k_index(np.diff(x), -2))
    tm_seq['latest_actionTime_diff_3_before_order'] = tm_seq.actionTime_seq.apply(lambda x: get_k_index(np.diff(x), -3))
    del tm_seq['actionTime_seq']
    ### 用户最近一单各actionType的时间差特征 ###
    tm_seq = tm_seq.sort_values(by = ['userid', 'orderTime'])
    latest_tm_seq = tm_seq.groupby('userid').last().reset_index()
    del latest_tm_seq['orderTime']
    order_action_feats = pd.merge(order_action_feats, latest_tm_seq, on = 'userid', how = 'left')
    order_action_feats = pd.merge(order_action_feats, tm_seq.groupby('userid').mean().reset_index(), on = 'userid', how = 'left')
    # 缓存
    pickle.dump(order_action_feats, open(order_action_dump_path, 'wb'))
    gc.collect()
    return order_action_feats
# lastest date Action
def gen_latestday_action_feats():
    action_train = pd.read_csv('trainingset/action_train.csv')
    action_test = pd.read_csv('test/action_test.csv')
    df = pd.concat([action_train, action_test], axis = 0)  # type:pd.DataFrame
    df['actionDt'] = df.actionTime.apply(lambda x: timestamp_transform(x))
    df['actionDt'] = pd.to_datetime(df.actionDt)
    ### 最近1天各actionType次数及比率特征 ###
    action_feats = gen_action_feats_mid(df, 1)
    del action_feats['weight']
    # 缓存
    pickle.dump(action_feats, open(latestdate_action_dump_path, 'wb'))
    gc.collect()
    print(action_feats.head())
    return action_feats
# Merge
def merge():
    start_time = time.time()
    # 读取特征
    # userProfile
    if os.path.exists(userProfile_dump_path):
        userProfile_feats = pickle.load(open(userProfile_dump_path, 'rb'))
    else:
        userProfile_feats = gen_userProfile_feats()
    # actions
    if os.path.exists(Action_dump_path):
        Action_feats = pickle.load(open(Action_dump_path, 'rb'))
    else:
        Action_feats = gen_action_feats()
    # orderHistory
    if os.path.exists(orderHistory_dump_path):
        orderHistory_feats = pickle.load(open(orderHistory_dump_path, 'rb'))
    else:
        orderHistory_feats = gen_orderHistory_feats()
    # userComment
    if os.path.exists(userComment_dump_path):
        userComment_feats = pickle.load(open(userComment_dump_path, 'rb'))
    else:
        userComment_feats = gen_userComment_feats()
    # order_action
    if os.path.exists(order_action_dump_path):
        order_action_feats = pickle.load(open(order_action_dump_path, 'rb'))
    else:
        order_action_feats = gen_order_action_feats()
    # latest date action
    if os.path.exists(latestdate_action_dump_path):
        latestdate_action_feats = pickle.load(open(latestdate_action_dump_path, 'rb'))
    else:
        latestdate_action_feats = gen_latestday_action_feats()
    print(u'userProfile的数据维度:', userProfile_feats.shape)
    print(u'Action_feats的数据维度:', Action_feats.shape)
    print(u'orderHistory_feats的数据维度:', orderHistory_feats.shape)
    print(u'userComment_feats的数据维度:', userComment_feats.shape)
    print(u'order_action_feats的数据维度:', order_action_feats.shape)
    print(u'latestdate_action_feats的数据维度:', latestdate_action_feats.shape)
    # 合并特征
    orderFuture_train = pd.read_csv('trainingset/orderFuture_train.csv')
    orderFuture_test = pd.read_csv('test/orderFuture_test.csv')
    orderFuture = pd.concat([orderFuture_train, orderFuture_test], axis = 0) #type:pd.DataFrame
    orderFuture = pd.merge(orderFuture, userProfile_feats, on = 'userid', how = 'left')
    orderFuture = pd.merge(orderFuture, Action_feats, on = 'userid', how = 'left')
    orderFuture = pd.merge(orderFuture, orderHistory_feats, on = 'userid', how = 'left')
    orderFuture = pd.merge(orderFuture, userComment_feats, on = 'userid', how = 'left')
    orderFuture = pd.merge(orderFuture, order_action_feats, on = 'userid', how = 'left')
    # 组合特征
    # 用户是否购买过精品服务
    orderFuture['has_bought_spe_before'] = 0
    orderFuture.loc[orderFuture.special_cnt > 0, 'has_bought_spe_before'] = 1
    orderFuture['has_bought_before'] = 0
    orderFuture.loc[orderFuture.order_cnt > 0, 'has_bought_before'] = 1
    orderFuture['fir_action_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.first_actionTime_1
    # 组合时间差特征
    orderFuture['latest_action_1_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType_1_tm
    orderFuture['latest_action_2_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType_2_tm
    orderFuture['latest_action_3_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType_3_tm
    orderFuture['latest_action1_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType1_tm
    orderFuture['latest_action2_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType2_tm
    orderFuture['latest_action3_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType3_tm
    orderFuture['latest_action4_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType4_tm
    orderFuture['latest_action5_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType5_tm
    orderFuture['latest_action6_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType6_tm
    orderFuture['latest_action7_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType7_tm
    orderFuture['latest_action8_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType8_tm
    orderFuture['latest_action9_order_tm_range'] = orderFuture.latest_orderTime - orderFuture.latest_actionType9_tm
    # 最后一个actionType是否是5\6\7
    orderFuture['is_latest_actionType_5'] = orderFuture.latest_actionType_1.apply(lambda x: is_actionType(x, 5))
    orderFuture['is_latest_actionType_6'] = orderFuture.latest_actionType_1.apply(lambda x: is_actionType(x, 6))
    orderFuture['is_latest_actionType_7'] = orderFuture.latest_actionType_1.apply(lambda x: is_actionType(x, 7))
    # 用户行为购买转化率
    orderFuture.order_cnt.fillna(0, inplace = True)
    orderFuture['action2order_ctr'] = orderFuture.order_cnt / orderFuture.action_cnt
    orderFuture['lastdate_action_ratio'] = orderFuture['windows 1 action cnt'] / orderFuture.action_cnt
    orderFuture['actionType5_to_actionType6_index'] = orderFuture.latest_actionType5_act_range - orderFuture.latest_actionType6_act_range
    # # 特征的缺失值情况
    # featurs_nans = orderFuture.isnull().sum(axis = 0).reset_index()
    # featurs_nans.columns = ['feature_name', 'nan_cnt']
    # featurs_nans = featurs_nans.sort_values(by = 'nan_cnt', ascending = False)
    # featurs_nans['nan_ratio'] = featurs_nans.nan_cnt / len(orderFuture)
    # featurs_nans.to_csv('cache/features_nans.csv')
    # 剔除缺失值较多的特征
    drop_col = ['latest_order_diff_3', 'order_diff_skew', 'latest_actionType9_actionTime_diff_2', 'latest_order_diff_2', 'order_diff_var']
    orderFuture.drop(drop_col,
                     axis = 1,
                     inplace = True)
    # 拆分训练集和测试集
    train = orderFuture[orderFuture.orderType.isnull() == False]
    test = orderFuture[orderFuture.orderType.isnull() == True]
    # 提取预测目标
    target = train.orderType
    del train['orderType']
    del test['orderType']
    print(u'train的数据维度:', train.shape)
    print(u'test的数据维度:', test.shape)
    print(u'[{}] 完成特征工程'.format(time.time() - start_time))
    return train, test, target
###################################### model #########################################
# LightGBM
def lgb_model_cv(df_train, df_test, target):
    # # 剔除userid
    # df_train = df_train.set_index('userid')
    # df_test = df_test.set_index('userid')
    # 构造交叉验证集
    pred = 0
    auc_score = 0
    K = 5
    kf = KFold(n_splits = K, random_state = 0, shuffle = True)
    np.random.seed(1024)
    for i, (train_index, test_index) in enumerate(kf.split(df_train)):
        train_y, test_y = target.iloc[train_index].copy(), target.iloc[test_index]
        train_X, test_X = df_train.iloc[train_index, :].copy(), df_train.iloc[test_index, :].copy()
        dval = lgb.Dataset(test_X,
                           label = test_y)
        dtrain = lgb.Dataset(train_X,
                             label = train_y)
        parameters = {'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'learning_rate': 0.01,
                      # 'scale_pos_weight': 3,
                      'metric': 'auc',
                      'num_leaves': 2 ** 5,
                      'max_depth': 10,
                      'feature_fraction': 0.8,
                      'bagging_fraction': 0.95,
                      'bagging_freq': 5,
                      #'is_unbalance': 'true',
                      'seed': 1024}
        lgb_model = lgb.train(parameters,
                              dtrain,
                              num_boost_round = 3000,
                              valid_sets = dval,
                              early_stopping_rounds = 100)
        pred_valid = lgb_model.predict(test_X, num_iteration = lgb_model.best_iteration)
        valid_result = pd.DataFrame({'True_Target': test_y,
                                     'Pred_Prob': pred_valid})
        auc_score += roc_auc_score(valid_result['True_Target'].values, valid_result['Pred_Prob'])
        # 预测
        pred += lgb_model.predict(df_test, num_iteration = lgb_model.best_iteration)
    pred /= K
    auc_score /= K
    print(u'线下auc得分:', auc_score)
    df_test['pred_prob'] = pred
    df_test.reset_index(inplace = True)
    return df_test

def main():
    start_time = time.time()
    # 生成特征及预测目标
    train, test, target = merge()
    # 训练模型
    result = lgb_model_cv(train, test, target)
    print(u'[{}] 完成模型训练'.format(time.time() - start_time))
    # 提交结果
    submission = pd.read_csv(u'submit_sample.csv')
    submission = pd.merge(submission, result[['userid', 'pred_prob']], on = 'userid', how = 'left')
    submission = submission[['userid', 'pred_prob']]
    submission = submission.rename(columns = {'pred_prob': 'orderType'})
    submission.to_csv('submission/lgb_submission.csv', index = False)
    print(u'[{}] 完成生成结果'.format(time.time() - start_time))

if __name__ == '__main__':
    main()
