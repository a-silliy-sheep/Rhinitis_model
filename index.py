
#region---加载模块-----------------------------------------------
#   from sklearn.ensemble import RandomForestClassifier  # 随机森林模块
#   from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#   from sklearn.model_selection import ShuffleSplit
import joblib  # 封装和调用模型
import numpy as np
import pandas as pd
#   from sklearn.tree import export_graphviz
#   import graphviz
#   import matplotlib.pyplot as plt
#   import random
import streamlit as st

    # 设置数据框输出对齐：
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180) # 设置打印宽度(**重要**)
# endregion

#region---框架函数-------------------------------------------------
def printDf(df):
    print(f"\033[1;34ma. 列名：\033[1;0m", f"{df.columns}", 
            f" \033[1;34m\nb. 各列数据类型：\033[1;0m", f"\n{df.dtypes}",
            f"\033[1;34m\nc. 维度: \033[1;0m", f"{df.shape}",
            f"\033[1;34m\nd. 前5行3列数据为: \033[1;0m", f"\n{df.iloc[0:5, 0:3]}")

def get_compositionDf(df):  # 查看分类变量的构成比
    for i in df.columns:
        dat = df[i].value_counts()
        dat_ind = dat.index
        dat_ind.name = 'Factor'
        dat2 = pd.DataFrame({'count': dat.values, 'ratio (%)':np.round(100*dat.values/sum(dat.values), 1)}, index= dat_ind)
        print(f'\033[1;34m {i}: \n\033[1;0m', dat2, '\n')

# endregion ----------

#region ---------加载数据--------------------------

#       df_train = pd.read_excel("D:\编程\python\模型部署\Rhinitis\元数据\dat_forPython.xlsx", sheet_name=0)
#       df_validation = pd.read_excel("D:\编程\python\模型部署\Rhinitis\元数据\dat_forPython.xlsx", sheet_name=1)
#       df_test = pd.read_excel("D:\编程\python\模型部署\Rhinitis\元数据\dat_forPython.xlsx", sheet_name=2)
#       
#       df_Train = pd.concat([df_train, df_validation], axis=0, join='inner')


# endregion


# region----模型训练和评估----------------------------------------
#       clf = RandomForestClassifier(n_estimators=100, random_state=2024)


#     # 创建ShuffleSplit对象，用于执行自动洗牌
#     ss = ShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3, random_state=0)

# 循环遍历每个拆分，并使用随机森林分类器对每个拆分进行训练和评估
#       sub_ind = random.sample(list(range(df_Train.shape[0])), 2000)  # 无放回采样

#       X_train, X_test = df_Train.iloc[:, 1:6], df_test.iloc[:, 1:6]
#       y_train, y_test = df_Train.iloc[:, 0], df_test.iloc[:, 0]
#       clf.fit(X_train, y_train)
#       y_pred = clf.predict(X_test)
#       y_pred2 = (clf.predict_proba(X_test)[:, 1] > 0.24).astype(int)
#       
#       joblib.dump(clf, 'RF.pkl')

clf = joblib.load('RF.pkl')

#       print("\033[1;34mConfusion Matrix:\n\033[1;0m", pd.DataFrame(data=confusion_matrix(y_test, y_pred2),
#                   index=pd.Series(['0', '1'], name='Gold'),
#                   columns=pd.Series(['0', '1'], name='Diagnose')))  # 输出混淆矩阵
#       print("\033[1;34mClassification Report:\n\033[1;0m", classification_report(y_test, y_pred2, target_names=['0', '1']))  # 输出混淆矩阵衍生的各指标
#       print("Accuracy:\n", accuracy_score(y_test, y_pred))
#       print(clf.score(X_test, y_test))

#       clf.feature_names_in_
#       importances = clf.feature_importances_  # 计算特征重要性
#       print(importances)


# region-----streamlit 部署在线版预测工具---------------------

st.title('A simple tool to predict :blue[allergic rhinitis] among 2-8 year old preschool child',)

# number = st.sidebar.slider('选择一个数字', 0, 100, 50)  # 数据展示条
# data = pd.DataFrame({'a':np.random.randn(10), 'b':np.random.randn(10)})

# region----
#           if st.button('显示消息'):  # 按钮
#               st.write('Streamlit 是真的酷！')
#           if st.checkbox('显示图表'):  # 复选框
#               st.line_chart([0, 1, 2, 3, 4])
#           
#           genre = st.radio(
#             "你最喜欢哪种类型的音乐？",
#             ('流行', '摇滚', '爵士')
#           )
#           st.write(f'你的选择是：{genre}')
#           
#           age = st.slider('你的年龄：', 0, 130, 25)
#           st.write('我', age, '岁')
#           
#           # 初始化一个计数器
#           if 'count' not in st.session_state:
#               st.session_state.count = 0
#               
#           # 创建一个增加计数的按钮
#           if st.button('增加'):
#               st.session_state.count += 1
#           
#           st.write('计数器', st.session_state.count)
# endregion------

df_columns = ['event', 'mom_with_AR', 'dad_with_AR', 'haveOlderSblings', 'child with AD', 'dadEducation']

question_col = ['Has mother ever suffered from allergic rhinitis?',
                'Has father ever suffered from allergic rhinitis?',
                'Does the child have older brothers and sisters?',
                'Has the child ever suffered from atopic dermatitis diagnosed by a doctor?',
                "What's father's highest educational level?"]
st.sidebar.write('Blank in these questions:\n')
X_testStream = [1 if st.sidebar.selectbox(i, ('yes', 'no')) == 'yes' else 0 for i in question_col[:4]]
last_q = st.sidebar.selectbox(question_col[4], (
                        'high school diploma', 'Bachelor degree', 'Master degree or above'))


X_testStream = X_testStream + [(1 if ('high' in  last_q) else 2 if ('Bachelor' in last_q) else 3)]


X_testStreamDf = pd.DataFrame(data=np.array([X_testStream]),
                                columns=pd.Series(df_columns[1:6], name='columns'))

st.empty()
st.write('Data maping: \n', X_testStreamDf)
st.empty()

if st.sidebar.button('Predict'):
    y_predSt = clf.predict_proba(X_testStreamDf)[:,1][0]
    y_predSt2 = y_predSt * 0.5/0.24 if y_predSt <= 0.24 else 0.5 + ((1-0.5)/(1-0.25))*(y_predSt - 0.25)
    st.markdown('''Based on the information you provided, the random forest model predicts the probability
        of your child developing allergic rhinitis between the ages of 2-8 as follows:''')
    st.markdown(f":red[{round(y_predSt2, 3)}]")

#   if st.button('Predict'):
#       y_predSt = clf.predict_proba(X_testStreamDf)[:,1][0]
#       y_predSt2 = y_predSt * 0.5/0.24 if y_predSt <= 0.24 else 0.5 + ((1-0.5)/(1-0.25))*(y_predSt - 0.25)
#       st.markdown('''Based on the information you provided, the random forest model predicts the probability
#           of your child developing allergic rhinitis between the ages of 2-8 as follows:''')
#       st.markdown(f":red[{round(y_predSt2, 3)}]")
#   
# endregion
