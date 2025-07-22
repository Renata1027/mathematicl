import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import os
# Windows中文显示（直接指定SimHei字体文件）
# ====================
def setup_chinese_font():
    """强制加载Windows系统SimHei字体，修复缓存目录获取问题"""
    # SimHei字体在Windows的默认路径（64位系统通用）
    simhei_path = r"C:\Windows\Fonts\simhei.ttf"

    # 检查字体文件是否存在
    if not os.path.exists(simhei_path):
        print("⚠️ 错误：未找到SimHei字体文件！")
        print(f"预期路径：{simhei_path}")
        print("解决方案：① 确认系统安装了SimHei字体；② 手动指定正确路径。")
        return None

    # 创建字体属性对象（强制绑定该字体文件）
    chinese_font = FontProperties(fname=simhei_path)

    # 清理Matplotlib字体缓存
    cache_dir = mpl.get_cachedir()
    for filename in os.listdir(cache_dir):
        if filename.startswith('fontlist') and filename.endswith('.json'):
            os.remove(os.path.join(cache_dir, filename))
    print("✅ SimHei字体加载成功，已清理字体缓存！")
    return chinese_font


# 初始化中文字体
chinese_font = setup_chinese_font()
if chinese_font:
    # 设置所有文本的默认字体
    mpl.rcParams['font.family'] = chinese_font.get_family()
    mpl.rcParams['font.sans-serif'] = chinese_font.get_name()
    # 解决负号显示问题（通常中文会遇到）
    mpl.rcParams['axes.unicode_minus'] = False
    print(f"✅ Matplotlib 全局默认字体已设置为: {chinese_font.get_name()}")
else:
    print("⚠️ 警告：中文字体未成功加载，无法设置全局默认字体。")
fp = 'D:/IBM_HR_Analytics.csv'
ibmdata = pd.read_csv(fp,index_col = 0)

# 初始数据清理
ibmdata_clean = ibmdata.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)
ibmdata_clean['Attrition'] = ibmdata_clean['Attrition'].map({'Yes':1, 'No':0})

# 1. 清理列名（去除空格和特殊字符）
ibmdata_clean.columns = ibmdata_clean.columns.str.strip()
print("清理后的列名:", ibmdata_clean.columns.tolist())

# 2. 调整分类列列表（使用实际列名）
cat_cols = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender',
    'JobRole', 'MaritalStatus', 'OverTime',
    'Education', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel',
    'WorkLifeBalance'
]


# 执行独热编码
ibmdata_clean = pd.get_dummies(ibmdata_clean, columns=cat_cols, drop_first=True)
print('当前列名：',ibmdata_clean.columns.tolist())

X_log = ibmdata_clean.drop('Attrition',axis = 1)
y_log = ibmdata_clean['Attrition']
# 处理不平衡
# 1：smote过采样
smote = SMOTE(random_state=42)
X_log_res,y_log_res = smote.fit_resample(X_log,y_log)
# 2.使用class_weight参数 (在模型训练步骤设置)

# 特征选择 - 选择与目标相关性高的特征
corr_matrix = ibmdata_clean.drop('Attrition', axis=1).corrwith(ibmdata_clean['Attrition']).abs()
selected_features_log = corr_matrix[corr_matrix > 0.05].index.tolist()
X_log = X_log_res[selected_features_log]

# 标准化
scaler = StandardScaler()
X_log_scaled = scaler.fit_transform(X_log)

# 划分数据集
X_log_train,X_log_test,y_log_train,y_log_test = train_test_split(
    X_log_scaled,y_log_res,test_size = 0.3,random_state = 42)
# 模型训练
# 使用smote方法
# model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
# 使用class_weight平衡：
model = LogisticRegression(penalty='l2',class_weight = 'balanced',solver='liblinear',random_state=42)
model.fit(X_log_train,y_log_train)
# 预测概率
y_log_prob = model.predict_log_proba(X_log_test)[:,1]
print("\n概率输出示例:", y_log_prob[:5])
# 混淆矩阵
y_log_pred = model.predict(X_log_test)
cm = confusion_matrix(y_log_test,y_log_pred)
plt.figure(figsize=(8,6))
ax = sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels = ['未离职', '离职'],
            yticklabels=['未离职', '离职'])
for label in ax.get_xticklabels():
    label.set_fontproperties(chinese_font)
# 迭代设置y轴刻度标签字体
for label in ax.get_yticklabels():
    label.set_fontproperties(chinese_font)
plt.xlabel('预测',fontproperties=chinese_font)
plt.ylabel('真实',fontproperties=chinese_font)
plt.title('混淆矩阵',fontproperties=chinese_font)
plt.show

# 召回率计算
tn,fp,fn,tp = cm.ravel()
recall = tp/(tp+fn)
print(f"召回率(离职检测率): {recall:.4f}")

# ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(y_log_test, y_log_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw = 2,label=f'ROC曲线（AUC = {roc_auc:.2f}）')
plt.plot([0,1],[0,1],color = 'navy',lw = 2,linestyle = '--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('假阳性率(FPR)',fontproperties=chinese_font)
plt.ylabel('真阳性率(TPR)',fontproperties=chinese_font)
plt.title('ROC曲线',fontproperties=chinese_font)
legend = plt.legend(loc="lower right")
for text in legend.get_texts():
    text.set_fontproperties(chinese_font)

plt.legend(loc="lower right")
plt.show()

# 分类报告
print("\n分类评估报告:")
print(classification_report(y_log_test, y_log_pred, target_names=['未离职', '离职']))