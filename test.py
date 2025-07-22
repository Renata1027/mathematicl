import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.font_manager import FontProperties

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# ====================
# 直接指定Windows系统SimHei字体文件
# ====================
def setup_chinese_font():
    # Windows系统SimHei字体通常路径（64位系统通用）
    simhei_path = "C:/Windows/Fonts/simhei.ttf"

    # 检查字体文件是否存在
    if not os.path.exists(simhei_path):
        print("警告：未找到SimHei字体文件，请检查路径是否正确！")
        print("默认路径：C:/Windows/Fonts/simhei.ttf")
        return None

    # 创建字体属性对象（强制使用该字体文件）
    chinese_font = FontProperties(fname=simhei_path)

    # 全局设置：所有文本默认使用该字体
    mpl.rcParams["font.family"] = [chinese_font.get_name()]
    mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 清理所有字体缓存（关键步骤）
    cache_dir = mpl.get_cachedir()
    for filename in os.listdir(cache_dir):
        if filename.startswith('fontlist') and filename.endswith('.json'):
            os.remove(os.path.join(cache_dir, filename))
    print("✅ 已加载SimHei字体并清理缓存")
    return chinese_font


# 初始化中文字体
chinese_font = setup_chinese_font()

# ====================
# 数据预处理
# ====================
df = sns.load_dataset('titanic')

# 填充年龄缺失值
age_median = df['age'].median()
df['age'] = df['age'].fillna(age_median)
print(f"年龄中位数: {age_median:.1f} 岁")

# 删除deck列
df = df.drop('deck', axis=1)

# 票价异常值处理
Q1, Q3 = df['fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df_cleaned = df[(df['fare'] >= lower) & (df['fare'] <= upper)]
print(f"票价异常值范围: [{lower:.2f}, {upper:.2f}]")

# ====================
# 可视化1：票价异常值对比
# ====================
plt.figure(figsize=(12, 6))

# 原始票价箱线图
plt.subplot(1, 2, 1)
plt.boxplot(df['fare'].dropna())
plt.title('原始票价箱线图', fontproperties=chinese_font)  # 显式指定字体
plt.ylabel('票价', fontproperties=chinese_font)

# 处理后票价箱线图
plt.subplot(1, 2, 2)
plt.boxplot(df_cleaned['fare'])
plt.title('处理后的票价箱线图', fontproperties=chinese_font)
plt.ylabel('票价', fontproperties=chinese_font)

plt.tight_layout()
plt.savefig('fare_boxplot.png', dpi=300)
plt.show()

# ====================
# 可视化2：多子图分析
# ====================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(20, 15))

# 子图1：舱位与幸存率
plt.subplot(2, 2, 1)
sns.barplot(x='pclass', y='survived', data=df_cleaned,
            estimator=np.mean, palette="viridis", errorbar=None)

# 添加百分比标签
for i, bar in enumerate(plt.gca().patches):
    rate = df_cleaned.groupby('pclass')['survived'].mean().iloc[i]
    plt.text(bar.get_x() + bar.get_width() / 2, rate + 0.03,
             f"{rate * 100:.1f}%", ha='center', fontproperties=chinese_font)

plt.title('不同舱位的幸存率', fontproperties=chinese_font, fontsize=12)
plt.xlabel('舱位等级', fontproperties=chinese_font, fontsize=10)
plt.ylabel('幸存率', fontproperties=chinese_font, fontsize=10)
plt.ylim(0, 1.1)

# 子图2：年龄与幸存率
plt.subplot(2, 2, 2)
sns.boxplot(x='survived', y='age', data=df_cleaned,
            palette="coolwarm", showfliers=False)

# 添加中位数标签
survived_age = df_cleaned.groupby('survived')['age'].median().reset_index()
for idx, row in survived_age.iterrows():
    plt.text(idx, row['age'] + 2,
             f"中位数: {row['age']:.1f}",
             ha='center', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8),
             fontproperties=chinese_font)

plt.title('幸存状态与年龄分布', fontproperties=chinese_font, fontsize=12)
plt.xlabel('是否幸存', fontproperties=chinese_font, fontsize=10)
plt.ylabel('年龄', fontproperties=chinese_font, fontsize=10)
plt.xticks([0, 1], ['未幸存', '幸存'], fontproperties=chinese_font)

# 子图3：特征相关性
plt.subplot(2, 2, 3)
corr_matrix = df.corr(numeric_only=True)
top_features = corr_matrix['survived'].abs().sort_values(ascending=False).index[1:4]
sns.heatmap(corr_matrix.loc[['survived'], top_features],
            annot=True, cmap="coolwarm", fmt=".2f",
            cbar=True, annot_kws={"fontproperties": chinese_font})  # 注释字体

plt.title('与幸存率相关性最高的特征', fontproperties=chinese_font, fontsize=12)
plt.yticks(rotation=0, fontproperties=chinese_font)

# 子图4：特征解释（中文核心区域）
plt.subplot(2, 2, 4)
plt.axis('off')

feature_explanations = [
    "1. 舱位等级 (pclass): 头等舱乘客优先获救",
    "2. 票价 (fare): 高票价乘客生存机会更高",
    "3. 年龄 (age): 儿童和老人优先获救"
]
corr_values = corr_matrix['survived'].loc[top_features].values

for i in range(3):
    # 特征名称+相关性
    plt.text(0.1, 0.8 - i * 0.2,
             f"{i + 1}. {top_features[i]} (相关性: {corr_values[i]:.2f})",
             fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5),
             fontproperties=chinese_font)  # 强制指定字体
    # 中文解释
    plt.text(0.1, 0.7 - i * 0.2,
             feature_explanations[i],
             fontsize=10,
             fontproperties=chinese_font)  # 强制指定字体

# 全局标题
plt.suptitle('泰坦尼克号幸存者特征分析', fontproperties=chinese_font, fontsize=18, y=0.98)
plt.tight_layout()
plt.savefig('titanic_analysis.png', dpi=300)
plt.show()

# ====================
# 模型训练与解释
# ====================
features = ['pclass', 'age', 'fare', 'sex']
X = df[features].copy()
X['sex'] = X['sex'].map({'male': 0, 'female': 1})
X['age'].fillna(X['age'].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, df['survived'], test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("\n===== 模型评估结果 =====")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("分类报告:")
print(classification_report(y_test, y_pred))

# 特征系数可视化
plt.figure(figsize=(10, 6))
feature_names = ['舱位', '年龄', '票价', '性别']
coefs = model.coef_[0]
plt.barh(feature_names, coefs, color='skyblue')
plt.axvline(0, color='gray', linestyle='--')
plt.title('逻辑回归特征系数（影响方向）', fontproperties=chinese_font)
plt.xlabel('系数值（正=增加幸存概率，负=降低）', fontproperties=chinese_font)
plt.yticks(fontproperties=chinese_font)  # y轴标签字体
plt.tight_layout()
plt.show()

print("\n特征影响解释:")
for name, coef in zip(feature_names, coefs):
    trend = "增加幸存概率" if coef > 0 else "降低幸存概率"
    print(f"{name}: 系数={coef:.4f} → {trend}")