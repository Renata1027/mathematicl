import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap.umap_ as umap
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
km = pd.read_csv(fp,index_col = 0)

km_original = km.copy()
# 数据清洗
km.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)  # 删除常量列
km['Attrition'] = km['Attrition'].map({'Yes':1, 'No':0})  # 标签编码

# 独热编码分类变量
cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
            'JobRole', 'MaritalStatus', 'OverTime']
km = pd.get_dummies(km, columns=cat_cols)

# 标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(km.drop('Attrition', axis=1))

# scaled_data是标准化后的数据（未进行任何PCA降维）
# 我们直接在标准化后的数据上进行UMAP降维

# UMAP参数调整建议：
# n_neighbors: 局部邻居的数量，值越大，UMAP越关注全局结构；值越小，越关注局部结构。
# min_dist: 簇的紧密程度，值越小，簇越紧密，点之间距离越小；值越大，点之间距离越大，簇更松散。
# n_components: 降维后的维度，通常设置为2或3便于可视化。

# 尝试UMAP降维到2D或3D
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(scaled_data) # 使用原始标准化数据进行UMAP
print(f"\nUMAP降维后特征数量: {X_umap.shape[1]}")

# 可视化UMAP结果（如果降维到2D）
if X_umap.shape[1] == 2:
    plt.figure(figsize=(10, 8))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], s=5, alpha=0.8)
    plt.title('UMAP Projection of Employee Data')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True)
    plt.show()
elif X_umap.shape[1] == 3: # 如果是3D，可视化会复杂一些，但有助于理解
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], s=5, alpha=0.8)
    ax.set_title('UMAP Projection of Employee Data (3D)')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')
    plt.show()

# 现在在UMAP降维后的数据上运行K-Means并计算轮廓系数
# 这里我们再次遍历K值来寻找最佳K
silhouette_scores_umap = []
k_values_umap = range(2, 11)

print("\nUMAP降维后不同K值下的轮廓系数：")
for k in k_values_umap:
    kmeans_umap = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels_umap = kmeans_umap.fit_predict(X_umap)
    score_umap = silhouette_score(X_umap, cluster_labels_umap)
    silhouette_scores_umap.append(score_umap)
    print(f"  K = {k}, 轮廓系数 = {score_umap:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values_umap, silhouette_scores_umap, marker='o', linestyle='--')
plt.title('Silhouette Score for Optimal K (with UMAP)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values_umap)
plt.grid(True)
plt.show()

k_final = 4
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
final_cluster_labels = kmeans_final.fit_predict(X_umap)

# 将聚类标签添加到原始数据框中
km_original['Cluster'] = final_cluster_labels

print("\n带有聚类标签的原始数据（部分）：")
print(km_original.head())
print(f"\n聚类簇数量: {km_original['Cluster'].nunique()}")
print(f"每个簇的样本数量:\n{km_original['Cluster'].value_counts().sort_index()}")

# 对每个聚类进行数值型特征的统计分析（均值）
print("\n每个聚类的平均特征值：")
# 这里我们选择所有数值型列进行均值计算，排除 'Cluster' 列本身
numeric_cols = km_original.select_dtypes(include=['int64', 'float64']).columns.drop('Cluster', errors='ignore')
cluster_summary_numeric = km_original.groupby('Cluster')[numeric_cols].mean()
print(cluster_summary_numeric)
# 可视化关键数值型特征的分布
# 选择一些对业务解读最有价值的特征进行箱线图或小提琴图可视化
# 确保这些特征在 df_original 中存在
features_for_viz_numeric = [
    'Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany',
    'TotalWorkingYears', 'NumCompaniesWorked', 'JobSatisfaction',
    'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance',
    'DailyRate', 'HourlyRate', 'PercentSalaryHike', 'StockOptionLevel',
    'TrainingTimesLastYear', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager',
    'Attrition_Yes' # 如果之前没有删除Attrition，这里可以分析其离职率
]
# 过滤掉不存在于 km_original 中的特征，并加入编码后的 'Attrition_Yes'
if 'Attrition' in km_original.columns: # 如果原始数据中包含'Attrition'列
    km_original['Attrition_Yes'] = (km_original['Attrition'] == 'Yes').astype(int)
    features_for_viz_numeric.append('Attrition_Yes')
feature_for_viz_numeric = [f for f in features_for_viz_numeric if f in km_original.columns]
plt.figure(figsize=(20, 25)) # 调整图表大小以容纳更多子图
for i, feature in enumerate(features_for_viz_numeric):
    plt.subplot(6, 4, i + 1) # 调整子图布局以适应特征数量
    sns.boxplot(x='Cluster', y=feature, data=km_original, palette='viridis')
    plt.title(f'{feature} by Cluster', fontsize=12)
    plt.xlabel('Cluster', fontsize=10)
    plt.ylabel(feature, fontsize=10)
plt.tight_layout()
plt.show()


# 对于分类变量，计算每个聚类中分类变量的频率或比例
print("\n--- 分类特征在各簇中的分布 ---")
categorical_features_for_viz = [
    'Gender', 'Education', 'EducationField', 'MaritalStatus', 'JobRole',
    'Department', 'BusinessTravel', 'OverTime'
]

for feature in categorical_features_for_viz:
    if feature in km_original.columns:
        print(f"\n--- {feature} by Cluster ---")
        # 使用 normalize='index' 得到每个簇内部的比例
        print(pd.crosstab(km_original['Cluster'], km_original[feature], normalize='index'))
        # 也可以可视化
        pd.crosstab(km_original['Cluster'], km_original[feature], normalize='index').plot(kind='bar', stacked=True, figsize=(8, 5))
        plt.title(f'{feature} Distribution by Cluster')
        plt.ylabel('Proportion')
        plt.show()