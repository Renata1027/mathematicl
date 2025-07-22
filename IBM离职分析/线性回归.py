import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import os

# ====================
# Windows中文显示设置（强制生效版）
# ====================
def setup_chinese_font():
    """强制加载Windows系统SimHei字体，确保中文显示"""
    # SimHei字体在Windows的默认路径
    simhei_path = r"C:\Windows\Fonts\simhei.ttf"

    # 检查字体文件是否存在
    if not os.path.exists(simhei_path):
        print("⚠️ 错误：未找到SimHei字体文件！")
        print(f"预期路径：{simhei_path}")
        print("解决方案：① 确认系统安装了SimHei字体；② 手动指定正确路径。")
        return None

    # 创建字体属性对象（强制绑定该字体文件）
    chinese_font = FontProperties(fname=simhei_path)

    # 全局字体设置（双重保障）
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 清理Matplotlib字体缓存
    cache_dir = mpl.get_cachedir()
    for filename in os.listdir(cache_dir):
        if filename.startswith('fontlist') and filename.endswith('.json'):
            os.remove(os.path.join(cache_dir, filename))
    print("✅ SimHei字体加载成功，已清理字体缓存！")
    return chinese_font

# 初始化中文字体
chinese_font = setup_chinese_font()
# 确保即使字体对象获取失败，也有基础中文配置
if chinese_font is None:
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
    plt.rcParams['axes.unicode_minus'] = False
    chinese_font = FontProperties()  # 创建空对象避免报错


# 数据加载
fp = 'D:/IBM_HR_Analytics.csv'
ibmdata = pd.read_csv(fp, index_col=0)
target = 'MonthlyIncome'
x = ibmdata.drop(columns=[target])
y = ibmdata[target]

# 特征类型识别
def identify_feature_types(df, unique_threshold=15):
    numeric_features = []
    categorical_features = []
    categoricals = [
        'Attrition', 'BusinessTravel',
        'Department', 'EducationField', 'Gender',
        'JobRole', 'MaritalStatus', 'OverTime',
        'Over18',
    ]
    numeric_categoricals = [
        'Education', 'EnvironmentSatisfaction',
        'JobInvolvement', 'JobLevel',
        'JobSatisfaction', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel',
        'WorkLifeBalance'
    ]

    for col in df.columns:
        if col in categoricals:
            categorical_features.append(col)
            continue
        if col in numeric_categoricals:
            categorical_features.append(col)
            continue
        else:
            numeric_features.append(col)
    return numeric_features, categorical_features

numeric_features, categorical_features = identify_feature_types(x)

print(f"连续数值型特征 ({len(numeric_features)}): {numeric_features}")
print(f"分类特征 ({len(categorical_features)}): {categorical_features}")

# 数据预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ]
)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
target_var = "MonthlyIncome"
if target_var in numeric_features:
    numeric_features.remove(target_var)

# 多重共线性检测（仅数值特征）
def calculate_vif(df, features, threshold=5):
    if not features:
        return []
    vif_data = pd.DataFrame()
    vif_data['feature'] = features
    try:
        vif_data['VIF'] = [variance_inflation_factor(df[features].values, i)
                           for i in range(len(features))]
        # 完整VIF
        print("\n所有特征的VIF值:")
        print(vif_data.sort_values("VIF", ascending=False))
    except Exception as e:
        print(f"计算错误: {e}")
        vif_data["VIF"] = np.nan
    high_vif = vif_data[vif_data['VIF'] > threshold].dropna()
    if not high_vif.empty:
        print("\n高VIF特征（多重共线性风险）:")
        print(high_vif.sort_values("VIF", ascending=False))
    return list(high_vif["feature"])
high_vif_features = calculate_vif(x_train, numeric_features)

# 特征重要性分析
rf = RandomForestRegressor(n_estimators=200, random_state=42, oob_score=True)
rf_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ]
)
rf_pipeline.fit(x_train, y_train)

# 获取特征名称
cat_encoder = preprocessor.named_transformers_['cat']
if hasattr(cat_encoder, 'get_feature_names_out'):
    cat_features = cat_encoder.get_feature_names_out(categorical_features)
else:
    cat_features = [f'cat_{i}' for i in range(len(cat_encoder.categorical_features_))]
all_features = numeric_features + list(cat_features)

print(all_features)

# 可视化特征重要性（中文标题）
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_pipeline.named_steps['regressor'].feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(14, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
plt.title('Top 20 重要特征 (随机森林)', fontproperties=chinese_font)  # 中文标题
plt.xlabel('重要性分数', fontproperties=chinese_font)  # 中文标签
plt.ylabel('特征名称', fontproperties=chinese_font)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# 创建端到端的机器学习管道
def build_regression_model(model):
    return Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ]
    )

# Lasso回归（L1正则）
lasso_model = build_regression_model(
    LassoCV(alphas=np.logspace(-4, 0, 50), cv=5, random_state=42, max_iter=5000)
)
lasso_model.fit(x_train, y_train)

# Ridge回归（L2回归）
ridge_model = build_regression_model(
    RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
)
ridge_model.fit(x_train, y_train)

# ElasticNet（结合L1和L2正则化）
elastic_model = build_regression_model(
    ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                 cv=5, max_iter=5000, n_jobs=-1)
)
elastic_model.fit(x_train, y_train)

def evaluate_model(model, model_name):
    # 交叉验证
    cv_scores = cross_val_score(model, x_train, y_train,
                               cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    # 测试集评估
    y_pred = model.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} 性能:")
    print(f"交叉验证 RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"测试集 R²: {r2:.4f}")

    # 获取模型选择的特征并可视化
    if model_name in ["Lasso", "ElasticNet"]:
        try:
            coefs = model.named_steps['regressor'].coef_
            selected_features = [feat for feat, coef in zip(all_features, coefs) if abs(coef) > 0.001]
            print(f"\n{model_name}选择的特征数量: {len(selected_features)}/{len(all_features)}")

            # 显示最重要的特征
            coef_df = pd.DataFrame({
                'Feature': all_features,
                'Coefficient': coefs
            }).sort_values('Coefficient', key=abs, ascending=False)

            print(f"最重要的{model_name}特征:")
            print(coef_df.head(10))

            # 可视化系数（中文标题）
            plt.figure(figsize=(10, 6))
            significant_coefs = coef_df[coef_df['Coefficient'].abs() > 0.01]
            sns.barplot(x='Coefficient', y='Feature',
                        data=significant_coefs.sort_values('Coefficient', ascending=False))
            plt.title(f'{model_name} 特征系数', fontproperties=chinese_font)  # 中文标题
            plt.xlabel('系数值', fontproperties=chinese_font)  # 中文标签
            plt.ylabel('特征名称', fontproperties=chinese_font)
            plt.tight_layout()
            plt.savefig(f'{model_name.lower()}_coefficients.png', dpi=300)
            plt.show()

            return selected_features
        except Exception as e:
            print(f"可视化错误: {e}")
            pass
    return []

# 评估模型
lasso_features = evaluate_model(lasso_model, "Lasso")
ridge_features = evaluate_model(ridge_model, "Ridge")
elastic_features = evaluate_model(elastic_model, "ElasticNet")

# 最终特征选择
all_selected_features = set(lasso_features + elastic_features + list(feature_importances.head(30)['Feature']))

print("\n" + "="*50)
print("分析与建议:")
print(f"1. 随机森林识别出 {len(feature_importances[feature_importances['Importance'] > 0.005])} 个重要特征")
print(f"2. Lasso选择 {len(lasso_features)} 个特征，ElasticNet选择 {len(elastic_features)} 个特征")
print(f"3. 合并后推荐 {len(all_selected_features)} 个特征用于最终建模")

print("\n推荐的特征选择策略:")
print("A. 对于需要解释性的模型:")
print("   - 使用Lasso或ElasticNet的特征选择结果")
print("B. 对于预测性能优先的模型:")
print("   - 使用随机森林重要性前30的特征")

# 保存重要特征列表
important_features_df = pd.DataFrame({
    'Feature': list(all_selected_features),
    'RF_Importance': [feature_importances.set_index('Feature').loc[f, 'Importance']
                      if f in feature_importances['Feature'].values else 0
                      for f in all_selected_features]
})
# 添加Lasso系数
if lasso_features:
    coef_df = pd.DataFrame({
        'Feature': all_features,
        'Lasso_Coefficient': lasso_model.named_steps['regressor'].coef_
    })
    important_features_df = important_features_df.merge(coef_df, on='Feature', how='left')

# 保存到CSV
important_features_df.sort_values('RF_Importance', ascending=False, inplace=True)
important_features_df.to_csv('selected_features.csv', index=False)
print("\n已保存选定的特征到 'selected_features.csv'")