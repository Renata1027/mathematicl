import pandas as pd
import numpy as np

df = pd.read_csv('D:/IBM_HR_Analytics.csv')
df_original = df.copy()


# 定义用于TOPSIS的指标列
# 注意：'OverTime' 是分类变量，需要独热编码
topsis_features = [
    'OverTime', 'WorkLifeBalance', 'JobSatisfaction',
    'EnvironmentSatisfaction', 'YearsAtCompany', 'Age'
]

# 提取相关数据，并保留EmployeeNumber以便后续排名

df_topsis = df_original[['EmployeeNumber']+topsis_features].copy()

# --- 2.1 正向化处理 ---

# OverTime: 'Yes' -> 1, 'No' -> 0 (已经是正向，因为Yes表示加班，风险高)

df_topsis['OverTime_Yes'] = (df_topsis['OverTime'] == 'Yes').astype(int)

df_topsis = df_topsis.drop(columns = ['OverTime'])

# 更新 topsis_features 列表

topsis_features[topsis_features.index('OverTime')] = 'OverTime_Yes'
# WorkLifeBalance, JobSatisfaction, EnvironmentSatisfaction (1-4，1最差，4最好)
# 转换为：5 - 原值，使值越小越好转化为值越大越好
df_topsis['WorkLifeBalance'] = 5 - df_topsis['WorkLifeBalance']
df_topsis['JobSatisfaction'] = 5 - df_topsis['JobSatisfaction']
df_topsis['EnvironmentSatisfaction'] = 5 - df_topsis['EnvironmentSatisfaction']

# YearsAtCompany, Age (数值，值越大越稳定，风险越低)
# 转换为：最大值 - 原值，使值越小越好转化为值越大越好
max_years_at_company = df_topsis['YearsAtCompany'].max()
max_age = df_topsis['Age'].max()

df_topsis['YearsAtCompany'] = max_years_at_company + 1 - df_topsis['YearsAtCompany'] # +1确保值不为0，且不与最大值重叠
df_topsis['Age'] = max_age + 1 - df_topsis['Age'] # +1确保值不为0

print("正向化处理后的数据（部分）：")
print(df_topsis.head())


# 存储员工ID
employee_ids = df_topsis['EmployeeNumber']
# 提取用于TOPSIS计算的特征数据
data_matrix = df_topsis[topsis_features].values

# 计算每个指标（列）的平方和的开方
norm_factors = np.sqrt(np.sum(data_matrix**2, axis=0))

# 规范化决策矩阵
normalized_matrix = data_matrix / norm_factors

print("\n规范化后的决策矩阵（部分）：")
print(pd.DataFrame(normalized_matrix, columns=topsis_features).head())

# 确保这些权重与 topsis_features 的顺序一致
# 这里使用之前AHP示例计算的权重
ahp_weights = {
        "OverTime_Yes": 0.5480,
        "WorkLifeBalance": 0.1827,
        "JobSatisfaction": 0.1256,
        "EnvironmentSatisfaction": 0.0628,
        "YearsAtCompany": 0.0270,
        "Age": 0.0540
}

# 根据 topsis_features 的顺序创建权重向量
weights_vector = np.array([ahp_weights[feature] for feature in topsis_features])
print("\nTOPSIS使用的权重向量:", weights_vector)

# 构建加权规范化矩阵

weighted_normalized_matrix = normalized_matrix * weights_vector

print("\n加权规范化矩阵（部分）：")
print(pd.DataFrame(weighted_normalized_matrix, columns=topsis_features).head())

# 确定正理想解 (A_plus) 和负理想解 (A_minus)
# 由于我们已经正向化处理，所以所有指标都是“越大越好”
A_plus = np.max(weighted_normalized_matrix, axis=0)
A_minus = np.min(weighted_normalized_matrix, axis=0)

print("\n正理想解 (A_plus):", A_plus)
print("负理想解 (A_minus):", A_minus)

# 计算每个员工到正理想解的距离 D_plus
D_plus = np.sqrt(np.sum((weighted_normalized_matrix - A_plus)**2, axis=1))

# 计算每个员工到负理想解的距离 D_minus
D_minus = np.sqrt(np.sum((weighted_normalized_matrix - A_minus)**2, axis=1))

print("\n每个员工到正理想解的距离 (D_plus, 部分):")
print(D_plus[:5])
print("\n每个员工到负理想解的距离 (D_minus, 部分):")
print(D_minus[:5])

# 计算相对接近度 (C_i) - TOPSIS得分
# 避免分母为0的情况
C_i = D_minus / (D_plus + D_minus)
# 处理可能出现的 D_plus + D_minus == 0 的情况（理论上在非零数据中很少出现，但稳健性考虑）
C_i = np.nan_to_num(C_i, nan=0.0) # 将NaN替换为0

print("\n每个员工的TOPSIS得分 (C_i, 部分):")
print(C_i[:5])

# 创建包含员工ID和TOPSIS得分的DataFrame
topsis_results = pd.DataFrame({
    'EmployeeNumber': employee_ids,
    'TOPSIS_Score': C_i
})

# 按TOPSIS_Score降序排序，得分越高表示离职风险越高
ranked_employees = topsis_results.sort_values(by='TOPSIS_Score', ascending=False)

print("\n--- 员工离职风险排名 TOP 10 ---")
print(ranked_employees.head(10))