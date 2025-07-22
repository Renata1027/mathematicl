import numpy as np

def calculate_ahp_weights_eigen(matrix):

  """
  计算判断矩阵的权重向量和一致性指标
  """
  matrix = np.array(matrix,dtype=float)
  n = matrix.shape[0] # 获取矩阵的行数
  # 计算特征值特征向量
  # np.linalg.eig 返回特征值和特征向量
  eigenvalues,eigenvectors = np.linalg.eig(matrix)
  # 找最大特征值和索引
  # 特征值可能是复数，但对于正互反矩阵，最大特征值通常是实数且唯一
  max_eigenvalue_index = np.argmax(eigenvalues.real) #通过 .real，我们只提取每个特征值的实部，舍弃其虚部
  lambda_max = eigenvalues.real[max_eigenvalue_index]

  # 获取对应于最大特征值的特征向量
  # 特征向量可能为负值，需要取绝对值，然后归一化

  weights = eigenvectors[:,max_eigenvalue_index].real
  weights = np.abs(weights) #取绝对值

  weights = weights/np.sum(weights)

  CI = (lambda_max-n)/(n-1)

  RI_values = {
        1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
  }

  """
  从 RI_values 字典中查找当前矩阵阶数 n 对应的随机一致性指标 RI 值。
  如果 n 值不在字典的键中（例如 n 大于10），
  它会安全地返回 0

  """
  RI = RI_values.get(n,0)


  CR = CI / RI if RI != 0 else 0

  return weights,CI,CR,lambda_max

print("--- 1. 目标层对准则层的判断矩阵 (使用特征根法) ---")
matrix_A = [
    [1, 5, 7],
    [1/5, 1, 3],
    [1/7, 1/3, 1]
]

weights_A, CI_A, CR_A, lambda_max_A = calculate_ahp_weights_eigen(matrix_A)
print(f"权重 (工作压力, 满意度, 稳定性): {weights_A}")
print(f"最大特征值 (lambda_max): {lambda_max_A:.4f}")
print(f"CI: {CI_A:.4f}, CR: {CR_A:.4f}")


print("\n--- 2. 工作压力（B）对加班（B1）和工作生活平衡（B2）的判断矩阵 (使用特征根法) ---")
matrix_B = [
    [1, 3],
    [1/3, 1]
]

weights_B, CI_B, CR_B, lambda_max_B = calculate_ahp_weights_eigen(matrix_B)
print(f"权重 (OverTime, WorkLifeBalance): {weights_B}")
print(f"最大特征值 (lambda_max): {lambda_max_B:.4f}")
print(f"CI: {CI_B:.4f}, CR: {CR_B:.4f}")

print("\n--- 3. 满意度（C）对工作满意度（C1）和环境满意度（C2）的判断矩阵 (使用特征根法) ---")
matrix_C = [
    [1, 2],
    [1/2, 1]
]
weights_C, CI_C, CR_C, lambda_max_C = calculate_ahp_weights_eigen(matrix_C)
print(f"权重 (JobSatisfaction, EnvironmentSatisfaction): {weights_C}")
print(f"最大特征值 (lambda_max): {lambda_max_C:.4f}")
print(f"CI: {CI_C:.4f}, CR: {CR_C:.4f}")

print("\n--- 4. 稳定性（D）对在公司年限（D1）和年龄（D2）的判断矩阵 (使用特征根法) ---")
matrix_D = [
    [1, 1/2],
    [2, 1]
]
weights_D, CI_D, CR_D, lambda_max_D = calculate_ahp_weights_eigen(matrix_D)
print(f"权重 (YearsAtCompany, Age): {weights_D}")
print(f"最大特征值 (lambda_max): {lambda_max_D:.4f}")
print(f"CI: {CI_D:.4f}, CR: {CR_D:.4f}")

# 准则层（一级指标）的权重
W_B_C_D = weights_A

# 方案层（二级指标）的权重
W_B1_B2 = weights_B
W_C1_C2 = weights_C
W_D1_D2 = weights_D

# 计算最终的组合权重
# OverTime (B1) 的总权重 = 工作压力 (B) 的权重 * OverTime (B1) 在工作压力下的权重
weight_OverTime = W_B_C_D[0] * W_B1_B2[0]
weight_WorkLifeBalance = W_B_C_D[0] * W_B1_B2[1]

weight_JobSatisfaction = W_B_C_D[1] * W_C1_C2[0]
weight_EnvironmentSatisfaction = W_B_C_D[1] * W_C1_C2[1]

weight_YearsAtCompany = W_B_C_D[2] * W_D1_D2[0]
weight_Age = W_B_C_D[2] * W_D1_D2[1]

final_weights = {
    "OverTime": weight_OverTime,
    "WorkLifeBalance": weight_WorkLifeBalance,
    "JobSatisfaction": weight_JobSatisfaction,
    "EnvironmentSatisfaction": weight_EnvironmentSatisfaction,
    "YearsAtCompany": weight_YearsAtCompany,
    "Age": weight_Age
}
# 按权重降序排序
sorted_weights = sorted(final_weights.items(), key=lambda item: item[1], reverse=True)

print("\n--- 最终离职风险因素权重（AHP组合权重）---")
for factor, weight in sorted_weights:
    print(f"{factor}: {weight:.4f}")