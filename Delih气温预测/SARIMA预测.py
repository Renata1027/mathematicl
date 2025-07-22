import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 引入 pmdarima 库
import pmdarima as pm
import os  # 用于获取CPU核心数
from matplotlib.font_manager import FontProperties


# ====================
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

# 备用方案：若SimHei不存在，尝试系统其他中文字体
if chinese_font is None:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 回退到微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
    print("⚠️ 已回退到Microsoft YaHei字体，仍需确保系统安装该字体！")

# --- 数据加载与预处理 ---
# 加载数据（指定日期列为索引，注意路径转义）
df = pd.read_csv(
    r"D:\kaggle数据集\Delhi\DailyDelhiClimate.csv",
    parse_dates=['date'],
    index_col='date'
)

# 处理缺失值（时间序列插值）
if df['meantemp'].isnull().sum() > 0:
    print(f"检测到 'meantemp' 列有 {df['meantemp'].isnull().sum()} 个缺失值，正在插值处理...")
    df['meantemp'] = df['meantemp'].interpolate(method='time')

# 定义季节周期
SEASONAL_PERIOD = 30

# 划分训练集与测试集（8:2比例）
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# --- 主程序：SARIMAX建模（多进程加速） ---
if __name__ == '__main__':
    print("--- 步骤2: 自动化确定 SARIMAX 参数 (pmdarima 加速) ---")

    # 自动分配CPU核心（保留1个核心给系统）
    n_cpu = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
    print(f"使用 {n_cpu} 个核心进行参数搜索...")

    # Auto ARIMA 自动寻优
    auto_model = pm.auto_arima(
        train['meantemp'],
        start_p=0, start_q=0,  # 非季节性AR/MA起始阶数
        test='adf',  # 自动检测差分阶数 d
        max_p=3, max_q=3,  # 非季节性AR/MA最大阶数
        m=SEASONAL_PERIOD,  # 季节周期（修正为180）
        seasonal=True,  # 启用季节性
        start_P=0, start_Q=0,  # 季节性AR/MA起始阶数
        max_P=2, max_Q=2,  # 季节性AR/MA最大阶数
        D=1,  # 季节性差分阶数
        trace=False,  # 关闭过程打印（如需调试可设为True）
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,  # 逐步优化（更快）
        n_jobs=n_cpu  # 并行计算
    )

    print("\nAuto ARIMA 模型总结:")
    print(auto_model.summary())

    # 提取最优参数
    order = auto_model.order  # (p, d, q)
    seasonal_order = auto_model.seasonal_order  # (P, D, Q, S)
    p, d, q = order
    P, D, Q, S = seasonal_order
    print(f"\n最优非季节性参数: p={p}, d={d}, q={q}")
    print(f"最优季节性参数: P={P}, D={D}, Q={Q}, S={S}")

    ### 步骤3: 拟合 SARIMAX 模型并诊断 ###
    print("\n--- 步骤3: 拟合 SARIMAX 模型 & 残差诊断 ---")
    try:
        model = SARIMAX(
            train['meantemp'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False, maxiter=500)  # 增加迭代防止不收敛
        print(results.summary())

        # 残差分析
        residuals = results.resid.dropna()

        # 可视化残差（4张子图）
        plt.figure(figsize=(15, 10))

        # 子图1：残差序列
        plt.subplot(2, 2, 1)
        plt.plot(residuals.index, residuals, color='blue')
        plt.title('SARIMAX模型残差序列', fontproperties=chinese_font)
        plt.axhline(y=0, color='red', linestyle='--')

        # 子图2：残差分布
        plt.subplot(2, 2, 2)
        sns.histplot(residuals, kde=True, color='green')
        plt.title('残差分布', fontproperties=chinese_font)

        # 子图3：残差ACF
        plt.subplot(2, 2, 3)
        plot_acf(residuals, lags=40, ax=plt.gca())
        plt.title('残差ACF', fontproperties=chinese_font)

        # 子图4：残差Q-Q图
        plt.subplot(2, 2, 4)
        from statsmodels.graphics.gofplots import qqplot

        qqplot(residuals, line='s', ax=plt.gca())
        plt.title('残差Q-Q图', fontproperties=chinese_font)

        plt.tight_layout()
        plt.savefig('sarimax_residuals_analysis.png', dpi=300)
        plt.show()

        # Ljung-Box 检验（白噪声验证）
        lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
        print("\nLjung-Box 检验结果:")
        print(lb_test)
        print("✅ 若p值 > 0.05，说明残差是白噪声（模型拟合良好）", fontproperties=chinese_font)

        ### 步骤4: 预测与评估 ###
        print("\n--- 步骤4: 预测结果与评估 ---")
        forecast_steps = len(test)
        forecast = results.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # 可视化预测
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train['meantemp'], label='训练数据', color='blue')
        plt.plot(test.index, test['meantemp'], label='实际值', color='green')
        plt.plot(forecast_mean.index, forecast_mean, label='预测值', color='red')
        plt.fill_between(
            conf_int.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color='pink', alpha=0.3
        )
        plt.title('SARIMAX模型预测结果', fontproperties=chinese_font)
        plt.legend(prop=chinese_font)  # 图例中文显示
        plt.savefig('sarimax_forecast.png', dpi=300)
        plt.show()

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(test['meantemp'], forecast_mean))
        print(f"测试集RMSE: {rmse:.2f}°C")

    except Exception as e:
        print(f"❌ 模型拟合/预测失败: {e}")
        print("建议：① 检查参数合理性；② 尝试增大maxiter；③ 验证数据完整性。")