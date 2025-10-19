import pandas as pd
import numpy as np 
import requests 
from io import StringIO 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.rcParams['figure.dpi'] = 160
# 黄金 - 影响因子建模

# 黄金价格：LBMA Gold Price PM，美元/盎司（FRED: GOLDPMGBD228NLBM）
# 实际利率：10Y TIPS收益率（FRED: DFII10）；或“名义10Y国债收益率（DGS10）− CPI同比（CPIAUCSL同比）/或通胀预期”
# 美元指数：DXY或FRED广义美元指数（TWEXB 或 DTWEXB）
# 地缘政治风险：GPR（Caldara & Iacoviello 的月度GPR总指数）
# 央行购金：世界黄金协会（WGC）央行净购金（季度，需分配至月度或直接用季度频）
# 其他可选：VIX、黄金ETF持仓（SPDR GLD）、回收与矿产供给（WGC），AISC成本（矿业公司汇总）

def fred_csv(series_id):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = ['date', series_id]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.to_csv(f'./data/{series_id}.csv')
    return df

def download_data():
    # gold = fred_csv('GOLDPMGBD228NLBM')  # 日频-下载失败
    # dgs10 = fred_csv('DGS10')             # 日频 - 美债10年预期利率 => 减去 通胀率 = 实际利率
    # cpi = fred_csv('CPIAUCSL')            # 月频 - 消费者价格指数 => 计算通胀率
    # dxy = fred_csv('DTWEXB')              # 日频（名义广义美元指数）- 
    # dfii = fred_csv('DFII10')            # 日频 - 外国债券收益率指数-10年期 约等于 实际利率
    GVZCLS = fred_csv('GVZCLS')

def MergeData():
    # 读取黄金价格数据
    gold = pd.read_csv('/Users/nobulamb/Documents/Money-More-Money/data/gld_price_100years.csv')
    # 转换日期格式（月/日/年 -> 年-月-日）
    gold['Date'] = pd.to_datetime(gold['Date'], format='%m/%d/%Y')
    gold.set_index('Date', inplace=True)
    # 按月份重采样求平均值
    gold_monthly = gold.resample('M').mean()
    gold_monthly.rename(columns={'Value': 'Gold_Price'}, inplace=True)
    
    # 读取实际利率数据
    dfii = pd.read_csv('/Users/nobulamb/Documents/Money-More-Money/data/DFII10.csv')
    # 转换日期格式
    dfii['date'] = pd.to_datetime(dfii['date'])
    dfii.set_index('date', inplace=True)
    # 按月份重采样求平均值
    dfii_monthly = dfii.resample('M').mean()
    dfii_monthly.rename(columns={'DFII10': 'Real_Interest_Rate'}, inplace=True)
    
    # 合并数据（按月份对齐）
    merged_data = pd.concat([gold_monthly, dfii_monthly], axis=1)
    # 只保留两个数据都有值的月份
    merged_data = merged_data.dropna()
    
    # 保存合并后的数据
    merged_data.to_csv('/Users/nobulamb/Documents/Money-More-Money/data/merged_gold_interest_monthly.csv')
    
    print("数据合并完成！")
    print(f"合并后的数据形状: {merged_data.shape}")
    print(f"数据时间范围: {merged_data.index.min()} 到 {merged_data.index.max()}")
    print("\n前5行数据:")
    print(merged_data.head())
    
    # === 线性拟合分析 ===
    print("\n" + "="*50)
    print("线性拟合分析：黄金价格 vs 实际利率")
    print("="*50)
    
    # 准备数据
    X = merged_data['Real_Interest_Rate']  # 自变量：实际利率
    y = merged_data['Gold_Price']          # 因变量：黄金价格
    
    # 添加常数项（截距）
    X_with_const = sm.add_constant(X)
    
    # 进行OLS线性回归
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # 输出回归结果
    print("\n回归结果摘要:")
    print(results.summary())
    
    # 提取关键统计指标
    r_squared = results.rsquared
    coefficients = results.params
    p_values = results.pvalues
    
    print(f"\n关键统计指标:")
    print(f"R-squared (拟合优度): {r_squared:.4f}")
    print(f"截距 (常数项): {coefficients['const']:.4f}")
    print(f"斜率 (实际利率系数): {coefficients['Real_Interest_Rate']:.4f}")
    print(f"实际利率系数p值: {p_values['Real_Interest_Rate']:.4f}")
    
    # 解释结果
    if p_values['Real_Interest_Rate'] < 0.05:
        significance = "显著"
    else:
        significance = "不显著"
    
    print(f"\n结果解释:")
    print(f"- 实际利率对黄金价格的影响{significance}")
    print(f"- R-squared为{r_squared:.2%}，说明模型解释了黄金价格变异的{r_squared:.2%}")
    print(f"- 实际利率每变化1个百分点，黄金价格预计变化{coefficients['Real_Interest_Rate']:.2f}美元/盎司")
    
    # 创建散点图和回归线
    plt.figure(figsize=(10, 6))
    
    # 散点图
    plt.scatter(X, y, alpha=0.6, color='blue', label='实际数据点')
    
    # 回归线
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = coefficients['const'] + coefficients['Real_Interest_Rate'] * x_line
    plt.plot(x_line, y_line, color='red', linewidth=2, label='回归线')
    
    plt.xlabel('dfii')
    plt.ylabel('gold')
    plt.title('dfii gold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig('/Users/nobulamb/Documents/Money-More-Money/data/gold_interest_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n散点图和回归线已保存为: data/gold_interest_regression.png")
    
    return merged_data, results

def predict_gold_price(interest_rate=1.7, months=2, start_date='2025-10-01'):
    """
    基于线性回归模型预测黄金价格，包含时间戳
    
    Parameters:
    interest_rate: 实际利率百分比
    months: 预测的月份数
    start_date: 预测起始日期，格式为'YYYY-MM-DD'
    
    Returns:
    包含时间戳的预测数据框
    """
    # 获取回归结果
    merged_data, results = MergeData()
    
    # 提取回归系数
    coefficients = results.params
    intercept = coefficients['const']
    slope = coefficients['Real_Interest_Rate']
    
    # 生成预测时间序列
    start_date = pd.to_datetime(start_date)
    dates = pd.date_range(start=start_date, periods=months, freq='M')
    
    # 进行预测
    predicted_prices = []
    for month in range(months):
        predicted_price = intercept + slope * interest_rate
        predicted_prices.append(predicted_price)
    
    # 创建预测结果数据框
    prediction_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Gold_Price': predicted_prices,
        'Real_Interest_Rate': interest_rate
    })
    prediction_df.set_index('Date', inplace=True)
    
    # 输出预测结果
    print("\n" + "="*60)
    print("黄金价格预测结果（基于时间序列）")
    print("="*60)
    print(f"预测起始时间: {start_date.strftime('%Y年%m月')}")
    print(f"预测条件: 实际利率保持在{interest_rate}%持续{months}个月")
    print(f"回归模型: 黄金价格 = {intercept:.2f} + {slope:.2f} × 实际利率")
    print(f"R-squared: {results.rsquared:.4f}")
    
    print(f"\n详细预测结果:")
    for i, (date, row) in enumerate(prediction_df.iterrows(), 1):
        print(f"第{i}个月 ({date.strftime('%Y年%m月')}): {row['Predicted_Gold_Price']:.2f} 美元/盎司")
    
    # 计算置信区间（95%置信水平）
    predictions = results.get_prediction(sm.add_constant([interest_rate]))
    prediction_summary = predictions.summary_frame(alpha=0.05)
    
    print(f"\n置信区间分析 (95%置信水平):")
    print(f"预测均值: {prediction_summary['mean'].iloc[0]:.2f} 美元/盎司")
    print(f"置信区间: [{prediction_summary['obs_ci_lower'].iloc[0]:.2f}, {prediction_summary['obs_ci_upper'].iloc[0]:.2f}] 美元/盎司")
    
    # 保存预测结果
    prediction_df.to_csv('/Users/nobulamb/Documents/Money-More-Money/data/gold_price_prediction_2025.csv')
    print(f"\n预测结果已保存为: data/gold_price_prediction_2025.csv")
    
    # 创建预测图表
    plt.figure(figsize=(12, 6))
    
    # 绘制历史数据
    plt.plot(merged_data.index, merged_data['Gold_Price'], 
             label='历史黄金价格', color='blue', alpha=0.7)
    
    # 绘制预测数据
    plt.plot(prediction_df.index, prediction_df['Predicted_Gold_Price'], 
             label='预测黄金价格', color='red', marker='o', linewidth=2)
    
    # 添加置信区间
    plt.fill_between(prediction_df.index, 
                    prediction_summary['obs_ci_lower'].iloc[0], 
                    prediction_summary['obs_ci_upper'].iloc[0],
                    color='red', alpha=0.2, label='95%置信区间')
    
    plt.xlabel('时间')
    plt.ylabel('黄金价格 (美元/盎司)')
    plt.title(f'黄金价格预测 (2025年10月起，实际利率{interest_rate}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 保存图表
    plt.savefig('/Users/nobulamb/Documents/Money-More-Money/data/gold_price_prediction_chart.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"预测图表已保存为: data/gold_price_prediction_chart.png")
    
    return prediction_df

def time_series_analysis():
    """
    时间序列分析：对黄金价格进行全面的时间序列建模
    """
    # 获取合并数据
    merged_data, _ = MergeData()
    
    print("\n" + "="*60)
    print("时间序列分析")
    print("="*60)
    
    # 1. 平稳性检验
    print("\n1. 平稳性检验 (ADF检验):")
    gold_price = merged_data['Gold_Price']
    
    # ADF检验
    adf_result = adfuller(gold_price.dropna())
    print(f"ADF统计量: {adf_result[0]:.4f}")
    print(f"p值: {adf_result[1]:.4f}")
    print(f"临界值:")
    for key, value in adf_result[4].items():
        print(f"  {key}: {value:.4f}")
    
    if adf_result[1] < 0.05:
        print("结论: 序列是平稳的")
    else:
        print("结论: 序列是非平稳的，需要进行差分")
    
    # 2. 时间序列分解
    print("\n2. 时间序列分解:")
    try:
        # 使用乘法模型进行分解（更适合金融时间序列）
        decomposition = seasonal_decompose(gold_price, model='multiplicative', period=12)
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(decomposition.observed)
        plt.title('原始序列')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 2)
        plt.plot(decomposition.trend)
        plt.title('趋势成分')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 3)
        plt.plot(decomposition.seasonal)
        plt.title('季节性成分')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 4)
        plt.plot(decomposition.resid)
        plt.title('残差成分')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/nobulamb/Documents/Money-More-Money/data/time_series_decomposition.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("时间序列分解图已保存")
    except Exception as e:
        print(f"时间序列分解失败: {e}")
    
    # 3. ARIMA模型
    print("\n3. ARIMA模型拟合:")
    try:
        # 自动选择ARIMA参数 (p,d,q)
        # 这里使用简单的(1,1,1)模型作为示例
        model_arima = ARIMA(gold_price, order=(1,1,1))
        results_arima = model_arima.fit()
        
        print("ARIMA(1,1,1)模型结果:")
        print(results_arima.summary())
        
        # 预测未来2个月
        forecast_arima = results_arima.forecast(steps=2)
        print(f"ARIMA模型预测结果:")
        for i, price in enumerate(forecast_arima, 1):
            print(f"  第{i}个月: {price:.2f} 美元/盎司")
            
    except Exception as e:
        print(f"ARIMA模型拟合失败: {e}")
    
    # 4. VAR模型（向量自回归）
    print("\n4. VAR模型 (向量自回归):")
    try:
        # 准备VAR模型数据
        var_data = merged_data[['Gold_Price', 'Real_Interest_Rate']].dropna()
        
        # 选择最佳滞后阶数
        model_var = VAR(var_data)
        lag_order = model_var.select_order(maxlags=12)
        best_lag = lag_order.aic
        
        print(f"最佳滞后阶数 (AIC准则): {best_lag}")
        
        # 拟合VAR模型
        results_var = model_var.fit(best_lag)
        print("VAR模型结果:")
        print(results_var.summary())
        
        # 预测未来2个月
        forecast_var = results_var.forecast(var_data.values[-best_lag:], steps=2)
        forecast_dates = pd.date_range(start=var_data.index[-1] + pd.DateOffset(months=1), 
                                      periods=2, freq='M')
        
        print(f"VAR模型预测结果:")
        for i, (date, values) in enumerate(zip(forecast_dates, forecast_var), 1):
            gold_pred, interest_pred = values
            print(f"  第{i}个月 ({date.strftime('%Y年%m月')}):")
            print(f"    黄金价格: {gold_pred:.2f} 美元/盎司")
            print(f"    实际利率: {interest_pred:.2f}%")
            
    except Exception as e:
        print(f"VAR模型拟合失败: {e}")
    
    # 5. 格兰杰因果关系检验
    print("\n5. 格兰杰因果关系检验:")
    try:
        # 准备数据（需要平稳序列）
        test_data = merged_data[['Gold_Price', 'Real_Interest_Rate']].dropna()
        
        # 进行格兰杰检验（最大滞后4期）
        granger_test = grangercausalitytests(test_data, maxlag=4, verbose=False)
        
        print("格兰杰因果关系检验结果 (实际利率 -> 黄金价格):")
        for lag in range(1, 5):
            p_value = granger_test[lag][0]['ssr_chi2test'][1]
            print(f"  滞后{lag}期: p值 = {p_value:.4f}", 
                  "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else "")
            
    except Exception as e:
        print(f"格兰杰检验失败: {e}")
    
    return merged_data

def advanced_gold_prediction(interest_rate=1.75, months=5, start_date='2025-10-01'):
    """
    高级黄金价格预测：结合时间序列模型和回归模型
    """
    print("\n" + "="*70)
    print("高级黄金价格预测（时间序列 + 回归模型）")
    print("="*70)
    
    # 获取数据和时间序列分析结果
    merged_data = time_series_analysis()
    
    # 同时使用传统回归模型进行预测
    _, regression_results = MergeData()
    coefficients = regression_results.params
    
    # 生成预测时间序列
    start_date = pd.to_datetime(start_date)
    dates = pd.date_range(start=start_date, periods=months, freq='M')
    
    # 多种预测方法的结果
    predictions = {
        '回归模型': [],
        'ARIMA模型': [],
        '组合预测': []
    }
    
    # 回归模型预测
    for month in range(months):
        reg_pred = coefficients['const'] + coefficients['Real_Interest_Rate'] * interest_rate
        predictions['回归模型'].append(reg_pred)
    
    # ARIMA模型预测（需要重新拟合最新数据）
    try:
        gold_price = merged_data['Gold_Price']
        model_arima = ARIMA(gold_price, order=(1,1,1))
        results_arima = model_arima.fit()
        arima_forecast = results_arima.forecast(steps=months)
        predictions['ARIMA模型'] = arima_forecast.tolist()
    except:
        predictions['ARIMA模型'] = predictions['回归模型'].copy()
    
    # 组合预测（加权平均）
    for i in range(months):
        combined = 0.6 * predictions['回归模型'][i] + 0.4 * predictions['ARIMA模型'][i]
        predictions['组合预测'].append(combined)
    
    # 创建预测结果数据框
    prediction_df = pd.DataFrame({
        'Date': dates,
        '回归模型预测': predictions['回归模型'],
        'ARIMA模型预测': predictions['ARIMA模型'],
        '组合预测': predictions['组合预测'],
        '实际利率假设': interest_rate
    })
    prediction_df.set_index('Date', inplace=True)
    
    # 输出详细预测结果
    print(f"\n预测条件: 从{start_date.strftime('%Y年%m月')}开始，实际利率保持在{interest_rate}%持续{months}个月")
    print(f"\n详细预测结果:")
    for i, (date, row) in enumerate(prediction_df.iterrows(), 1):
        print(f"\n第{i}个月 ({date.strftime('%Y年%m月')}):")
        print(f"  回归模型: {row['回归模型预测']:.2f} 美元/盎司")
        print(f"  ARIMA模型: {row['ARIMA模型预测']:.2f} 美元/盎司")
        print(f"  组合预测: {row['组合预测']:.2f} 美元/盎司")
    
    # 保存预测结果
    prediction_df.to_csv('/Users/nobulamb/Documents/Money-More-Money/data/advanced_gold_prediction.csv')
    print(f"\n预测结果已保存为: data/advanced_gold_prediction.csv")
    
    # 创建对比图表
    plt.figure(figsize=(14, 8))
    
    # 历史数据
    plt.plot(merged_data.index, merged_data['Gold_Price'], 
             label='历史黄金价格', color='blue', alpha=0.7, linewidth=2)
    
    # 不同模型的预测
    colors = ['red', 'green', 'purple']
    models = ['回归模型预测', 'ARIMA模型预测', '组合预测']
    
    for i, model in enumerate(models):
        plt.plot(prediction_df.index, prediction_df[model], 
                label=model, color=colors[i], marker='o', linewidth=2)
    
    plt.xlabel('时间')
    plt.ylabel('黄金价格 (美元/盎司)')
    plt.title(f'黄金价格多模型预测对比 (2025年10月起)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.savefig('/Users/nobulamb/Documents/Money-More-Money/data/advanced_prediction_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"预测对比图已保存为: data/advanced_prediction_comparison.png")
    
    return prediction_df

def ARIMA():
    # 1. 数据读取与预处理
    df = pd.read_csv('/Users/nobulamb/Documents/Money-More-Money/data/merged_gold_interest_monthly.csv')
    
    # 修复：确保Date列转换为datetime格式
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 2. ARIMAX建模
    model_arimax = sm.tsa.ARIMA(
        endog=df['Gold_Price'], 
        exog=df['Real_Interest_Rate'],
        order=(1,1,1)
    )
    results = model_arimax.fit()  

    future_rates = [3, 4, 5, 6, 5, 5]  # 示例数据（单位：%）
    future_dates = pd.date_range(start='2025-10-31', periods=6, freq='M')  # 生成未来日期  

    forecast = results.get_forecast(
        steps=6,  # 预测6期
        exog=future_rates  # 传入未来实际利率
    )

    pred_mean = forecast.predicted_mean          # 点预测值（金价）
    conf_int = forecast.conf_int(alpha=0.05)     # 95%置信区间

    # === 新增：逐行输出预测结果 ===
    print("\\n" + "="*50)
    print("ARIMA模型预测结果（逐月输出）")
    print("="*50)
    
    for i, (date, price, lower, upper) in enumerate(zip(future_dates, pred_mean, conf_int.iloc[:, 0], conf_int.iloc[:, 1]), 1):
        print(f"第{i}个月 ({date.strftime('%Y年%m月')}):")
        print(f"  预测金价: {price:.2f} 美元/盎司")
        print(f"  95%置信区间: [{lower:.2f}, {upper:.2f}] 美元/盎司")
        print(f"  假设实际利率: {future_rates[i-1]}%")
        print("-" * 40)

    # 输出模型统计信息
    print(f"\\n模型统计信息:")
    print(f"AIC: {results.aic:.2f}")
    print(f"BIC: {results.bic:.2f}")
    print(f"HQIC: {results.hqic:.2f}")

    import matplotlib.pyplot as plt

    # 修复绘图部分：使用正确的日期格式
    plt.figure(figsize=(12, 6))
    
    # 绘制历史金价 - 使用df.index（已经是datetime）
    plt.plot(df.index, df['Gold_Price'], label='历史数据', color='blue')

    # 绘制预测金价及置信区间
    plt.plot(future_dates, pred_mean, label='预测值', color='red', linestyle='--', marker='o')
    plt.fill_between(
        future_dates,
        conf_int.iloc[:, 0],  # 置信区间下限
        conf_int.iloc[:, 1],  # 置信区间上限
        color='pink', alpha=0.3, label='95%置信区间'
    )

    plt.title('黄金价格预测（基于实际利率）')
    plt.xlabel('时间')
    plt.ylabel('金价 (美元/盎司)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 保存图表
    plt.savefig('/Users/nobulamb/Documents/Money-More-Money/data/arima_prediction.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("ARIMA预测图表已保存为: data/arima_prediction.png")

# 调用高级预测函数
# advanced_results = advanced_gold_prediction(interest_rate=1.7, months=6, start_date='2025-10-01')
# ARIMA()
download_data()