# 下载黄金价格天级数据
# 下载dfii10数据
# 按日期合并
# 绘制图表：黄金价格时间曲线、 dfii10时间曲线、黄金价格参差率时间曲线、dfii10参差率时间曲线
# 
# 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import requests
from io import StringIO 

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def download_file(series_id):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = ['date', series_id]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    file_path = f'./data/{series_id}.csv'
    df.to_csv(file_path)
    return file_path

def align_daily_data(gold_file, dfii_file, days=360):
    """按天对齐黄金价格和实际利率数据，并支持按周平均"""
    
    print("正在读取数据文件...")
    
    # 读取黄金价格数据
    gold = pd.read_csv(gold_file)
    gold['Date'] = pd.to_datetime(gold['Date'], format='%m/%d/%Y')
    gold.set_index('Date', inplace=True)
    gold.rename(columns={'Value': 'Gold_Price'}, inplace=True)
    
    # 读取dfii10数据
    dfii = pd.read_csv(dfii_file)
    dfii['date'] = pd.to_datetime(dfii['date'])
    dfii.set_index('date', inplace=True)
    dfii.rename(columns={'DFII10': 'Real_Interest_Rate'}, inplace=True)
    
    print(f"黄金价格数据范围: {gold.index.min()} 到 {gold.index.max()}")
    print(f"实际利率数据范围: {dfii.index.min()} 到 {dfii.index.max()}")
    
    # 按天合并数据（使用outer join保留所有日期）
    merged_daily = pd.concat([gold, dfii], axis=1, join='outer')
    
    # 计算数据完整性统计
    total_days = len(merged_daily)
    gold_only_days = merged_daily['Gold_Price'].notna() & merged_daily['Real_Interest_Rate'].isna()
    dfii_only_days = merged_daily['Real_Interest_Rate'].notna() & merged_daily['Gold_Price'].isna()
    both_days = merged_daily['Gold_Price'].notna() & merged_daily['Real_Interest_Rate'].notna()
    
    print(f"\n数据完整性统计:")
    print(f"总天数: {total_days}")
    print(f"只有黄金价格的天数: {gold_only_days.sum()} ({gold_only_days.sum()/total_days*100:.1f}%)")
    print(f"只有实际利率的天数: {dfii_only_days.sum()} ({dfii_only_days.sum()/total_days*100:.1f}%)")
    print(f"两个数据都有的天数: {both_days.sum()} ({both_days.sum()/total_days*100:.1f}%)")
    
    # 创建两个数据集：完整数据集和配对数据集
    complete_data = merged_daily  # 包含所有日期的完整数据集
    paired_data = merged_daily.dropna()  # 只保留两个变量都有值的数据
    
    print(f"\n配对数据时间范围: {paired_data.index.min()} 到 {paired_data.index.max()}")
    print(f"配对数据天数: {len(paired_data)}")
    
    # 按周平均处理
    print("\n开始按周平均处理...")
    
    # 按周重采样（取周平均值）
    weekly_complete = complete_data.resample('W').mean()
    weekly_paired = paired_data.resample('W').mean()
    
    print(f"按周平均后数据周数: {len(weekly_paired)}")
    print(f"按周平均时间范围: {weekly_paired.index.min()} 到 {weekly_paired.index.max()}")
    
    # 筛选最近90天的数据（转换为周数）
    if len(paired_data) > 0:
        recent_days_start = paired_data.index.max() - pd.Timedelta(days=days)
        recent_days_pairedata = paired_data[paired_data.index >= recent_days_start]
        print(f"\n最近{days}天数据时间范围: {recent_days_pairedata.index.min()} 到 {recent_days_pairedata.index.max()}")
        print(f"最近{days}天数据天数: {len(recent_days_pairedata)}")
        recent_days_complete = complete_data[complete_data.index >= recent_days_start]
        
        # 按周平均的最近数据
        recent_weeks_paired = recent_days_pairedata.resample('W').mean()
        recent_weeks_complete = recent_days_complete.resample('W').mean()
    else:
        recent_days_pairedata = pd.DataFrame()
        recent_weeks_paired = pd.DataFrame()
        recent_weeks_complete = pd.DataFrame()
        print(f"没有足够的数据进行最近{days}天分析")
    
    # 保存合并后的数据
    complete_data.to_csv('/Users/nobulamb/Documents/Money-More-Money/data/merged_gold_interest_daily_complete.csv')
    paired_data.to_csv('/Users/nobulamb/Documents/Money-More-Money/data/merged_gold_interest_daily_paired.csv')
    weekly_complete.to_csv('/Users/nobulamb/Documents/Money-More-Money/data/merged_gold_interest_weekly_complete.csv')
    weekly_paired.to_csv('/Users/nobulamb/Documents/Money-More-Money/data/merged_gold_interest_weekly_paired.csv')
    
    print("\n数据已保存:")
    print("- merged_gold_interest_daily_complete.csv: 包含所有日期的完整数据")
    print("- merged_gold_interest_daily_paired.csv: 只包含两个变量都有值的日度数据")
    print("- merged_gold_interest_weekly_complete.csv: 按周平均的完整数据")
    print("- merged_gold_interest_weekly_paired.csv: 按周平均的配对数据")
    
    return {
        'daily_complete': recent_days_complete,
        'daily_paired': recent_days_pairedata,
        'weekly_complete': weekly_complete,
        'weekly_paired': weekly_paired,
        'recent_days_complete': recent_days_complete,
        'recent_days_paired': recent_days_pairedata,
        'recent_weeks_complete': recent_weeks_complete,
        'recent_weeks_paired': recent_weeks_paired
    }

def plot_simplified_dashboard(data_dict, days_window=90):
    """绘制简化的3个看板，支持时间窗口选择"""
    
    # 创建1x3的图表布局
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'黄金价格与实际利率分析（时间窗口：{days_window}天）', fontsize=16)
    
    # 获取最近指定天数的数据
    if len(data_dict['daily_paired']) > 0:
        recent_start_date = data_dict['daily_paired'].index.max() - pd.Timedelta(days=days_window)
        recent_daily_data = data_dict['daily_paired'][data_dict['daily_paired'].index >= recent_start_date]
        recent_weekly_data = data_dict['weekly_paired'][data_dict['weekly_paired'].index >= recent_start_date]
    else:
        recent_daily_data = pd.DataFrame()
        recent_weekly_data = pd.DataFrame()
    
    # 看板1：黄金价格（周平均）、实际利率（周平均）的合并曲线图
    if len(recent_weekly_data) > 0:
        ax1 = axes[0]
        # 黄金价格（左轴）
        color1 = 'gold'
        ax1.plot(recent_weekly_data.index, recent_weekly_data['Gold_Price'], 
                label='黄金价格（周平均）', color=color1, linewidth=2)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('黄金价格（美元/盎司）', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # 实际利率（右轴）
        ax1_right = ax1.twinx()
        color2 = 'blue'
        ax1_right.plot(recent_weekly_data.index, recent_weekly_data['Real_Interest_Rate'], 
                      label='实际利率（周平均）', color=color2, linewidth=2, linestyle='--')
        ax1_right.set_ylabel('实际利率（%）', color=color2)
        ax1_right.tick_params(axis='y', labelcolor=color2)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_right.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.set_title(f'看板1：黄金价格 vs 实际利率（周平均）')
    
    # 看板2：黄金价格波动率、实际利率波动率的合并曲线图
    if len(recent_weekly_data) > 1:
        ax2 = axes[1]
        
        # 计算周度环比变化率（本周 vs 上周）
        gold_volatility = recent_weekly_data['Gold_Price'].pct_change() * 100  # 百分比变化
        interest_volatility = recent_weekly_data['Real_Interest_Rate'].pct_change() * 100  # 百分比变化
        
        # 黄金价格波动率（左轴）
        color1 = 'orange'
        ax2.plot(gold_volatility.index, gold_volatility, 
                label='黄金价格周度变化率（%）', color=color1, linewidth=2)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('黄金价格周度变化率（%）', color=color1)
        ax2.tick_params(axis='y', labelcolor=color1)
        ax2.grid(True, alpha=0.3)
        
        # 实际利率波动率（右轴）
        ax2_right = ax2.twinx()
        color2 = 'darkblue'
        ax2_right.plot(interest_volatility.index, interest_volatility, 
                     label='实际利率周度变化率（%）', color=color2, linewidth=2, linestyle='--')
        ax2_right.set_ylabel('实际利率周度变化率（%）', color=color2)
        ax2_right.tick_params(axis='y', labelcolor=color2)
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax2.set_title(f'看板2：周度环比变化率分析')
    
    # 看板3：实际利率的每日曲线图
    if len(recent_daily_data) > 0:
        ax3 = axes[2]
        
        ax3.plot(recent_daily_data.index, recent_daily_data['Real_Interest_Rate'], 
                label='实际利率（日度）', color='green', linewidth=1, alpha=0.7)
        ax3.set_xlabel('日期')
        ax3.set_ylabel('实际利率（%）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title(f'看板3：实际利率日度曲线')
    
    plt.tight_layout()
    
    # 保存图表
    filename = f'/Users/nobulamb/Documents/Money-More-Money/data/dashboard_{days_window}d.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"图表已保存为: {filename}")
    
    # 输出数据统计
    if len(recent_daily_data) > 0:
        print(f"\n=== 数据统计（最近{days_window}天）===")
        print(f"日度数据天数: {len(recent_daily_data)}")
        print(f"周度数据周数: {len(recent_weekly_data)}")
        print(f"时间范围: {recent_daily_data.index.min().strftime('%Y-%m-%d')} 到 {recent_daily_data.index.max().strftime('%Y-%m-%d')}")
        
        if len(recent_weekly_data) > 1:
            correlation = recent_weekly_data['Gold_Price'].corr(recent_weekly_data['Real_Interest_Rate'])
            print(f"周度数据相关系数: {correlation:.4f}")

def generate_dashboards(data_dict, days_windows=[30, 90, 180, 365]):
    """生成多个时间窗口的看板"""
    
    for days in days_windows:
        print(f"\n正在生成 {days} 天时间窗口的看板...")
        plot_simplified_dashboard(data_dict, days)

if __name__ == "__main__":
    print("开始按天和周对齐黄金价格和实际利率数据...")
    
    # 执行数据对齐
    gold_file = '/Users/nobulamb/Documents/Money-More-Money/data/chart_20251019T082932.csv'
    dfii_file = download_file('DFII10')
    data_dict = align_daily_data(gold_file, dfii_file)
    
    # 生成简化看板（默认90天窗口）
    plot_simplified_dashboard(data_dict, 180)
    
    # 可选：生成多个时间窗口的看板
    # generate_dashboards(data_dict, days_windows=[30, 90, 180])
    
    print("\n数据分析完成！")