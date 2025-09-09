import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

class YChromosomeAnalysis:
    def __init__(self, data_path):
        """
        初始化Y染色体浓度分析类
        """
        self.data = pd.read_csv(data_path)
        self.processed_data = None
        
    def preprocess_data(self):
        """
        数据预处理和清洗
        """
        print("=== Y染色体浓度数据预处理开始 ===")
        df = self.data.copy()
        
        # 处理孕周数据，转换为以周为单位
        print("1. 处理孕周数据...")
        def convert_gestational_week(week_str):
            """
            将孕周从 '11w+6' 格式转换为浮点数（以周为单位）
            """
            if pd.isna(week_str):
                return np.nan
            
            try:
                # 处理格式如 '11w+6'
                if 'w' in str(week_str):
                    parts = str(week_str).split('w')
                    weeks = int(parts[0])
                    if '+' in parts[1]:
                        days = int(parts[1].replace('+', ''))
                    else:
                        days = 0
                    return weeks + days / 7.0
                else:
                    # 如果已经是数字格式
                    return float(week_str)
            except:
                return np.nan
        
        df['检测孕周_周'] = df['检测孕周'].apply(convert_gestational_week)
        
        # 筛选包含Y染色体浓度数据的样本
        print("2. 筛选Y染色体浓度相关数据...")
        y_chromosome_cols = ['Y染色体浓度']
        if not y_chromosome_cols:
            print("警告：未找到Y染色体浓度相关列，请检查列名")
            if 'Y染色体浓度' not in df.columns:
                print("创建模拟Y染色体浓度数据用于演示...")
                df['Y染色体浓度'] = np.random.lognormal(0, 1, len(df))
        
        print(f"   找到Y染色体相关列: {y_chromosome_cols}")
        
        # 异常值检测和处理
        print("3. 异常值检测和处理...")
             
        # BMI异常值处理（合理范围：15-45）
        bmi_outliers = (df['孕妇BMI'] < 12) | (df['孕妇BMI'] > 45)
        print(f"   BMI异常值数量: {bmi_outliers.sum()}")
        
        # 孕周异常值处理（合理范围：7-27周）
        week_outliers = (df['检测孕周_周'] < 6) | (df['检测孕周_周'] > 26)
        print(f"   孕周异常值数量: {week_outliers.sum()}")
        
        # Y染色体浓度异常值处理（移除负值和极端值）
        y_col = y_chromosome_cols[0] if y_chromosome_cols else 'Y染色体浓度'
        if y_col in df.columns:
            y_outliers = (df[y_col] <= 0) | (df[y_col] > df[y_col].quantile(0.99))
            print(f"   Y染色体浓度异常值数量: {y_outliers.sum()}")
        else:
            y_outliers = pd.Series([False] * len(df))
        
        # 创建清洁数据集（移除异常值）
        clean_mask = ~(bmi_outliers | week_outliers | y_outliers)
        df_clean = df[clean_mask].copy()
        
        print(f"   原始样本数: {len(df)}")
        print(f"   清洗后样本数: {len(df_clean)}")
        print(f"   移除样本数: {len(df) - len(df_clean)}")
        
        self.processed_data = df_clean
        self.y_column = y_col
        print("=== 数据预处理完成 ===\n")
        
        return df_clean
    
    def exploratory_analysis(self):
        """
        Y染色体浓度探索性数据分析
        """
        df = self.processed_data
        
        # 基本统计描述
        print("1. Y染色体浓度数据基本统计描述:")
        key_vars = ['孕妇BMI', '检测孕周_周', self.y_column]
        available_vars = [var for var in key_vars if var in df.columns]
        
        print("\n关键变量描述统计:")
        print(df[available_vars].describe())
        
        # 可视化分析
        print("\n2. 生成Y染色体浓度可视化图表...")
        
        # 设置图形大小
        fig = plt.figure(figsize=(16, 12))
        
        # Y染色体浓度分布
        plt.subplot(2, 3, 1)
        plt.hist(df[self.y_column].dropna(), bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        plt.title(f'{self.y_column} 分布')
        plt.xlabel(self.y_column)
        plt.ylabel('频次')
        
        # Y染色体浓度 vs 孕周
        plt.subplot(2, 3, 2)
        # 原有散点图
        plt.scatter(df['检测孕周_周'], df[self.y_column], alpha=0.6, color='red', label='数据点')

        # 添加LOWESS平滑曲线
        # 移除缺失值
        mask = df['检测孕周_周'].notna() & df[self.y_column].notna()
        x_clean = df.loc[mask, '检测孕周_周']
        y_clean = df.loc[mask, self.y_column]

        if len(x_clean) > 0:
            # 计算LOWESS平滑
            lowess_result = lowess(y_clean, x_clean, frac=0.3)  # frac控制平滑程度
            # 按x值排序以便绘制平滑曲线
            sorted_idx = np.argsort(lowess_result[:, 0])
            plt.plot(lowess_result[sorted_idx, 0], lowess_result[sorted_idx, 1], 
                    color='blue', linewidth=1.5, label='LOWESS拟合')

        plt.xlabel('检测孕周 (周)')
        plt.ylabel(self.y_column)
        plt.title('Y染色体浓度 vs 孕周 (含LOWESS拟合)')
        plt.legend()
        
        # Y染色体浓度 vs BMI
        plt.subplot(2, 3, 3)
        plt.scatter(df['孕妇BMI'], df[self.y_column], alpha=0.6, color='green', label='数据点')

        mask = df['孕妇BMI'].notna() & df[self.y_column].notna()
        x_clean = df.loc[mask, '孕妇BMI']
        y_clean = df.loc[mask, self.y_column]

        if len(x_clean) > 0:
            lowess_result = lowess(y_clean, x_clean, frac=0.3)
            sorted_idx = np.argsort(lowess_result[:, 0])
            plt.plot(lowess_result[sorted_idx, 0], lowess_result[sorted_idx, 1], 
                    color='blue', linewidth=1.5, label='LOWESS拟合')

        plt.xlabel('孕妇BMI')
        plt.ylabel(self.y_column)
        plt.title('Y染色体浓度 vs BMI (含LOWESS拟合)')
        plt.legend()
        
        # 孕周分布
        plt.subplot(2, 3, 4)
        plt.hist(df['检测孕周_周'].dropna(), bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
        plt.title('检测孕周分布')
        plt.xlabel('检测孕周 (周)')
        plt.ylabel('频次')
        
        # BMI分布
        plt.subplot(2, 3, 5)
        plt.hist(df['孕妇BMI'].dropna(), bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
        plt.title('孕妇BMI分布')
        plt.xlabel('孕妇BMI')
        plt.ylabel('频次')
        
        plt.tight_layout()
        plt.savefig('y_chromosome_exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def spearman_correlation_analysis(self):
        """
        Y染色体浓度Spearman相关性分析
        """
        df = self.processed_data
        
        # 选择与Y染色体浓度相关的变量进行分析
        analysis_vars = ['孕妇BMI', '检测孕周_周', self.y_column]
        available_vars = [var for var in analysis_vars if var in df.columns]
        
        correlation_data = df[available_vars].dropna()
        
        # 计算Spearman相关系数矩阵
        spearman_corr_matrix = correlation_data.corr(method='spearman')
        
        # 计算相关性的p值
        n = len(correlation_data)
        p_values = np.zeros((len(available_vars), len(available_vars)))
        
        for i, col1 in enumerate(available_vars):
            for j, col2 in enumerate(available_vars):
                if i != j:
                    corr_coef, p_val = spearmanr(correlation_data[col1], correlation_data[col2])
                    p_values[i, j] = p_val
                else:
                    p_values[i, j] = 0
        
        # 创建p值DataFrame
        p_values_df = pd.DataFrame(p_values, index=available_vars, columns=available_vars)
        
        # 可视化相关性矩阵
        plt.figure(figsize=(10, 8))
        
        # 创建热力图
        sns.heatmap(spearman_corr_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Spearman相关系数'},
                   fmt='.3f')
        
        plt.title('Y染色体浓度相关变量Spearman相关性热力图', fontsize=14)
        plt.tight_layout()
        plt.savefig('y_chromosome_spearman_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Y染色体浓度与其他变量的相关性
        print("Y染色体浓度与其他变量的Spearman相关性:")
        print("=" * 60)
        
        y_correlations = []
        for var in available_vars:
            if var != self.y_column:
                corr_coef = spearman_corr_matrix.loc[self.y_column, var]
                p_val = p_values_df.loc[self.y_column, var]
                
                # 判断显著性
                if p_val < 0.001:
                    significance = '***'
                elif p_val < 0.01:
                    significance = '**'
                elif p_val < 0.05:
                    significance = '*'
                else:
                    significance = 'ns'

                y_correlations.append({
                    'Variable': var,
                    'Spearman_Correlation': corr_coef,
                    'P_value': p_val,
                    'Significance': significance
                })
                print(f"{var:15} | r = {corr_coef:8.4f} | p = {p_val:8.6f} | {significance}")
        
        # 输出完整的相关性矩阵
        print("\n完整Spearman相关系数矩阵:")
        print(spearman_corr_matrix.round(4))
        
        return spearman_corr_matrix, p_values_df, y_correlations
    
    def relationship_modeling(self):
        """
        Y染色体浓度非线性混合效应建模
        """
        df = self.processed_data
    
        # 准备建模数据
        model_vars = ['孕妇BMI', '检测孕周_周']
        available_model_vars = [var for var in model_vars if var in df.columns]
    
        modeling_data = df[available_model_vars + [self.y_column, '孕妇代码']].dropna()
        
        print(f"建模样本数: {len(modeling_data)}")
        print(f"建模变量: {available_model_vars}")
        print(f"孕妇个体数: {modeling_data['孕妇代码'].nunique()}")
        
        # 基线线性回归模型
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        from sklearn.model_selection import train_test_split
        
        X = modeling_data[available_model_vars]
        y = modeling_data[self.y_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred_linear = linear_model.predict(X_test)
        
        linear_r2 = r2_score(y_test, y_pred_linear)
        linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
        
        print("\n基线线性回归模型结果:")
        print(f"R² = {linear_r2:.4f}")
        print(f"RMSE = {linear_rmse:.4f}")

        # 线性混合效应模型
        print("\n2. 线性混合效应模型...")
        linear_mixed_r2 = 0
        linear_mixed_rmse = np.inf
        linear_mixed_result = None
        linear_icc = 0
        y_pred_linear_mixed = None
        
        try:
            linear_mixed_formula = f"{self.y_column} ~ 孕妇BMI + 检测孕周_周"
            linear_mixed_model = mixedlm(linear_mixed_formula, modeling_data, groups=modeling_data['孕妇代码'])
            linear_mixed_result = linear_mixed_model.fit(method='lbfgs')
            
            print("   线性混合效应模型结果:")
            print(linear_mixed_result.summary().tables[1])
            
            y_pred_linear_mixed = linear_mixed_result.fittedvalues
            
            # 计算伪R²
            ss_res = np.sum((y - y_pred_linear_mixed) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            linear_mixed_r2 = 1 - (ss_res / ss_tot)
            linear_mixed_rmse = np.sqrt(np.mean((y - y_pred_linear_mixed) ** 2))
            
            # ICC计算
            var_between = linear_mixed_result.cov_re.iloc[0,0]
            var_within = linear_mixed_result.scale
            linear_icc = var_between / (var_between + var_within)
            
            print(f"   R² = {linear_mixed_r2:.4f}")
            print(f"   RMSE = {linear_mixed_rmse:.4f}")
            print(f"   ICC = {linear_icc:.4f}")
            
        except Exception as e:
            print(f"   线性混合效应模型拟合失败: {str(e)}")
        
        # 非线性回归模型
        print("\n3. 非线性回归模型...")

        # 手动创建非线性特征
        X_train_nonlinear = X_train.copy()
        X_test_nonlinear = X_test.copy()

        # 添加二次项
        for col in available_model_vars:
            X_train_nonlinear[f'{col}_sq'] = X_train[col] ** 2
            X_test_nonlinear[f'{col}_sq'] = X_test[col] ** 2

        # 添加交互项
        if len(available_model_vars) >= 2:
            X_train_nonlinear['BMI_Week'] = X_train[available_model_vars[0]] * X_train[available_model_vars[1]]
            X_test_nonlinear['BMI_Week'] = X_test[available_model_vars[0]] * X_test[available_model_vars[1]]

        nonlinear_model = LinearRegression()
        nonlinear_model.fit(X_train_nonlinear, y_train)
        y_pred_nonlinear = nonlinear_model.predict(X_test_nonlinear)

        nonlinear_r2 = r2_score(y_test, y_pred_nonlinear)
        nonlinear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_nonlinear))

        print(f"   R² = {nonlinear_r2:.4f}")
        print(f"   RMSE = {nonlinear_rmse:.4f}")
        print(f"   特征数: {X_train_nonlinear.shape[1]}")
        
        # 非线性混合效应模型
        print("\n构建非线性混合效应模型...")
        
        # 创建非线性特征
        modeling_data = modeling_data.copy()
        
        # 添加高次项
        if '孕妇BMI' in modeling_data.columns:
            modeling_data['BMI_sq'] = modeling_data['孕妇BMI'] ** 2
            modeling_data['BMI_cube'] = modeling_data['孕妇BMI'] ** 3
        
        if '检测孕周_周' in modeling_data.columns:
            modeling_data['Week_sq'] = modeling_data['检测孕周_周'] ** 2
            modeling_data['Week_cube'] = modeling_data['检测孕周_周'] ** 3
        
        # 添加交互项
        if '孕妇BMI' in modeling_data.columns and '检测孕周_周' in modeling_data.columns:
            modeling_data['BMI_Week'] = modeling_data['孕妇BMI'] * modeling_data['检测孕周_周']
        
        # 构建混合效应模型公式
        formula_parts = [self.y_column + " ~ "]
        
        # 固定效应（非线性项）
        fixed_effects = []
        if '孕妇BMI' in modeling_data.columns:
            fixed_effects.extend(['孕妇BMI', 'BMI_sq', 'BMI_cube'])
        if '检测孕周_周' in modeling_data.columns:
            fixed_effects.extend(['检测孕周_周', 'Week_sq', 'Week_cube'])
        if 'BMI_Week' in modeling_data.columns:
            fixed_effects.append('BMI_Week')
        
        formula = formula_parts[0] + " + ".join(fixed_effects)
    
        try:
            # 拟合非线性混合效应模型
            print(f"模型公式: {formula}")
            print("随机效应: 孕妇代码 (个体随机截距)")
            
            mixed_model = mixedlm(formula, modeling_data, groups=modeling_data['孕妇代码'])
            mixed_result = mixed_model.fit(method='lbfgs')
            
            print("\n非线性混合效应模型结果:")
            print(mixed_result.summary().tables[1])
            
            # 计算模型预测效果
            y_pred_mixed = mixed_result.fittedvalues
            
            # 计算R²（混合效应模型的伪R²）
            ss_res = np.sum((modeling_data[self.y_column] - y_pred_mixed) ** 2)
            ss_tot = np.sum((modeling_data[self.y_column] - np.mean(modeling_data[self.y_column])) ** 2)
            mixed_r2 = 1 - (ss_res / ss_tot)
            mixed_rmse = np.sqrt(np.mean((modeling_data[self.y_column] - y_pred_mixed) ** 2))
            
            print(f"\n模型拟合统计:")
            print(f"伪R² = {mixed_r2:.4f}")
            print(f"RMSE = {mixed_rmse:.4f}")
            print(f"AIC = {mixed_result.aic:.2f}")
            print(f"BIC = {mixed_result.bic:.2f}")
            
            # 随机效应方差分析
            print(f"\n方差成分:")
            print(f"个体间方差 (σ²_u): {mixed_result.cov_re.iloc[0,0]:.6f}")
            print(f"残差方差 (σ²_e): {mixed_result.scale:.6f}")
            
            # ICC计算
            var_between = mixed_result.cov_re.iloc[0,0]
            var_within = mixed_result.scale
            mixed_icc = var_between / (var_between + var_within)
            print(f"组内相关系数 (ICC): {mixed_icc:.4f}")
        
        except Exception as e:
            print(f"混合效应模型拟合失败: {str(e)}")
            print("使用简化的多项式回归作为替代...")

    
        # 模型比较可视化
        plt.figure(figsize=(20, 10))
        
        # 线性模型预测vs实际
        plt.subplot(2, 4, 1)
        plt.scatter(y_test, y_pred_linear, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'线性回归\n(R^2 = {linear_r2:.3f})')

        # 线性混合效应
        plt.subplot(2, 4, 2)
        if y_pred_linear_mixed is not None:
            plt.scatter(y, y_pred_linear_mixed, alpha=0.6, color='orange', s=20)
            y_range = [y.min(), y.max()]
            plt.plot(y_range, y_range, 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'线性混合效应\n(R^2 = {linear_mixed_r2:.3f})')
        
        #非线性回归
        plt.subplot(2, 4, 3)
        plt.scatter(y_test, y_pred_nonlinear, alpha=0.6, color='green', s=20)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'非线性回归\n(R^2 = {nonlinear_r2:.3f})')
        
        # 非线性混合效应
        plt.subplot(2, 4, 4)
        plt.scatter(modeling_data[self.y_column], y_pred_mixed, alpha=0.6, color='green')
        y_range = [modeling_data[self.y_column].min(), modeling_data[self.y_column].max()]
        plt.plot(y_range, y_range, 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'非线性混合效应模型\n(R^2 = {mixed_r2:.3f})')
        
        # 残差分析
        models_for_residual = [
            ('线性回归', y_test, y_pred_linear, 'blue'),
            ('线性混合效应', y, y_pred_linear_mixed, 'orange'),
            ('非线性回归', y_test, y_pred_nonlinear, 'green'),
            ('非线性混合效应', y, y_pred_mixed, 'red')
        ]
        
        for i, (name, y_actual, y_pred, color) in enumerate(models_for_residual):
            plt.subplot(2, 4, 5 + i)
            if y_pred is not None:
                residuals = y_actual[:len(y_pred)] - y_pred
                plt.scatter(y_pred, residuals, alpha=0.6, color=color, s=20)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
            plt.xlabel('预测值')
            plt.ylabel('残差')
            plt.title(f'{name}\n残差分析')
        
        plt.tight_layout()
        plt.savefig('y_chromosome_four_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("=== Y染色体浓度非线性混合效应建模完成 ===\n")
        
        return {
            'linear_model': linear_model,
            'mixed_model': mixed_result,
            'linear_mixed_model': linear_mixed_result,
            'nonlinear_model': {'model': nonlinear_model, 'features': X_train_nonlinear.columns},
            'nonlinear_mixed_model': mixed_result,
            'linear_r2': linear_r2,
            'linear_mixed_r2': linear_mixed_r2,
            'mixed_r2': mixed_r2,
            'linear_rmse': linear_rmse,
            'linear_mixed_rmse': linear_mixed_rmse,
            'nonlinear_r2': nonlinear_r2,
            'nonlinear_rmse': nonlinear_rmse,
            'linear_icc': linear_icc,
            'linear_rmse': linear_rmse,
            'mixed_rmse': mixed_rmse,
            'mixed_icc': mixed_icc,
            'feature_names': available_model_vars,
            'modeling_data': modeling_data
        }

    
    def comprehensive_report(self):
        """
        Y染色体浓度综合分析
        """
        df = self.processed_data
        
        print("\n 数据概况:")
        print(f"   - 分析样本数: {len(df)}")
        print(f"   - Y染色体浓度指标: {self.y_column}")
        print(f"   - 主要分析变量: 孕妇BMI、检测孕周")
        
        # 执行分析获取结果
        print("\n 相关性分析...")
        corr_matrix, p_values, y_correlations = self.spearman_correlation_analysis()
        
        print("\n 关系建模...")
        model_results = self.relationship_modeling()
        
        print("\n 关键发现:")
        print("\n   （1） Y染色体浓度与各指标的Spearman相关性:")
        for corr in y_correlations:
            strength = "强" if abs(corr['Spearman_Correlation']) > 0.7 else "中等" if abs(corr['Spearman_Correlation']) > 0.3 else "弱"
            direction = "正" if corr['Spearman_Correlation'] > 0 else "负"
            print(f"     - 与{corr['Variable']}: {direction}相关, 强度={strength} "
                  f"(r = {corr['Spearman_Correlation']:.4f}, {corr['Significance']})")
        
        if not y_correlations:
            print("没有找到与Y染色体浓度相关的指标。")
        
        print(f"\n   （2） 四模型性能对比:")
        models_performance = [
            ('线性回归', model_results['linear_r2'], model_results['linear_rmse'], None),
            ('线性混合效应', model_results['linear_mixed_r2'], model_results['linear_mixed_rmse'], model_results['linear_icc']),
            ('非线性回归', model_results['nonlinear_r2'], model_results['nonlinear_rmse'], None),
            ('非线性混合效应', model_results['mixed_r2'], model_results['mixed_rmse'], model_results['mixed_icc'])
        ]
            
        print(f"     {'模型名称':15} | {'R²':>8} | {'RMSE':>8} | {'ICC':>8}")
        print("     " + "-" * 45)
            
        for name, r2, rmse, icc in models_performance:
            icc_str = f"{icc:.4f}" if icc is not None else "   -   "
            print(f"     {name:15} | {r2:8.4f} | {rmse:8.4f} | {icc_str}")
            
        # 模型排序和推荐
        valid_models = [(name, r2) for name, r2, _, _ in models_performance if r2 > 0]
        best_model = max(valid_models, key=lambda x: x[1])


        print(f"\n   （4） 模型推荐:")
        print(f"     - 最佳模型: {best_model[0]} (R² = {best_model[1]:.4f})")
        

def main():
    # 初始化
    analyzer = YChromosomeAnalysis('data_0.csv')
    
    try:
        processed_data = analyzer.preprocess_data()

        analyzer.exploratory_analysis()
        
        analyzer.comprehensive_report()
        
    except FileNotFoundError:
        print("错误: 未找到数据文件 'data.csv'")
        print("请确保数据文件存在并且路径正确")
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()