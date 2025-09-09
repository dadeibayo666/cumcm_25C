import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from scipy import stats
from scipy.stats import spearmanr
from scipy.optimize import minimize
from rapidfuzz import process
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import itertools

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MultiFactorNIPTOptimizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.significant_factors = {}
        self.factor_weights = {}
        self.comprehensive_factor = None
        self.optimal_groups = None
        self.optimal_timepoints = {}
        self.pareto_frontier = {}
        self.error_analysis_results = {}
        self.target_variable = 'Y染色体浓度'
        self.correlation_matrix = None   # >>> 新增：保存相关矩阵
        # >>> 新增 BMI 聚类信息
        self.bmi_elbow_k = None
        self.bmi_intervals = None

    def load_and_preprocess_data(self):
        print("=== 多因素数据预处理开始 ===")
        try:
            self.raw_data = pd.read_csv(self.data_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.raw_data = pd.read_csv(self.data_path, encoding='gbk')
            except UnicodeDecodeError:
                self.raw_data = pd.read_csv(self.data_path, encoding='latin-1')
        print(f"原始数据量: {len(self.raw_data)}")
        self.raw_data = self.raw_data.loc[:, ~self.raw_data.columns.str.contains('^Unnamed')]
        column_mapping = {'检测孕周': '孕周', 'IVF妊娠': '妊娠方式'}
        self.raw_data = self.raw_data.rename(columns=column_mapping)

        male_data = self.raw_data[self.raw_data['Y染色体浓度'].notna()].copy()
        male_data['孕周_数值'] = male_data['孕周'].apply(self._convert_gestational_week)
        male_data = male_data[
            (male_data['孕周_数值'] >= 10) &
            (male_data['孕周_数值'] <= 25) &
            (male_data['孕周_数值'].notna())
        ].copy()
        male_data['孕周_离散'] = (male_data['孕周_数值'] * 2).round() / 2

        required_columns = ['孕妇BMI', '孕周_离散', 'Y染色体浓度']
        optional_columns = ['年龄', '怀孕次数', '生产次数', '妊娠方式', '身高', '体重']

        for col in required_columns:
            if col in male_data.columns:
                male_data[col] = pd.to_numeric(male_data[col], errors='coerce')

        for col in optional_columns:
            if col in male_data.columns:
                if col == '妊娠方式':
                    pregnancy_mapping = {'自然受孕': 0, 'IUI（人工授精）': 1, 'IVF（试管婴儿）': 1}
                    male_data[col] = male_data[col].map(pregnancy_mapping).fillna(0).astype(int)
                elif col == '怀孕次数':
                    def convert_pregnancy_times(value):
                        if pd.isna(value): return np.nan
                        s = str(value).strip()
                        if s in ['1','2']: return int(s)
                        if s in ['≥3','>=3'] or s in ['3','4','5','6','7','8','9']:
                            return 3
                        import re
                        nums = re.findall(r'\d+', s)
                        if nums:
                            return min(int(nums[0]), 3)
                        return np.nan
                    male_data[col] = male_data[col].apply(convert_pregnancy_times)
                else:
                    male_data[col] = pd.to_numeric(male_data[col], errors='coerce')

        available_columns = required_columns.copy()
        for col in optional_columns:
            if col in male_data.columns:
                available_columns.append(col)

        self.processed_data = male_data[available_columns].dropna().copy()

        print(f"\n最终处理数据量: {len(self.processed_data)}")
        print(f"可用变量: {available_columns}")
        print("=== 数据预处理完成 ===\n")
        return self.processed_data

    def _convert_gestational_week(self, week_str):
        try:
            if 'w' in str(week_str):
                parts = str(week_str).split('w')
                weeks = int(parts[0])
                days = int(parts[1].replace('+', '')) if '+' in parts[1] else 0
                return weeks + days / 7.0
            else:
                return float(week_str)
        except:
            return np.nan

    # >>> 新增：相关系数热力图方法
    def plot_correlation_heatmap(self, method='spearman', save=True, figsize=(9, 7)):
        """
        计算并绘制 Spearman 相关系数热力图（含 Y染色体浓度）。
        """
        if self.processed_data is None:
            raise ValueError("请先调用 load_and_preprocess_data()")
        numeric_df = self.processed_data.select_dtypes(include=[np.number]).copy()

        # 避免单调常数列
        drop_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() < 3]
        if drop_cols:
            numeric_df = numeric_df.drop(columns=drop_cols)

        corr = numeric_df.corr(method=method)
        self.correlation_matrix = corr

        print("=== 相关系数矩阵 (前10行) ===")
        print(corr.head(10))

        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                    annot=True, fmt=".2f", square=True, cbar_kws={'shrink': .8})
        plt.title(f'{method.capitalize()} 相关系数热力图')
        plt.tight_layout()
        if save:
            fname = f"correlation_heatmap_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"热力图已保存: {fname}")
        plt.show()
        return corr

    def stepwise_regression_analysis(self, alpha=0.05):
        """
        Spearman 相关分析（替代逐步回归）
        """
        print("=== Spearman 相关分析开始（替代逐步回归） ===")
        if self.processed_data is None:
            raise ValueError("请先执行 load_and_preprocess_data()")
        target = self.target_variable
        if target not in self.processed_data.columns:
            raise ValueError(f"缺少目标变量列: {target}")

        candidate_vars = [c for c in [
            '孕妇BMI', '孕周_离散', '年龄', '怀孕次数', '生产次数', '妊娠方式', '身高', '体重'
        ] if c in self.processed_data.columns and c != target]

        print("候选变量:", candidate_vars)
        results = []
        for var in candidate_vars:
            try:
                x = self.processed_data[var].astype(float)
                y = self.processed_data[target].astype(float)
                if x.nunique() < 3:
                    print(f"  {var}: 唯一值过少，跳过")
                    continue
                rho, p = spearmanr(x, y)
                print(f"  {var}: Spearman ρ = {rho:.4f}, p = {p:.4g}")
                if not np.isnan(rho):
                    results.append((var, rho, p))
            except Exception as e:
                print(f"  {var}: 计算失败 - {e}")

        sig = [(v, r, p) for v, r, p in results if p < alpha]
        sig.sort(key=lambda x: abs(x[1]), reverse=True)

        if not sig:
            print(f"\n没有变量在显著性水平 α={alpha} 下显著。")
            self.selected_vars = []
            self.final_coefficients = np.array([])
            self.significant_factors = {}
        else:
            print(f"\n显著变量 (p < {alpha}):")
            self.selected_vars = [v for v, _, _ in sig]
            self.final_coefficients = np.array([r for _, r, _ in sig])
            self.significant_factors = {
                v: {'coefficient': r, 'p_value': p, 'importance': abs(r)}
                for v, r, p in sig
            }
            for v, r, p in sig:
                print(f"  {v}: ρ = {r:.4f}, p = {p:.4g}")

        print("=== Spearman 相关分析完成 ===\n")
        # 自动绘制热力图
        try:
            self.plot_correlation_heatmap(method='spearman', save=True)
        except Exception as e:
            print(f"相关热力图绘制失败: {e}")
        return self.significant_factors

    def calculate_factor_weights(self):
        print("=== 计算因素权重开始 ===")
        if not hasattr(self, 'selected_vars') or not self.selected_vars:
            print("警告: 没有找到显著的影响因素")
            self.factor_weights = {}
            return
        coeffs_abs = [abs(coef) for coef in self.final_coefficients]
        total = sum(coeffs_abs)
        if total == 0:
            print("警告: 所有相关系数为0")
            self.factor_weights = {}
            return
        weights = [c / total for c in coeffs_abs]
        self.factor_weights = dict(zip(self.selected_vars, weights))
        print("各因素权重 (|Spearman ρ| 归一化):")
        for v, w in self.factor_weights.items():
            print(f"  {v}: {w:.1%}")
        print(f"\n权重总和验证: {sum(weights):.3f}")
        self._plot_factor_weights()
        print("=== 计算因素权重完成 ===")

    def _plot_factor_weights(self):
        if not self.factor_weights:
            return
        factors = list(self.factor_weights.keys())
        weights = list(self.factor_weights.values())
        angles = np.linspace(0, 2*np.pi, len(factors), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        weights = weights + [weights[0]]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, weights, 'o-', linewidth=2)
        ax.fill(angles, weights, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(factors)
        ax.set_ylim(0, max(weights) * 1.1 if weights else 1)
        ax.set_title('多因素权重雷达图 (Spearman)', pad=20)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"factor_weights_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def construct_comprehensive_factor(self):
        print("=== 构建综合影响因子开始 ===")
        if not self.factor_weights:
            self.calculate_factor_weights()
        if not self.factor_weights:
            print("无法构建综合因子：没有显著变量")
            return None
        scaler = StandardScaler()
        selected_vars = list(self.factor_weights.keys())
        for var in selected_vars:
            self.processed_data[f'{var}_标准化'] = scaler.fit_transform(
                self.processed_data[[var]]
            ).flatten()
        K_w = np.zeros(len(self.processed_data))
        for var, w in self.factor_weights.items():
            K_w += w * self.processed_data[f'{var}_标准化']
        self.processed_data['综合影响因子'] = K_w
        self.comprehensive_factor = K_w
        print("综合影响因子构建完成")
        print(f"K_w 范围: [{K_w.min():.4f}, {K_w.max():.4f}]")
        print(f"K_w 均值: {K_w.mean():.4f} ± {K_w.std():.4f}")
        print("=== 构建综合影响因子完成 ===\n")
        return K_w

    # >>> 修改：增强K-均值聚类 - 增加BMI肘部法则区间划分与输出（最小改动保留原逻辑）


    def enhanced_clustering(self, bmi_elbow=True, bmi_k_range=(2,8)):
        print("=== 增强K-均值聚类开始 ===")
        if self.comprehensive_factor is None:
            self.construct_comprehensive_factor()
        if self.comprehensive_factor is None:
            print("无法聚类：缺少综合影响因子")
            return None
        
        K_w = self.processed_data['综合影响因子'].values.reshape(-1, 1)
        optimal_k = self._determine_optimal_k(K_w)
        self.processed_data['聚类标签'] = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(K_w)
        self.optimal_k = optimal_k

        if bmi_elbow and '孕妇BMI' in self.processed_data.columns:
            print("\n--- BMI 区间划分 ---")
            self._determine_bmi_intervals(bmi_k_range)

        print("=== 增强K-均值聚类完成 ===\n")
        return self.processed_data['聚类标签']

    def _determine_optimal_k(self, K_w):
        k_range_cf = list(range(2, min(8, len(self.processed_data)//50 + 1)))
        if len(k_range_cf) == 0:
            k_range_cf = [2, 3]  # 兜底
        inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(K_w).inertia_ for k in k_range_cf]

        if len(k_range_cf) >= 2:
            distances = self._calculate_elbow_distances(k_range_cf, inertias)
            elbow_index = int(np.argmax(distances))  # 选择WCSS最小的k值
            optimal_k = k_range_cf[elbow_index]
            print(f"[综合因子] K值范围: {k_range_cf}")
            print("WCSS(Inertia):", [round(v, 2) for v in inertias])
            print("肘部距离:", [round(d, 4) for d in distances])
            print(f"[综合因子] 肘部法则选定最优K = {optimal_k}")
        else:
            optimal_k = k_range_cf[0]
            print(f"[综合因子] 可选K过少，直接取 {optimal_k}")
        
        return optimal_k

    def _calculate_elbow_distances(self, k_range, inertias):
        x1, y1 = k_range[0], inertias[0]
        x2, y2 = k_range[-1], inertias[-1]
        distances = []
        for k_val, inertia in zip(k_range, inertias):
            num = abs((y2 - y1) * k_val - (x2 - x1) * inertia + x2 * y1 - y2 * x1)
            den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(num / den if den != 0 else 0)
        return distances

    def _determine_bmi_intervals(self, bmi_k_range):
        bmi_vals = self.processed_data['孕妇BMI'].values.reshape(-1, 1)
        uniq = np.unique(bmi_vals)
        if len(uniq) >= 3:
            k_min = max(2, bmi_k_range[0])
            k_max = min(bmi_k_range[1], len(uniq) - 1)
            ks = list(range(k_min, k_max + 1))
            inertias_bmi = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(bmi_vals).inertia_ for k in ks]

            distances_bmi = self._calculate_elbow_distances(ks, inertias_bmi)
            elbow_idx_bmi = int(np.argmax(distances_bmi))
            self.bmi_elbow_k = ks[elbow_idx_bmi]
            bmi_labels = KMeans(n_clusters=self.bmi_elbow_k, random_state=42, n_init=10).fit_predict(bmi_vals)
            self.processed_data['BMI聚类标签'] = bmi_labels
            self._calculate_bmi_intervals(bmi_vals)
        else:
            print("BMI 唯一值过少，跳过 BMI 肘部法则划分")

    def _calculate_bmi_intervals(self, bmi_vals):
        centers = KMeans(n_clusters=self.bmi_elbow_k, random_state=42, n_init=10).fit(bmi_vals).cluster_centers_.flatten()
        order = np.argsort(centers)
        centers_sorted = centers[order]
        boundaries = [(centers_sorted[i] + centers_sorted[i + 1]) / 2 for i in range(len(centers_sorted) - 1)]
        lower, upper = bmi_vals.min(), bmi_vals.max()
        edges = sorted([lower] + boundaries + [upper])
        intervals = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        intervals[-1] = (intervals[-1][0], float('inf'))
        self.bmi_intervals = intervals

        print("\nBMI 分组区间（右开，最后∞）:")
        for idx, (a, b) in enumerate(intervals):
            print(f"  聚类区间 {idx+1}: [{a:.2f}, {'∞' if np.isinf(b) else f'{b:.2f}'} )")
        print("\nBMI 分组区间样本量：")
        for idx, (a, b) in enumerate(intervals):
            cnt = (self.processed_data['孕妇BMI'] >= a).sum() if np.isinf(b) else ((self.processed_data['孕妇BMI'] >= a) & (self.processed_data['孕妇BMI'] < b)).sum()
            print(f"  聚类区间 {idx+1}: n={cnt}")

    def construct_comprehensive_scoring_function(self):
        print("=== 构建综合评分函数开始 ===")
        def calculate_metrics_for_timepoint(cluster_data, timepoint, threshold=4.0):
            subset = cluster_data[cluster_data['孕周_离散'] == timepoint]
            if len(subset) < 3: return None
            y_conc = subset['Y染色体浓度']
            if y_conc.max() <= 1: threshold = 0.04
            mean_conc = y_conc.mean()
            std_conc = y_conc.std()
            success_rate = (y_conc >= threshold).mean()
            n = len(y_conc)
            conf_lower = mean_conc - 1.96 * (std_conc / np.sqrt(n))
            time_risk = self._calculate_time_risk(timepoint)
            detection_risk = 1 - success_rate
            total_risk = 0.6 * time_risk + 0.4 * detection_risk
            return {
                'timepoint': timepoint,
                'sample_size': n,
                'mean_concentration': mean_conc,
                'std_concentration': std_conc,
                'success_rate': success_rate,
                'confidence_lower': conf_lower,
                'time_risk': time_risk,
                'detection_risk': detection_risk,
                'total_risk': total_risk,
                'score': (1/(total_risk+0.01))*success_rate
            }

        if not hasattr(self, 'optimal_k'):
            print("尚未聚类，先执行 enhanced_clustering()")
            return None
        timepoints = np.arange(10, 25.5, 0.5)
        self.timepoint_metrics = {}
        for c in range(self.optimal_k):
            cluster_data = self.processed_data[self.processed_data['聚类标签'] == c]
            metrics = []
            for t in timepoints:
                m = calculate_metrics_for_timepoint(cluster_data, t)
                if m: metrics.append(m)
            self.timepoint_metrics[c] = metrics
            print(f"聚类 G'{c+1}: 计算 {len(metrics)} 个有效时点")
        print("=== 综合评分函数构建完成 ===\n")
        return self.timepoint_metrics

    def _calculate_time_risk(self, timepoint):
        if timepoint <= 12: return 1.0
        elif 12 < timepoint <= 25: return 3.0
        else: return 5.0

    def multi_constraint_optimization(self):
        print("=== 多重约束优化开始 ===")
        if not hasattr(self, 'timepoint_metrics'):
            self.construct_comprehensive_scoring_function()
        threshold = 0.04 if self.processed_data['Y染色体浓度'].max() <= 1 else 4.0
        optimal_results = {}
        for c in range(self.optimal_k):
            metrics = self.timepoint_metrics.get(c, [])
            if not metrics: continue
            feasible = []
            for m in metrics:
                cv = m['std_concentration']/m['mean_concentration'] if m['mean_concentration']>0 else float('inf')
                cond1 = m['confidence_lower'] >= threshold
                cond2 = cv <= 0.15
                cond3 = m['success_rate'] >= 0.8
                if cond1 and cond2 and cond3:
                    feasible.append(m)
            if feasible:
                best = max(feasible, key=lambda x: x['score']); level='full'
            else:
                relaxed = [m for m in metrics if m['success_rate'] >= 0.7]
                if relaxed:
                    best = max(relaxed, key=lambda x: x['score']); level='relaxed'
                else:
                    best = max(metrics, key=lambda x: x['score']); level='minimal'
            optimal_results[c] = {
                'optimal_timepoint': best['timepoint'],
                'success_rate': best['success_rate'],
                'total_risk': best['total_risk'],
                'mean_concentration': best['mean_concentration'],
                'confidence_lower': best['confidence_lower'],
                'sample_size': best['sample_size'],
                'constraint_level': level,
                'score': best['score']
            }
        self.optimal_timepoints = optimal_results
        print("多重约束优化结果:")
        for c, r in optimal_results.items():
            print(f"\n聚类 G'{c+1}组:")
            print(f"  最佳时点: {r['optimal_timepoint']:.2f}周")
            print(f"  达标率: {r['success_rate']:.1%}")
            print(f"  总风险: {r['total_risk']:.3f}")
            print(f"  约束级别: {r['constraint_level']}")
            print(f"  综合评分: {r['score']:.3f}")
        print("=== 多重约束优化完成 ===\n")
        return optimal_results

    def pareto_frontier_analysis(self):
        print("=== 帕累托前沿分析开始 ===")
        if not hasattr(self, 'timepoint_metrics'):
            self.construct_comprehensive_scoring_function()
        self.pareto_results = {}
        for c in range(self.optimal_k):
            metrics = self.timepoint_metrics.get(c, [])
            if not metrics: continue
            risks = [m['total_risk'] for m in metrics]
            success = [m['success_rate'] for m in metrics]
            tps = [m['timepoint'] for m in metrics]
            pareto_pts = []; pareto_tps=[]
            for i,(r,s) in enumerate(zip(risks, success)):
                dominated=False
                for j,(r2,s2) in enumerate(zip(risks, success)):
                    if j!=i and (r2<=r and s2>=s) and (r2<r or s2>s):
                        dominated=True; break
                if not dominated:
                    pareto_pts.append((r,s)); pareto_tps.append(tps[i])
            order = np.argsort([p[0] for p in pareto_pts])
            pareto_pts = [pareto_pts[i] for i in order]
            pareto_tps = [pareto_tps[i] for i in order]
            self.pareto_results[c] = {
                'pareto_points': pareto_pts,
                'pareto_timepoints': pareto_tps,
                'all_risks': risks,
                'all_success_rates': success,
                'all_timepoints': tps
            }
            print(f"聚类 G'{c+1}组帕累托前沿点数: {len(pareto_pts)}")
        self._plot_pareto_frontiers()
        print("=== 帕累托前沿分析完成 ===\n")
        return self.pareto_results

    def _plot_pareto_frontiers(self):
        if not hasattr(self, 'pareto_results'): return
        n = len(self.pareto_results)
        cols = min(3, n)
        rows = (n + cols - 1)//cols
        colors = plt.cm.Set3(np.linspace(0,1,n))
        
        # 计算需要的图形数量
        num_figures = (n + 1) // 2
        
        for fig_num in range(num_figures):
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            for i in range(2):
                cluster_index = fig_num * 2 + i
                if cluster_index >= n: break
                ax = axes[i]
                cluster = cluster_index
                data = self.pareto_results[cluster]
                ax.scatter(data['all_risks'], data['all_success_rates'],
                           alpha=0.6, color='lightgray', s=30, label='所有时点')
                pr = [p[0] for p in data['pareto_points']]
                ps = [p[1] for p in data['pareto_points']]
                ax.scatter(pr, ps, color=colors[cluster], s=100, edgecolors='black',
                           linewidth=2, label='帕累托前沿', zorder=5)
                if len(pr) > 1:
                    ax.plot(pr, ps, 'r--', lw=2, alpha=0.7)
                if cluster in self.optimal_timepoints:
                    opt = self.optimal_timepoints[cluster]
                    ax.scatter(opt['total_risk'], opt['success_rate'], marker='*',
                               s=200, color='red', edgecolors='black', linewidth=2,
                               label=f"最优 ({opt['optimal_timepoint']:.2f}周)", zorder=10)
                ax.set_xlabel('总风险'); ax.set_ylabel('达标率')
                ax.set_title(f"聚类 G'{cluster+1}组 风险-达标率帕累托前沿")
                ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"pareto_frontiers_{fig_num+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        dpi=300, bbox_inches='tight')
            plt.show()

    def detection_error_analysis(self, n_simulations=2000):
        print("=== 检测误差影响分析开始 ===")
        if not self.optimal_timepoints:
            self.multi_constraint_optimization()
        error_results = {}
        error_levels = [0.05,0.10,0.15,0.20]
        for cluster in self.optimal_timepoints.keys():
            optimal_time = self.optimal_timepoints[cluster]['optimal_timepoint']
            cluster_data = self.processed_data[
                (self.processed_data['聚类标签']==cluster) &
                (self.processed_data['孕周_离散']==optimal_time)
            ]
            if len(cluster_data) < 5:
                print(f"聚类 G'{cluster+1}组样本量不足，跳过误差分析")
                continue
            original_conc = cluster_data['Y染色体浓度'].values
            original_success = self.optimal_timepoints[cluster]['success_rate']
            threshold = 0.04 if original_conc.max() <= 1 else 4.0
            impact = {}
            for level in error_levels:
                sims=[]
                for _ in range(n_simulations):
                    noise = np.random.normal(0, level*np.mean(original_conc), len(original_conc))
                    noisy = np.maximum(original_conc+noise, 0)
                    sims.append((noisy>=threshold).mean())
                sims = np.array(sims)
                impact[level] = {
                    'mean_success_rate': sims.mean(),
                    'std_success_rate': sims.std(),
                    'confidence_lower': np.percentile(sims, 2.5),
                    'confidence_upper': np.percentile(sims, 97.5),
                    'success_rate_decrease': original_success - sims.mean(),
                    'robust_probability': (sims >= 0.8).mean()
                }
            error_results[cluster] = {
                'original_success_rate': original_success,
                'optimal_timepoint': optimal_time,
                'sample_size': len(cluster_data),
                'error_analysis': impact
            }
        self.error_analysis_results = error_results
        print("\n检测误差影响分析结果:")
        print("="*80)
        for cluster, res in error_results.items():
            print(f"\n聚类 G'{cluster+1}组 (最优时点: {res['optimal_timepoint']:.2f}周):")
            print(f"原始达标率: {res['original_success_rate']:.1%}")
            print("误差水平     模拟达标率     95%置信区间     鲁棒性概率")
            print("-"*60)
            for lvl, imp in res['error_analysis'].items():
                print(f"{lvl:8.0%}     {imp['mean_success_rate']:8.1%}    "
                      f"[{imp['confidence_lower']:.1%}, {imp['confidence_upper']:.1%}]     "
                      f"{imp['robust_probability']:8.1%}")
        self._plot_error_analysis()
        print("\n=== 检测误差影响分析完成 ===\n")
        return error_results

    def _plot_error_analysis(self):
        if not self.error_analysis_results: return
        fig,(ax1,ax2)=plt.subplots(1,2, figsize=(15,6))
        error_levels=[0.05,0.10,0.15,0.20]
        colors = plt.cm.Set3(np.linspace(0,1,len(self.error_analysis_results)))
        for i,(cluster,res) in enumerate(self.error_analysis_results.items()):
            mean_rates=[res['error_analysis'][lvl]['mean_success_rate'] for lvl in error_levels]
            ci_lower=[res['error_analysis'][lvl]['confidence_lower'] for lvl in error_levels]
            ci_upper=[res['error_analysis'][lvl]['confidence_upper'] for lvl in error_levels]
            ax1.plot(error_levels, mean_rates, 'o-', color=colors[i], label=f"G'{cluster+1}组")
            ax1.fill_between(error_levels, ci_lower, ci_upper, alpha=0.3, color=colors[i])
        ax1.axhline(0.8, color='red', ls='--', alpha=0.7)
        ax1.set_xlabel('检测误差水平'); ax1.set_ylabel('模拟达标率')
        ax1.set_title('检测误差对达标率影响'); ax1.legend(); ax1.grid(True, alpha=0.3)

        for i,(cluster,res) in enumerate(self.error_analysis_results.items()):
            robust=[res['error_analysis'][lvl]['robust_probability'] for lvl in error_levels]
            ax2.plot(error_levels, robust, 's-', color=colors[i], label=f"聚类 G'{cluster+1}组")
        ax2.axhline(0.9, color='orange', ls='--', alpha=0.7)
        ax2.set_xlabel('检测误差水平'); ax2.set_ylabel('鲁棒性概率')
        ax2.set_title('鲁棒性概率 (达标率≥80%)'); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def cross_validation_stability(self, n_folds=5):
        print("=== 交叉验证稳定性检验开始 ===")
        if not self.factor_weights:
            print("缺少因素权重，先运行 calculate_factor_weights()")
            return
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results=[]
        for fold,(train_idx,test_idx) in enumerate(kf.split(self.processed_data)):
            print(f"处理第 {fold+1}/{n_folds} 折...")
            train = self.processed_data.iloc[train_idx]
            X_train = train[list(self.factor_weights.keys())]
            y_train = train[self.target_variable]
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_train)
            model = LinearRegression().fit(Xs, y_train)
            coefs = np.abs(model.coef_)
            weights = coefs / coefs.sum() if coefs.sum()>0 else np.ones_like(coefs)/len(coefs)
            test = self.processed_data.iloc[test_idx]
            X_test = scaler.transform(test[list(self.factor_weights.keys())])
            composite = np.sum(X_test * weights, axis=1)
            kmeans = KMeans(n_clusters=getattr(self,'optimal_k',3), random_state=42)
            labels = kmeans.fit_predict(composite.reshape(-1,1))
            if len(np.unique(labels))>1:
                sil = silhouette_score(composite.reshape(-1,1), labels)
            else:
                sil = 0
            fold_results.append({
                'fold': fold+1,
                'weights': dict(zip(self.factor_weights.keys(), weights)),
                'silhouette_score': sil
            })
        weight_var={}
        for var in self.factor_weights.keys():
            ws=[fr['weights'][var] for fr in fold_results]
            weight_var[var]={
                'mean': np.mean(ws),
                'std': np.std(ws),
                'cv': np.std(ws)/np.mean(ws) if np.mean(ws)>0 else float('inf')
            }
        sil_scores=[fr['silhouette_score'] for fr in fold_results]
        sil_mean=np.mean(sil_scores); sil_std=np.std(sil_scores)
        print("\n交叉验证稳定性结果:")
        print("="*50)
        for var,st in weight_var.items():
            print(f"  {var}: 均值={st['mean']:.4f}, CV={st['cv']:.4f}")
        print(f"\n聚类轮廓系数: 均值={sil_mean:.4f}, Std={sil_std:.4f}, 范围=[{min(sil_scores):.4f},{max(sil_scores):.4f}]")
        print("=== 交叉验证稳定性检验完成 ===\n")
        return {'weight_variations': weight_var, 'silhouette_mean': sil_mean}

    def generate_comprehensive_report(self):
        print("="*80)
        print("多因素 Y染色体浓度 综合分析报告 (Spearman + BMI肘部区间)")
        print("="*80)
        print(f"\n1. 数据信息: 样本数={len(self.processed_data)}; 窗口=10-25周")

        if self.significant_factors:
            print("\n2. 显著相关因素 (ρ, p, 权重):")
            for v, info in self.significant_factors.items():
                print(f"  {v}: ρ={info['coefficient']:.4f}, p={info['p_value']:.4g}, w={self.factor_weights.get(v,0):.3f}")

        if self.bmi_intervals:
            print("\n3. BMI 肘部法则聚类区间:")
            for i,(a,b) in enumerate(self.bmi_intervals):
                print(f"  聚类区间 {i+1}: [{a:.2f}, {'∞' if np.isinf(b) else f'{b:.2f}'} )")

        if self.optimal_timepoints:
            print("\n4. 综合因子聚类最佳检测时点:")
            for c,r in self.optimal_timepoints.items():
                print(f"  聚类 G'{c+1}': {r['optimal_timepoint']:.2f}周, 达标率 {r['success_rate']:.1%}, 风险值 {r['total_risk']:.3f}")

        if hasattr(self,'correlation_matrix') and self.correlation_matrix is not None:
            print("\n5. 相关矩阵已生成（见热力图文件）。")

        if self.factor_weights:
            dominant = max(self.factor_weights.items(), key=lambda x: x[1])
            print(f"\n6. 主导因素: {dominant[0]} (权重 {dominant[1]:.1%})")

        print("="*80)

    def run_complete_analysis(self):
        print("开始多因素 Y染色体浓度 综合分析...")
        try:
            self.load_and_preprocess_data()
            self.stepwise_regression_analysis()
            self.calculate_factor_weights()
            self.construct_comprehensive_factor()
            # 这里调用增强聚类，自动包含 BMI 肘部法则区间输出
            self.enhanced_clustering(bmi_elbow=True)
            self.construct_comprehensive_scoring_function()
            self.multi_constraint_optimization()
            self.pareto_frontier_analysis()
            self.detection_error_analysis()
            self.cross_validation_stability()
            self.generate_comprehensive_report()
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    optimizer = MultiFactorNIPTOptimizer('data_0.csv')
    optimizer.run_complete_analysis()