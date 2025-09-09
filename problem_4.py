import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from dataclasses import dataclass
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from scipy.stats import chi2

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ------------------ 基础配置 ------------------
DEFAULT_INPUT = "data_1.csv"
DEFAULT_OUTPUT = "female_predictions_data1.csv"

# 扩展权重集合
WEIGHT_CANDIDATES = [
    (0.70, 0.25, 0.05),
    (0.65, 0.30, 0.05),
    (0.60, 0.35, 0.05),
    (0.60, 0.30, 0.10),
    (0.55, 0.35, 0.10),
    (0.55, 0.25, 0.20),
    (0.50, 0.40, 0.10),
    (0.50, 0.30, 0.20),
]

THRESH_LOW_RANGE = np.arange(0.25, 0.61, 0.02) 
THRESH_HIGH_RANGE = np.arange(0.60, 0.85, 0.02)  

MAHAL_FEATURES = [
    "Z13", "Z18", "Z21", "Zx", "gc_dev", "eff_unique", "filter_rate"
]

BASE_FEATURES = [
    "Z13", "Z18", "Z21", "Zx",
    "absZ13", "absZ18", "absZ21",
    "maxAbsZ", "sumAbsZ", "z_var",
    "raw_reads", "map_ratio", "dup_ratio", "unique_mapped",
    "eff_unique", "gc_total", "gc_dev",
    "gc13", "gc18", "gc21", "gc_mean321", "gc_mean_diff",
    "BMI", "gest_week",
    "gravidity_num", "parity_num", "is_multipara",
    "BMI_bin_idx",
    "mahal_dist", "stat_risk",
    "rule_score_adj",
    "cm_IVF", "cm_自然受孕"
]

SEQ_EXTRA_FEATURES = [
    "Z13_max", "Z13_min", "Z13_range", "Z13_last", "Z13_first", "Z13_slope",
    "Z18_max", "Z18_min", "Z18_range", "Z18_last", "Z18_first", "Z18_slope",
    "Z21_max", "Z21_min", "Z21_range", "Z21_last", "Z21_first", "Z21_slope",
]

GC_EXTRA_FEATURES = [
    "gc_out_of_range", "gc_out_mild", "gc_out_severe",
    "gc_zscore", "gc_iqr_flag", "gc_sigma_outlier",
    "gc_penalty"
]

COLMAP = {
    "孕妇代码": "patient_id",
    "年龄": "age",
    "身高": "height",
    "体重": "weight",
    "IVF妊娠": "conception_method",
    "检测日期": "test_date",
    "检测抽血次数": "draw_index",
    "检测孕周": "gest_week_raw",
    "孕妇BMI": "BMI",
    "原始读段数": "raw_reads",
    "在参考基因组上比对的比例": "map_ratio",
    "重复读段的比例": "dup_ratio",
    "唯一比对的读段数": "unique_mapped",
    "GC含量": "gc_total",
    "13号染色体的Z值": "Z13",
    "18号染色体的Z值": "Z18",
    "21号染色体的Z值": "Z21",
    "X染色体的Z值": "Zx",
    "X染色体浓度": "X_conc",
    "13号染色体的GC含量": "gc13",
    "18号染色体的GC含量": "gc18",
    "21号染色体的GC含量": "gc21",
    "被过滤掉读段数的比例": "filter_rate",
    "染色体的非整倍体": "aneu",
    "怀孕次数": "gravidity_raw",
    "生产次数": "parity_raw",
    "胎儿是否健康": "fetal_health"
}

@dataclass
class FoldMetric:
    fold: int
    roc_auc: float
    pr_auc: float
    recall_t50: float
    specificity_t50: float

# ------------------ 工具函数 ------------------
def parse_gest_week(s: str):
    if pd.isna(s): return np.nan
    s = str(s).strip().lower()
    m = re.match(r'(\d+)\s*w\+?(\d*)', s)
    if m:
        return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0)/7
    if re.fullmatch(r'\d+', s):
        return float(s)
    m2 = re.match(r'(\d+)\s*w$', s)
    if m2: return float(m2.group(1))
    return np.nan

def normalize_gravidity(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if "≥" in s:
        d = re.findall(r'\d+', s)
        return float(d[0]) if d else 3.0
    try: return float(s)
    except: return np.nan

def normalize_parity(x): return normalize_gravidity(x)

def bmi_bin(bmi):
    if pd.isna(bmi): return "NA", -1
    if 20 <= bmi < 28: return "[20,28)", 0
    if 28 <= bmi < 32: return "[28,32)", 1
    if 32 <= bmi < 36: return "[32,36)", 2
    if 36 <= bmi < 40: return "[36,40)", 3
    if bmi >= 40: return ">=40", 4
    return "<20", 5

def quality_flag(row, depth_p10, filter_rate_max=0.03, gc_dev_max=0.12):
    return int(
        (row["raw_reads"] < depth_p10) or
        (row["filter_rate"] > filter_rate_max) or
        (abs(row["gc_dev"]) > gc_dev_max)
    )

def rule_score_fn(row):
    if row["maxAbsZ"] >= 4: return 1.0
    cnt = sum([row["absZ13"] >= 3, row["absZ18"] >= 3, row["absZ21"] >= 3])
    if cnt >= 2: return 0.6
    if row.get("quality_flag", 0) == 1: return 0.3
    return 0.0

def mahalanobis(X, mean, inv_cov):
    diff = X - mean
    left = diff @ inv_cov
    return (left * diff).sum(axis=1)

def compute_mahal_ref(df_norm, feats):
    X = df_norm[feats].values
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    return mean, inv_cov

def linear_slope(xs: np.ndarray, ys: np.ndarray):
    if len(xs) < 2: return 0.0
    xm, ym = xs.mean(), ys.mean()
    denom = ((xs - xm)**2).sum()
    if denom == 0: return 0.0
    return ((xs - xm)*(ys - ym)).sum()/denom

def build_sequence_features(df_all: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for pid, g in df_all.groupby("patient_id", sort=False):
        g = g.sort_values("gest_week")
        rec = {"patient_id": pid}
        for zc in ["Z13","Z18","Z21"]:
            arr = g[zc].values
            if len(arr)==0:
                for suf in ["max","min","range","last","first","slope"]:
                    rec[f"{zc}_{suf}"] = np.nan
                continue
            rec[f"{zc}_max"] = np.nanmax(arr)
            rec[f"{zc}_min"] = np.nanmin(arr)
            rec[f"{zc}_range"] = rec[f"{zc}_max"] - rec[f"{zc}_min"]
            rec[f"{zc}_first"] = arr[0]
            rec[f"{zc}_last"] = arr[-1]
            weeks = g["gest_week"].values
            rec[f"{zc}_slope"] = linear_slope(weeks, arr) if len(arr)>=2 else 0.0
        feats.append(rec)
    return pd.DataFrame(feats)

# ------------------ GC 特征增强 ------------------
def build_gc_features(df: pd.DataFrame, gc_col="gc_total"):
    if gc_col not in df:
        df[gc_col] = np.nan
    gc = df[gc_col].astype(float)
    df["gc_out_of_range"] = ((gc < 0.40) | (gc > 0.60)).astype(int)
    df["gc_out_mild"] = (((gc >= 0.38) & (gc < 0.40)) | ((gc > 0.60) & (gc <= 0.62))).astype(int)
    df["gc_out_severe"] = ((gc < 0.38) | (gc > 0.62)).astype(int)
    mu = gc.mean()
    sigma = gc.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        sigma = 1e-6
    df["gc_zscore"] = (gc - mu) / sigma
    q1 = gc.quantile(0.25)
    q3 = gc.quantile(0.70)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    df["gc_iqr_flag"] = ((gc < lower) | (gc > upper)).astype(int)
    df["gc_sigma_outlier"] = (df["gc_zscore"].abs() > 3).astype(int)
    def gc_tier(row):
        if row["gc_out_severe"] == 1:
            return "Severe"
        if row["gc_out_of_range"] == 1 or row["gc_out_mild"] == 1:
            return "Mild"
        return "Normal"
    df["gc_quality_tier"] = df.apply(gc_tier, axis=1)
    # 惩罚分 (Severe=1.0, Mild/Out=0.4)
    df["gc_penalty"] = (
        df["gc_out_severe"]*1.0 +
        ((df["gc_out_of_range"] | df["gc_out_mild"]) * (1 - df["gc_out_severe"]) * 0.4)
    )
    print(f"[GC] mean={mu:.4f} std={sigma:.4f} severe%={df['gc_out_severe'].mean():.3f} "
          f"range_out%={df['gc_out_of_range'].mean():.3f} sigma_out%={df['gc_sigma_outlier'].mean():.3f}")
    return df

# ------------------ 多种阈值搜索 ------------------
def search_thresholds_multi(
    y_true, risk, lows, highs,
    min_recall=0.90, min_specificity=0.35, min_precision=0.35,
    coverage_cap=0.55
):
    import itertools
    y_true = np.array(y_true)
    recs=[]
    for tl, th in itertools.product(lows, highs):
        if th <= tl: continue
        pred_all = (risk >= tl).astype(int)
        coverage = pred_all.mean()
        if coverage_cap is not None and coverage > coverage_cap:
            continue
        pred_hr = (risk >= th).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, pred_all).ravel()
        except ValueError:
            continue
        recall_all = tp/(tp+fn) if (tp+fn) else 0
        spec_all = tn/(tn+fp) if (tn+fp) else 0
        prec_all = tp/(tp+fp) if (tp+fp) else 0
        if (min_recall is not None) and (recall_all < min_recall): continue
        if (min_specificity is not None) and (spec_all < min_specificity): continue
        if (min_precision is not None) and (prec_all < min_precision): continue
        try:
            tn2, fp2, fn2, tp2 = confusion_matrix(y_true, pred_hr).ravel()
        except ValueError:
            tn2=fp2=fn2=tp2=0
        prec_hr = tp2/(tp2+fp2) if (tp2+fp2) else 0
        recall_hr = tp2/(tp2+fn2) if (tp2+fn2) else 0
        J_all = recall_all + spec_all - 1
        recs.append(dict(
            thresh_low=tl, thresh_high=th,
            recall_all=recall_all, specificity_all=spec_all, precision_all=prec_all,
            recall_hr=recall_hr, precision_hr=prec_hr,
            coverage_all=coverage, coverage_hr=pred_hr.mean(),
            J_all=J_all
        ))
    if not recs: return pd.DataFrame()
    df = pd.DataFrame(recs)
    return df.sort_values(["J_all","recall_all"], ascending=[False,False]).reset_index(drop=True)

def summarize_threshold_sets(df, topk=5):
    if df.empty: return {}
    return {
        "high_recall": df.sort_values("recall_all", ascending=False).head(topk),
        "balanced": df.sort_values("J_all", ascending=False).head(topk),
        "high_precision_highrisk": df.sort_values("precision_hr", ascending=False).head(topk)
    }

def search_thresholds_target_band(
    y_true, risk, lows, highs,
    target_recall=0.70, recall_band=0.05,
    min_specificity=0.35, min_precision=0.35,
    optimize="precision",
    coverage_cap=0.55
):
    import itertools
    y_true = np.array(y_true)
    lower = max(0.0, target_recall - recall_band)
    upper = min(1.0, target_recall + recall_band)
    recs=[]
    for tl, th in itertools.product(lows, highs):
        if th <= tl: continue
        pred_all = (risk >= tl).astype(int)
        cov = pred_all.mean()
        if coverage_cap is not None and cov > coverage_cap:
            continue
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, pred_all).ravel()
        except ValueError:
            continue
        recall_all = tp/(tp+fn) if (tp+fn) else 0
        if recall_all < lower or recall_all > upper: continue
        spec_all = tn/(tn+fp) if (tn+fp) else 0
        prec_all = tp/(tp+fp) if (tp+fp) else 0
        if (min_specificity is not None) and (spec_all < min_specificity): continue
        if (min_precision is not None) and (prec_all < min_precision): continue
        J_all = recall_all + spec_all - 1
        f1 = (2*prec_all*recall_all/(prec_all+recall_all)) if (prec_all+recall_all)>0 else 0
        recs.append(dict(
            thresh_low=tl, thresh_high=th,
            recall_all=recall_all, specificity_all=spec_all, precision_all=prec_all,
            J_all=J_all, f1=f1, coverage_all=cov
        ))
    if not recs: return pd.DataFrame()
    df = pd.DataFrame(recs)
    if optimize == "precision":
        df = df.sort_values(["precision_all","J_all"], ascending=[False,False])
    elif optimize == "J":
        df = df.sort_values(["J_all","precision_all"], ascending=[False,False])
    elif optimize == "f1":
        df = df.sort_values(["f1","precision_all"], ascending=[False,False])
    return df.reset_index(drop=True)

# ------------------ 主流程 ------------------
def run_pipeline(
    data_path: Path,
    output_path: Path,
    folds: int = 5,
    min_recall: float = 0.90,
    use_all: bool = False,
    agg_strategy: str = "first",  # first | best_z | last
    engineer_seq: bool = False,
    random_state: int = 42,
    # 新增策略相关参数
    multi_objective: bool = False,
    min_specificity: float = 0.35,
    min_precision: float = 0.35,
    enable_target_band: bool = False,
    target_recall: float = 0.70,
    recall_band: float = 0.05,
    band_optimize: str = "precision",
    coverage_cap: float = 0.55
):
    np.random.seed(random_state)

    print(f"加载数据: {data_path}")
    df_raw = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    rename_dict = {c: COLMAP[c] for c in COLMAP if c in df_raw.columns}
    df_all = df_raw.rename(columns=rename_dict).copy()

    # 标签
    df_all["aneu"] = df_all["aneu"].fillna("").astype(str).str.strip()
    df_all["record_label"] = df_all["aneu"].apply(lambda x: 0 if x == "" else 1)
    patient_label = df_all.groupby("patient_id")["record_label"].max().rename("label_patient")

    # 孕周
    df_all["gest_week"] = df_all["gest_week_raw"].apply(parse_gest_week)

    # Z & 派生
    for zc in ["Z13","Z18","Z21","Zx"]:
        if zc not in df_all: df_all[zc] = np.nan
    df_all["absZ13"] = df_all["Z13"].abs()
    df_all["absZ18"] = df_all["Z18"].abs()
    df_all["absZ21"] = df_all["Z21"].abs()
    df_all["maxAbsZ"] = df_all[["absZ13","absZ18","absZ21"]].max(axis=1)
    df_all["sumAbsZ"] = df_all[["absZ13","absZ18","absZ21"]].sum(axis=1)
    df_all["z_var"] = df_all[["Z13","Z18","Z21"]].var(axis=1)

    df_all["eff_unique"] = df_all["map_ratio"] * (1 - df_all["dup_ratio"])
    df_all["gc_dev"] = (df_all["gc_total"] - 0.5).abs()
    df_all["gc_mean321"] = df_all[["gc13","gc18","gc21"]].mean(axis=1)
    df_all["gc_mean_diff"] = df_all["gc_mean321"] - df_all["gc_total"]

    # BMI
    bmi_bins = df_all["BMI"].apply(lambda x: bmi_bin(x))
    df_all["BMI_bin"] = bmi_bins.apply(lambda x: x[0])
    df_all["BMI_bin_idx"] = bmi_bins.apply(lambda x: x[1])

    # Gravidity / Parity
    df_all["gravidity_num"] = df_all["gravidity_raw"].apply(normalize_gravidity).clip(upper=3)
    df_all["parity_num"] = df_all["parity_raw"].apply(normalize_parity).clip(upper=3)
    df_all["is_multipara"] = (df_all["parity_num"] > 0).astype(int)

    # 受孕方式
    df_all["conception_method"] = df_all["conception_method"].fillna("未知")
    df_all["cm_IVF"] = (df_all["conception_method"].str.contains("IVF（试管婴儿）")).astype(int)
    df_all["cm_自然受孕"] = (df_all["conception_method"] == "自然受孕").astype(int)

    # 质量 flag (记录级)
    depth_p10 = np.percentile(df_all.loc[df_all["record_label"] == 0, "raw_reads"].dropna(), 10) \
        if (df_all["record_label"] == 0).sum() > 5 else df_all["raw_reads"].quantile(0.1)
    df_all["quality_flag"] = df_all.apply(lambda r: quality_flag(r, depth_p10), axis=1)

    df_all["rule_score"] = df_all.apply(rule_score_fn, axis=1)

    # 序列特征
    seq_df = None
    if use_all and engineer_seq:
        print("构建时间序列特征 ...")
        seq_df = build_sequence_features(df_all)

    # 聚合策略
    def pick_index(g: pd.DataFrame):
        if agg_strategy == "first":
            return g.sort_values("gest_week").iloc[0]
        elif agg_strategy == "best_z":
            return g.sort_values("maxAbsZ", ascending=False).iloc[0]
        elif agg_strategy == "last":
            return g.sort_values("gest_week").iloc[-1]
        else:
            raise ValueError("未知聚合策略，应为 first|best_z|last")

    if use_all:
        sel_records=[]
        for pid, g in df_all.groupby("patient_id", sort=False):
            g_valid = g.dropna(subset=["gest_week"])
            if g_valid.empty: g_valid = g
            sel_records.append(pick_index(g_valid))
        df = pd.DataFrame(sel_records).reset_index(drop=True)
    else:
        first_idx = df_all.groupby("patient_id")["gest_week"].idxmin()
        df = df_all.loc[first_idx].copy().reset_index(drop=True)
        agg_strategy = "first"
        print("未指定 --use_all_records，使用首次记录。")

    df = df.merge(patient_label, left_on="patient_id", right_index=True, how="left")

    # GC 增强特征
    df = build_gc_features(df, gc_col="gc_total")

    # 调整质量标志: severe 或 sigma_outlier 也算质量风险
    df["quality_flag"] = df["quality_flag"] | df["gc_out_severe"] | df["gc_sigma_outlier"]

    # 调整 rule_score -> rule_score_adj (惩罚 GC 质量)
    df["rule_score_adj"] = df["rule_score"] - 0.5 * df["gc_penalty"]
    df["rule_score_adj"] = df["rule_score_adj"].clip(lower=0)

    # 序列特征合并
    if seq_df is not None:
        df = df.merge(seq_df, on="patient_id", how="left")
        for c in SEQ_EXTRA_FEATURES:
            if c not in df: df[c] = np.nan
        full_features = BASE_FEATURES + GC_EXTRA_FEATURES + SEQ_EXTRA_FEATURES
    else:
        full_features = BASE_FEATURES + GC_EXTRA_FEATURES

    # Mahalanobis
    norm_ref = df[df["label_patient"] == 0].dropna(subset=MAHAL_FEATURES)
    if len(norm_ref) >= 10:
        mean_vec, inv_cov = compute_mahal_ref(norm_ref, MAHAL_FEATURES)
        X_m = df[MAHAL_FEATURES].fillna(df[MAHAL_FEATURES].median()).values
        df["mahal_dist"] = mahalanobis(X_m, mean_vec, inv_cov)
        p_tail = 1 - chi2.cdf(df["mahal_dist"], df=len(MAHAL_FEATURES))
        df["stat_risk"] = 1 - p_tail  # chi2.cdf(d²)
    else:
        print("正常样本不足(<10)，stat_risk=0.5")
        df["mahal_dist"] = np.nan
        df["stat_risk"] = 0.5

    # 重新 rule_score_adj 若需要（这里已在构造后处理，无需重复）
    for f in full_features:
        if f not in df:
            df[f] = np.nan
    df[full_features] = df[full_features].fillna(df[full_features].median())

    y = df["label_patient"].values
    groups = df["patient_id"].values
    print(f"\n聚合策略: {agg_strategy} | 使用全部记录: {use_all} | 序列特征: {engineer_seq}")
    print(f"孕妇总数: {len(df)}  正例数: {y.sum()}  正例比例: {y.mean():.3f}")

    # 交叉验证
    gkf = GroupKFold(n_splits=folds)
    oof_log = np.zeros(len(df))
    oof_rf = np.zeros(len(df))
    oof_lgb = np.zeros(len(df)) if HAS_LGB else None

    for fold,(tr,va) in enumerate(gkf.split(df,y,groups),1):
        X_tr = df.iloc[tr][full_features].values
        X_va = df.iloc[va][full_features].values
        y_tr = y[tr]; y_va = y[va]

        pipe_log = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                random_state=random_state,
                max_iter=1000
            ))
        ])
        pipe_log.fit(X_tr, y_tr)
        p_log = pipe_log.predict_proba(X_va)[:,1]
        oof_log[va] = p_log

        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_va)[:,1]
        oof_rf[va] = p_rf

        if HAS_LGB:
            lgb_train = lgb.Dataset(X_tr, y_tr)
            lgb_val = lgb.Dataset(X_va, y_va, reference=lgb_train)
            params = {
                "objective":"binary","metric":"auc",
                "learning_rate":0.05,"num_leaves":31,
                "feature_fraction":0.8,"bagging_fraction":0.8,
                "bagging_freq":4,"seed":random_state,
                "verbosity":-1,"is_unbalance":True
            }
            try:
                model_lgb = lgb.train(
                    params,lgb_train, valid_sets=[lgb_val],
                    num_boost_round=1000, early_stopping_rounds=50,
                    verbose_eval=False
                )
            except TypeError:
                model_lgb = lgb.train(
                    params,lgb_train, valid_sets=[lgb_val],
                    num_boost_round=1000,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=200)
                    ]
                )
            best_iter = getattr(model_lgb,"best_iteration",None)
            p_lgb = model_lgb.predict(X_va, num_iteration=best_iter)
            oof_lgb[va] = p_lgb
            p_fold = (p_log + p_rf + p_lgb)/3
        else:
            p_fold = (p_log + p_rf)/2

        roc_auc = roc_auc_score(y_va, p_fold)
        pr_auc = average_precision_score(y_va, p_fold)
        y_tmp = (p_fold >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_va, y_tmp).ravel()
        recall_t50 = tp/(tp+fn) if (tp+fn) else 0
        spec_t50 = tn/(tn+fp) if (tn+fp) else 0
        print(f"[Fold {fold}] ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} "
              f"Recall@0.5={recall_t50:.3f} Spec@0.5={spec_t50:.3f}")

    if HAS_LGB:
        df["p_model"] = (oof_log + oof_rf + oof_lgb)/3
    else:
        df["p_model"] = (oof_log + oof_rf)/2

    # 风险融合 (使用调整后的 rule_score_adj)
    best_auc = -1
    best_weight = None
    for (w1,w2,w3) in WEIGHT_CANDIDATES:
        fused = w1*df["p_model"] + w2*df["stat_risk"] + w3*df["rule_score_adj"]
        auc = roc_auc_score(y, fused)
        if auc > best_auc:
            best_auc = auc
            best_weight = (w1,w2,w3)
    df["risk_fused"] = best_weight[0]*df["p_model"] + best_weight[1]*df["stat_risk"] + best_weight[2]*df["rule_score_adj"]
    pr_auc_total = average_precision_score(y, df["risk_fused"])
    print(f"\n最佳融合权重 {best_weight}, OOF ROC-AUC={best_auc:.4f}, PR-AUC={pr_auc_total:.4f}")

    # 阈值策略：优先级 目标区间 > 多目标 > 单目标回退
    theta_low = theta_high = None
    used_strategy = None

    if enable_target_band:
        print(f"\n执行目标召回区间搜索: 目标={target_recall:.2f} ± {recall_band:.2f}")
        band_df = search_thresholds_target_band(
            y_true=y,
            risk=df["risk_fused"].values,
            lows=THRESH_LOW_RANGE,
            highs=THRESH_HIGH_RANGE,
            target_recall=target_recall,
            recall_band=recall_band,
            min_specificity=min_specificity,
            min_precision=min_precision,
            optimize=band_optimize,
            coverage_cap=coverage_cap
        )
        if band_df.empty:
            print("目标区间内未找到满足约束的组合，继续尝试其他策略。")
        else:
            print("目标区间候选前10：")
            print(band_df.head(10))
            best_row = band_df.iloc[0]
            theta_low = best_row["thresh_low"]
            theta_high = best_row["thresh_high"]
            print(f"\n采用目标区间阈值 θ_low={theta_low:.2f} θ_high={theta_high:.2f} "
                  f"(Recall={best_row.recall_all:.3f} Spec={best_row.specificity_all:.3f} "
                  f"Precision={best_row.precision_all:.3f} J={best_row.J_all:.3f})")
            used_strategy = "target_band"

    if theta_low is None and multi_objective:
        print("\n执行多目标阈值搜索 ...")
        multi_df = search_thresholds_multi(
            y_true=y,
            risk=df["risk_fused"].values,
            lows=THRESH_LOW_RANGE,
            highs=THRESH_HIGH_RANGE,
            min_recall=min_recall,
            min_specificity=min_specificity,
            min_precision=min_precision,
            coverage_cap=coverage_cap
        )
        if multi_df.empty:
            print("多目标未找到组合，继续回退单目标。")
        else:
            summary = summarize_threshold_sets(multi_df, topk=5)
            print("\n=== 多目标推荐集合 (Top5) ===")
            for k, sub in summary.items():
                print(f"\n>>> {k}")
                print(sub[['thresh_low','thresh_high','recall_all','specificity_all',
                           'precision_all','precision_hr','coverage_all','coverage_hr','J_all']])
            row_sel = summary["balanced"].iloc[0]
            theta_low = row_sel.thresh_low
            theta_high = row_sel.thresh_high
            print(f"\n采用多目标 Balanced 阈值 θ_low={theta_low:.2f} θ_high={theta_high:.2f} "
                  f"(Recall={row_sel.recall_all:.3f} Spec={row_sel.specificity_all:.3f} "
                  f"Precision={row_sel.precision_all:.3f} J={row_sel.J_all:.3f})")
            used_strategy = "multi_objective"

    if theta_low is None:
        print("\n进入单目标回退模式 (递减 min_recall)")
        search_min_recall = min_recall
        chosen = False
        lambda_cov = 0.8  # 覆盖率惩罚系数，可调 (0.5~1.0)
        best_row_global = None

        while search_min_recall >= 0.60 and not chosen:
            recs = []
            for tl in THRESH_LOW_RANGE:
                for th in THRESH_HIGH_RANGE:
                    if th <= tl:
                        continue
                    preds_all = (df["risk_fused"].values >= tl).astype(int)
                    coverage = preds_all.mean()
                    # 覆盖率硬约束
                    if coverage_cap is not None and coverage > coverage_cap:
                        continue
                    try:
                        tn, fp, fn, tp = confusion_matrix(y, preds_all).ravel()
                    except ValueError:
                        continue
                    recall_all = tp / (tp + fn) if (tp + fn) else 0
                    if recall_all < search_min_recall:
                        continue
                    spec_all = tn / (tn + fp) if (tn + fp) else 0
                    prec_all = tp / (tp + fp) if (tp + fp) else 0

                    # 软/硬约束
                    if (min_specificity is not None) and (spec_all < min_specificity):
                        continue
                    if (min_precision is not None) and (prec_all < min_precision):
                        continue

                    J = recall_all + spec_all - 1
                    # 覆盖率惩罚 (即使未超过 coverage_cap 也略微惩罚大的覆盖率)
                    cov_penalty = 0.0
                    if coverage_cap is not None:
                        cov_penalty = lambda_cov * max(0.0, coverage - coverage_cap)
                    J_penalized = J - cov_penalty

                    recs.append(dict(
                        tl=tl, th=th,
                        recall=recall_all,
                        specificity=spec_all,
                        precision=prec_all,
                        coverage=coverage,
                        J=J,
                        J_penalized=J_penalized
                    ))
            if recs:
                rdf = (pd.DataFrame(recs)
                    .sort_values(["J_penalized", "recall", "specificity"],
                                    ascending=[False, False, False])
                    .reset_index(drop=True))
                row = rdf.iloc[0]
                theta_low, theta_high = row.tl, row.th
                print(f"\n单目标阈值 (min_recall尝试={search_min_recall:.2f}) "
                    f"θ_low={theta_low:.2f} θ_high={theta_high:.2f} "
                    f"(Recall={row['recall']:.3f} Spec={row['specificity']:.3f} "
                    f"Precision={row['precision']:.3f} Coverage={row['coverage']:.3f} "
                    f"J={row['J']:.3f} J_penalized={row['J_penalized']:.3f})")
                best_row_global = row
                chosen = True
                used_strategy = "single_fallback"
            else:
                search_min_recall -= 0.05

        if not chosen:
            print("单目标回退未找到满足约束的组合，使用默认 θ_low=0.35 θ_high=0.65")
            theta_low, theta_high = 0.35, 0.65
            used_strategy = "default"

    # 分类
    def final_category(r):
        if r["risk_fused"] >= theta_high:
            return "HighRisk"
        elif r["risk_fused"] >= theta_low:
            return "Retest"
        return "Normal"
    df["final_category"] = df.apply(final_category, axis=1)

    # 性能
    y_pred = (df["risk_fused"] >= theta_low).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    recall = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    precision = tp/(tp+fp) if (tp+fp) else 0
    fpr = 1 - spec
    coverage_all = y_pred.mean()

    print(f"\n最终策略: {used_strategy}")
    print("整体性能 (Retest+HighRisk=阳性):")
    bal_acc = 0.5 * (recall + spec)
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0
    print(f"BalancedAcc={bal_acc:.3f} F1={f1:.3f}")
    print(f"Recall={recall:.3f} Specificity={spec:.3f} Precision={precision:.3f} "
          f"FPR={fpr:.3f} Coverage={coverage_all:.3f}")

    for cat in ["HighRisk","Retest","Normal"]:
        sub = df[df["final_category"]==cat]
        if len(sub)==0: continue
        cat_ppv = sub["label_patient"].mean()
        print(f"{cat}: 数={len(sub)} 阳性率={cat_ppv:.3f}")

    # 输出
    out_cols = [
        "patient_id","gest_week","aneu","label_patient",
        "Z13","Z18","Z21","Zx","absZ13","absZ18","absZ21","maxAbsZ",
        "mahal_dist","stat_risk","rule_score","rule_score_adj",
        "gc_total","gc_penalty","gc_out_of_range","gc_out_mild","gc_out_severe",
        "gc_zscore","gc_iqr_flag","gc_sigma_outlier","gc_quality_tier",
        "p_model","risk_fused","final_category"
    ]
    if engineer_seq:
        out_cols += [c for c in SEQ_EXTRA_FEATURES if c in df.columns]
    for c in out_cols:
        if c not in df: df[c] = np.nan
    df[out_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {output_path}")
    print("分类分布：")
    print(df["final_category"].value_counts())

    return dict(
        weights=best_weight,
        theta_low=theta_low,
        theta_high=theta_high,
        strategy=used_strategy,
        roc_auc=best_auc,
        pr_auc=pr_auc_total,
        recall=recall,
        precision=precision,
        specificity=spec,
        coverage=coverage_all,
        balanced_accuracy=bal_acc,
        f1=f1,
    )

# ------------------ 辅助函数 ------------------
def auto_find_default():
    here = Path(__file__).resolve().parent
    target = here / DEFAULT_INPUT
    if target.exists(): return target
    csvs = list(here.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("未找到 data_1.csv 且目录无其它 csv。")
    if len(csvs) == 1:
        print(f"使用唯一 CSV: {csvs[0].name}")
        return csvs[0]
    print("可用 CSV：")
    for c in csvs: print(" -", c.name)
    raise FileNotFoundError("请提供 data_1.csv 或使用 --data 指定。")

def parse_args():
    p = argparse.ArgumentParser(description="女胎 13/18/21 非整倍体风险判定 - 综合增强版")
    p.add_argument("--data", type=str, default=None, help="CSV 数据文件路径 (默认自动寻找)")
    p.add_argument("--out", type=str, default=DEFAULT_OUTPUT, help="输出文件名")
    p.add_argument("--folds", type=int, default=5, help="GroupKFold 折数")
    p.add_argument("--min_recall", type=float, default=0.90, help="单/多目标模式下的基础最小召回")
    p.add_argument("--use_all_records", action="store_true", help="使用所有检测记录 (用于 best_z/last/序列特征)")
    p.add_argument("--agg_strategy", type=str, default="first", choices=["first","best_z","last"])
    p.add_argument("--engineer_seq", action="store_true", help="构建时间序列特征 (需配合 --use_all_records)")
    p.add_argument("--seed", type=int, default=42)

    # 多目标
    p.add_argument("--multi_objective", action="store_true", help="启用多目标阈值搜索")
    p.add_argument("--min_specificity", type=float, default=0.35, help="特异度下限")
    p.add_argument("--min_precision", type=float, default=0.35, help="精准率下限")

    # 目标召回区间
    p.add_argument("--enable_target_band", action="store_true", help="启用目标召回区间模式")
    p.add_argument("--target_recall", type=float, default=0.70, help="目标召回中心值")
    p.add_argument("--recall_band", type=float, default=0.05, help="召回允许偏差 (±band)")
    p.add_argument("--band_optimize", type=str, default="precision", choices=["precision","J","f1"], help="目标区间优化依据")

    # 覆盖率限制
    p.add_argument("--coverage_cap", type=float, default=0.55, help="Retest+HighRisk 最大占比(0~1)")

    return p.parse_args()

# ------------------ 入口 ------------------
if __name__ == "__main__":
    args = parse_args()
    if args.data is None:
        try:
            data_file = auto_find_default()
        except FileNotFoundError as e:
            print(str(e)); exit(1)
    else:
        data_file = Path(args.data)
        if not data_file.exists():
            print(f"指定文件不存在: {data_file}")
            exit(1)

    if args.engineer_seq and (not args.use_all_records):
        print("提示: 自动开启 --use_all_records 以支持序列特征")
        args.use_all_records = True

    results = run_pipeline(
        data_path=data_file,
        output_path=Path(args.out),
        folds=args.folds,
        min_recall=args.min_recall,
        use_all=args.use_all_records,
        agg_strategy=args.agg_strategy,
        engineer_seq=args.engineer_seq,
        random_state=args.seed,
        multi_objective=args.multi_objective,
        min_specificity=args.min_specificity,
        min_precision=args.min_precision,
        enable_target_band=args.enable_target_band,
        target_recall=args.target_recall,
        recall_band=args.recall_band,
        band_optimize=args.band_optimize,
        coverage_cap=args.coverage_cap
    )

    print("\n运行摘要：")
    for k,v in results.items():
        print(f"  {k}: {v}")