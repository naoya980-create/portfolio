#%% [markdown]
# ## データセット概要（Kaggle: Prediction of Mohs Hardness with Machine Learning）
#
# 組成から抽出した原子・電子的特徴量（ファンデルワールス半径、共有結合半径、価電子数など 11 種）を用い、
# 鉱物のモース硬度を予測する機械学習モデルを学習する。データは自然鉱物 622 件（結晶系別）と
# 検証用の合成単結晶 51 件。CRC Handbook および AMCSD の実験値に基づく。

# %% [markdown]
# ### shape・統計量の確認
# 訓練データの形状・列情報・欠損・基本統計量・目的変数（モース硬度）の分布を確認する。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 日本語フォント設定（文字化け防止）
plt.rcParams["font.family"] = ["MS Gothic", "Meiryo", "sans-serif"]
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# --- 定数 ---
DATA_DIR = Path("data")
CRYSTALS_PATH = DATA_DIR / "Artificial_Crystals_Dataset.csv"
MINERAL_PATH = DATA_DIR / "Mineral_Dataset_Supplementary_Info.csv"


def drop_unnamed_index_column(df: pd.DataFrame) -> None:
    """CSV 先頭の Unnamed: 0 列（行番号）を削除する。"""
    if df is None or df.empty:
        return
    first_col = df.columns[0]
    if isinstance(first_col, str) and first_col.startswith("Unnamed"):
        df.drop(columns=[first_col], inplace=True)


def get_target_column(df: pd.DataFrame, default_last: bool = True) -> str:
    """目的変数列（Mohs/Hardness）を取得する。"""
    candidates = [c for c in df.columns if "mohs" in c.lower()]
    if not candidates:
        candidates = [c for c in df.columns if "hardness" in c.lower()]
    if candidates:
        return candidates[0]
    return df.columns[-1] if default_last else ""


def load_data():
    """CSV を読み込み、Unnamed 列を削除して返す。"""
    if not CRYSTALS_PATH.exists():
        raise FileNotFoundError(
            f"{CRYSTALS_PATH} が見つかりません。"
            " data/ に CSV を配置してください。"
        )
    df_artificial = pd.read_csv(CRYSTALS_PATH)
    df_mineral = pd.read_csv(MINERAL_PATH) if MINERAL_PATH.exists() else None
    drop_unnamed_index_column(df_artificial)
    drop_unnamed_index_column(df_mineral)
    return df_artificial, df_mineral

# %% [markdown]
# ### 1. データの読み込み
# 使用データ: `Artificial_Crystals_Dataset.csv`（人工結晶）、`Mineral_Dataset_Supplementary_Info.csv`（鉱物・補足情報）。

# %%
df_artificial, df_mineral = load_data()
print("Artificial_Crystals_Dataset 読み込み完了.")
if df_mineral is not None:
    print("Mineral_Dataset_Supplementary_Info 読み込み完了.")
else:
    print("Mineral_Dataset_Supplementary_Info は存在しません（省略可）. ")

# %% [markdown]
# ### 2. Shape（行数・列数）

# %%
print("【Artificial_Crystals_Dataset】")
print("df_artificial.shape:", df_artificial.shape)
print("行数 len(df_artificial):", len(df_artificial))
print("列数 len(df_artificial.columns):", len(df_artificial.columns))
if df_mineral is not None:
    print("\n【Mineral_Dataset_Supplementary_Info】")
    print("df_mineral.shape:", df_mineral.shape)

# %% [markdown]
# ### 3. 列情報（列名・型）

# %%
print("列名 df_artificial.columns.tolist():")
print(df_artificial.columns.tolist())
print("\ndf_artificial.dtypes:")
print(df_artificial.dtypes)

# %% [markdown]
# ### 3.1. 列名の意味と日本語訳の対応
#
# #### 基本情報列
#
# | 列名 | 日本語訳 | 意味 |
# |------|----------|------|
# | `Formula` | 化学式 | 材料の化学組成式（例: MnTeMoO6, MgH2） |
# | `Crystal structure` | 結晶構造 | 結晶系（tetragonal: 正方晶、monoclinic: 単斜晶、cubic: 立方晶 など） |
# | `Hardness (Mohs)` / `Hardness` | モース硬度 | 材料の硬さを表す値（1〜10のスケール） |
#
# #### 原子・電子的特徴量（Total = 合計値、Average = 平均値）
#
# | 列名 | 日本語訳 | 意味 |
# |------|----------|------|
# | `allelectrons_Total` | 全電子数（合計） | 組成中の全原子の電子数の合計 |
# | `allelectrons_Average` | 全電子数（平均） | 組成中の原子1個あたりの平均電子数 |
# | `val_e_Average` | 価電子数（平均） | 原子1個あたりの平均価電子数（最外殻電子数） |
# | `atomicweight_Average` | 原子量（平均） | 原子1個あたりの平均原子量 |
# | `ionenergy_Average` | イオン化エネルギー（平均） | 中性原子から電子を1個取り除くのに必要なエネルギー（平均） |
# | `el_neg_chi_Average` | 電気陰性度（平均） | ポーリング電気陰性度の平均値（電子を引き寄せる強さ） |
# | `R_vdw_element_Average` | ファンデルワールス半径（平均） | 原子間の弱い相互作用を考慮した半径の平均 |
# | `R_cov_element_Average` | 共有結合半径（平均） | 共有結合における原子半径の平均 |
# | `zaratio_Average` | Z/A比（平均） | 原子番号（Z）と原子量（A）の比の平均 |
# | `density_Total` | 密度（合計） | 組成全体の密度関連値の合計 |
# | `density_Average` | 密度（平均） | 原子1個あたりの平均密度関連値 |
#
# **注意**: `Total` は組成全体の合計値、`Average` は原子1個あたりの平均値を表します。
# これらは組成から計算された原子・電子的特徴量で、モース硬度の予測に使用されます。

# %% [markdown]
# ### 4. 欠損値

# %%
missing = df_artificial.isna().sum()
print("列ごとの欠損数 (df_artificial.isna().sum()):")
print(missing[missing > 0] if missing.any() else "欠損なし")
if not missing.any():
    print("(全列 0 件)")

# %% [markdown]
# ### 5. 基本統計量（describe）

# %%
print("df_artificial.describe():")
print(df_artificial.describe().to_string())

# %% [markdown]
# ### 6. 目的変数（Mohs hardness）の確認
# Artificial_Crystals は "Hardness (Mohs)"、Mineral は "Hardness" を目的変数として使用します。

# %%
target_col = get_target_column(df_artificial)
print("目的変数として使用する列:", repr(target_col))

# %%
print("目的変数 describe():")
print(df_artificial[target_col].describe())
print("\n目的変数 value_counts() (件数順):")
print(df_artificial[target_col].value_counts().sort_index())

# %%
fig, ax = plt.subplots(figsize=(8, 4))
df_artificial[target_col].hist(bins=min(30, df_artificial[target_col].nunique()), ax=ax, edgecolor="black", alpha=0.8)
ax.set_xlabel(target_col)
ax.set_ylabel("頻度")
ax.set_title(f"目的変数の分布: {target_col}")
plt.tight_layout()
plt.show()
#%%
print(df_artificial.head())
#%%
df_mineral.head() if df_mineral is not None else None
# %%
numeric_cols = df_artificial.select_dtypes(include="number").columns

# 列名と日本語訳の対応（3.1の表に基づく）
COL_LABELS_JA = {
    "allelectrons_Total": "全電子数（合計）",
    "allelectrons_Average": "全電子数（平均）",
    "val_e_Average": "価電子数（平均）",
    "atomicweight_Average": "原子量（平均）",
    "ionenergy_Average": "イオン化エネルギー（平均）",
    "el_neg_chi_Average": "電気陰性度（平均）",
    "R_vdw_element_Average": "ファンデルワールス半径（平均）",
    "R_cov_element_Average": "共有結合半径（平均）",
    "zaratio_Average": "Z/A比（平均）",
    "density_Total": "密度（合計）",
    "density_Average": "密度（平均）",
    "Hardness (Mohs)": "モース硬度",
    "Hardness": "モース硬度",
}


def get_columns_to_drop_from_high_correlation(
    df: pd.DataFrame, numeric_columns: pd.Index, threshold: float = 0.9
) -> list[str]:
    """相関係数が threshold 以上のペアについて、各ペアの一方を削除対象として返す。"""
    df_num = df[numeric_columns].select_dtypes(include="number")
    corr = df_num.corr()
    to_drop = []
    seen = set()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) >= threshold:
                col_i, col_j = corr.columns[i], corr.columns[j]
                # ペアの後ろの列（col_j）を削除対象とする
                if col_j not in seen:
                    to_drop.append(col_j)
                    seen.add(col_j)
    return to_drop


# %%
# 相関係数 0.9 以上のペアを検出（df_mineral の数値列で相関を計算）
# df_mineral がない場合は df_artificial を使用
df_for_corr = df_mineral if df_mineral is not None else df_artificial
target_col_corr = get_target_column(df_for_corr)
numeric_cols_for_corr = (
    df_for_corr.select_dtypes(include="number")
    .drop(columns=[target_col_corr], errors="ignore")
    .columns
)
cols_to_drop = get_columns_to_drop_from_high_correlation(
    df_for_corr, numeric_cols_for_corr, threshold=0.9
)

if cols_to_drop:
    print("相関係数 0.9 以上だったため削除する変数:", cols_to_drop)
    numeric_cols_reduced = [c for c in numeric_cols if c not in cols_to_drop]
else:
    print("相関係数 0.9 以上のペアはありませんでした。")
    numeric_cols_reduced = list(numeric_cols)

# 相関行列の表示（削除前）
corr_matrix = df_for_corr[numeric_cols_for_corr].corr()
high_corr_pairs = [
    (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
    for i in range(len(corr_matrix))
    for j in range(i + 1, len(corr_matrix))
    if abs(corr_matrix.iloc[i, j]) >= 0.9
]
if high_corr_pairs:
    print("相関係数 0.9 以上のペア:")
    for c1, c2, r in high_corr_pairs:
        print(f"  {c1} vs {c2}: r = {r:.4f}")

# %%
# df_artificialの散布図行列（相関0.9以上の片方を削除した変数で作成）
if "Crystal structure" in df_artificial.columns:
    vars_for_pairplot = [c for c in numeric_cols_reduced if c in df_artificial.columns]
    if len(vars_for_pairplot) > 1:
        g = sns.pairplot(
            df_artificial, vars=vars_for_pairplot, hue="Crystal structure", palette="tab10"
        )
        n = len(vars_for_pairplot)
        for i in range(n):
            for j in range(n):
                ax = g.axes[i, j]
                xl = ax.get_xlabel()
                yl = ax.get_ylabel()
                ax.set_xlabel(COL_LABELS_JA.get(xl, xl), fontsize=9, rotation=45, ha="right")
                ax.set_ylabel(COL_LABELS_JA.get(yl, yl), fontsize=9)
        plt.suptitle(
            "df_artificialの散布図行列（相関0.9以上の変数を削除、結晶構造による色分け）", y=1.02
        )
        plt.show()
    else:
        print("df_artificial: 数値カラムが1つ以下のため、散布図行列は作成できません。")
else:
    print("df_artificialに 'crystal structure' 列がありません。")
#%% [markdown]
# ### 7. LightGBM による 5 分割交差検証（削除前後の比較）
# df_mineral のモース硬度を目的変数として予測モデルを学習・評価する。
# 相関 0.9 以上の変数削除前後で LightGBM の性能を比較する。

# %%
def run_lightgbm_cv(X: pd.DataFrame, y: pd.Series, feature_names: list[str], label: str) -> list[float]:
    """LightGBM 5 分割交差検証を実行し、各 Fold の RMSE を返す。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
        params = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42}
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_train, lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        rmse = float(root_mean_squared_error(y_test, y_pred))
        rmses.append(rmse)
        if fold == 1:
            importances = model.feature_importance(importance_type="gain")
            feat_imp = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)
            print(f"  [{label}] 特徴量重要度:")
            for f, imp in feat_imp:
                print(f"    {f}: {imp:.2f}")
        fold += 1
    return rmses


if df_mineral is None:
    raise FileNotFoundError("Mineral_Dataset_Supplementary_Info.csv が必要です。LightGBM 解析をスキップする場合はこのセルを省略してください。")

target_col = get_target_column(df_mineral)
print(f"df_mineral 目的変数として使用する列: {target_col}")

# 数値特徴量（全変数）
features_all = (
    df_mineral.select_dtypes(include="number")
    .drop(columns=[target_col], errors="ignore")
    .columns.tolist()
)
if not features_all:
    raise ValueError("数値特徴量が見つかりません。")

# 相関 0.9 以上で削除した後の特徴量
features_reduced = [f for f in features_all if f not in cols_to_drop]

X_all = df_mineral[features_all].fillna(df_mineral[features_all].mean())
X_reduced = df_mineral[features_reduced].fillna(df_mineral[features_reduced].mean())
y = df_mineral[target_col].fillna(df_mineral[target_col].mean())

# --- 削除前（全変数）---
print("\n【削除前】全変数での LightGBM 5 分割交差検証")
rmses_all = run_lightgbm_cv(X_all, y, features_all, "削除前")
print(f"RMSE (平均±標準偏差): {np.mean(rmses_all):.4f} ± {np.std(rmses_all):.4f}")

# --- 削除後（相関0.9以上の片方を削除）---
print("\n【削除後】相関 0.9 以上の変数を削除した LightGBM 5 分割交差検証")
rmses_reduced = run_lightgbm_cv(X_reduced, y, features_reduced, "削除後")
print(f"RMSE (平均±標準偏差): {np.mean(rmses_reduced):.4f} ± {np.std(rmses_reduced):.4f}")

# --- 比較 ---
print("\n【比較】削除前 vs 削除後")
print(f"  削除前: {np.mean(rmses_all):.4f} ± {np.std(rmses_all):.4f}")
print(f"  削除後: {np.mean(rmses_reduced):.4f} ± {np.std(rmses_reduced):.4f}")
diff = np.mean(rmses_reduced) - np.mean(rmses_all)
print(f"  差 (削除後 - 削除前): {diff:+.4f} {'(悪化)' if diff > 0 else '(改善)' if diff < 0 else '(同程度)'}")

#%% [markdown]
# ### 8. Lasso回帰による変数選択
# 削除後の変数（相関0.9以上の片方を除いた変数）を用いて Lasso 回帰を行い、
# 係数が 0 になる変数（削除候補）を探す。

# %%
X_lasso = X_reduced.copy()
scaler = StandardScaler()
X_lasso_scaled = scaler.fit_transform(X_lasso)

lasso_cv = LassoCV(cv=5, random_state=42).fit(X_lasso_scaled, y)
print(f"LassoCV 最適 alpha: {lasso_cv.alpha_:.6f}")

print("\n係数一覧:")
coef_df = pd.DataFrame(
    {"変数": features_reduced, "係数": lasso_cv.coef_}
).sort_values("係数", key=abs, ascending=False)
print(coef_df.to_string(index=False))

zeros = [f for f, c in zip(features_reduced, lasso_cv.coef_) if abs(c) < 1e-6]
if zeros:
    print(f"\n係数が 0 の変数（削除候補）: {zeros}")
else:
    print("\n係数が 0 の変数はありません。")
    # 係数が非常に小さい変数も表示
    small = [(f, c) for f, c in zip(features_reduced, lasso_cv.coef_) if abs(c) < 0.01 and abs(c) >= 1e-6]
    if small:
        print("係数が 0.01 未満の変数（削除を検討）:", [f for f, _ in small])

#%% [markdown]
# ### 9. density_Total 削除前後の LightGBM 性能比較
# Lasso で係数がほぼ 0 だった density_Total を削除し、RMSE が同程度か確認する。

# %%
features_final = [f for f in features_reduced if f != "density_Total"]
X_final = df_mineral[features_final].fillna(df_mineral[features_final].mean())

print("【7変数】相関削除後の変数（density_Total 含む）")
rmses_7 = run_lightgbm_cv(X_reduced, y, features_reduced, "7変数")
print(f"RMSE (平均±標準偏差): {np.mean(rmses_7):.4f} ± {np.std(rmses_7):.4f}")

print("\n【6変数】density_Total を削除")
rmses_6 = run_lightgbm_cv(X_final, y, features_final, "6変数")
print(f"RMSE (平均±標準偏差): {np.mean(rmses_6):.4f} ± {np.std(rmses_6):.4f}")

print("\n【比較】7変数 vs 6変数（density_Total 削除後）")
print(f"  7変数: {np.mean(rmses_7):.4f} ± {np.std(rmses_7):.4f}")
print(f"  6変数: {np.mean(rmses_6):.4f} ± {np.std(rmses_6):.4f}")
diff_density = np.mean(rmses_6) - np.mean(rmses_7)
print(f"  差 (6変数 - 7変数): {diff_density:+.4f} {'(悪化)' if diff_density > 0 else '(改善)' if diff_density < 0 else '(同程度)'}")

#%% [markdown]
# ### 10. 6変数 LightGBM のバリデーション予測 vs 実測グラフ

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_true_list, y_pred_list = [], []
for train_idx, test_idx in kf.split(X_final):
    X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42}
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_true_list.extend(y_test.values)
    y_pred_list.extend(y_pred)

y_true_arr = np.array(y_true_list)
y_pred_arr = np.array(y_pred_list)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true_arr, y_pred_arr, alpha=0.7, edgecolors="black", linewidths=0.5)
min_val = min(y_true_arr.min(), y_pred_arr.min())
max_val = max(y_true_arr.max(), y_pred_arr.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="予測=実測")
ax.set_xlabel("実測値（モース硬度）")
ax.set_ylabel("予測値（モース硬度）")
ax.set_title("6変数 LightGBM: バリデーションデータの予測 vs 実測")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
# %%
