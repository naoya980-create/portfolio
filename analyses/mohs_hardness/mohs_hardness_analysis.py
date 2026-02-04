#%% [markdown]
"""
モース硬度データセット（Kaggle: Prediction of Mohs Hardness with Machine Learning）の概要

硬さ（Hardness）とは、材料が永久的または塑性的な変形に抵抗する度合いを定量的に表す値であり、
セラミックコーティングや研磨材など多くの用途における材料設計で非常に重要な役割を果たします。
硬さ試験は非破壊で実施でき、かつ材料の塑性特性を把握するために簡便な方法であるため、特に有用です。

本研究では、自然界に存在する材料を対象に、組成から得られる原子・電子的特徴量を直接用いて
硬さを予測する機械（統計的）学習アプローチが提案されています。
まず、組成からファンデルワールス半径、共有結合半径、価電子数などの原子・電子的特徴量が抽出されています。

この研究では、こうした組成由来の特徴量が、異なる化学空間・結晶構造・結晶クラスをもつ鉱物の
モース硬度を予測するのに利用できるかどうかを検証するため、複数の分類器が学習されています。
分類モデルの学習と評価に用いられたデータセットは、
「Physical and Optical Properties of Minerals CRC Handbook of Chemistry and Physics」および
「American Mineralogist Crystal Structure Database」に報告されている
自然鉱物の実験的なモース硬度データ、結晶クラス、化学組成に由来します。

元のデータベースは 369 種類のユニークな鉱物名から構成されていますが、
同じ鉱物名に対して複数の組成の組み合わせが存在するため、まずそれらに対して組成の順列操作を行いました。
その結果、ユニークな組成をもつ 622 個の鉱物からなるデータベースが得られ、
その内訳は単斜晶 210、菱面体晶 96、六方晶 89、正方晶 80、立方晶 73、
斜方晶 50、三斜晶 22、三方晶 1、アモルファス構造 1 となります。

モデル性能を検証するため、独立した検証用データセットも別途作成されています。
この検証データセットには、文献に報告された 51 個の合成単結晶の組成、結晶構造、
およびモース硬度の値が含まれます。
その内訳は単斜晶 15、正方晶 7、六方晶 7、斜方晶 6、立方晶 4、菱面体晶 3 です。

さらに著者は、上記 CRC Handbook から取得した自然鉱物を対象に、
組成に基づく特徴量記述子のデータベースを構築しています。
この包括的な組成ベースのデータセットにより、
多様な鉱物組成と結晶クラスにわたって硬さを予測できるモデルを学習することが可能となります。
自然鉱物データセットおよび人工単結晶データセットの各材料は、
11 種類の原子記述子で表現されています。
これらの元素特徴には、電子数、価電子数、原子番号、
最も一般的な酸化状態におけるポーリング電気陰性度、共有結合半径、
ファンデルワールス半径、中性原子のイオン化エネルギーが含まれます。
"""

# %% [markdown]
# ## Mohs 硬度データセット — shape・統計量の確認
# Kaggle の "Prediction of Mohs Hardness with Machine Learning" を用い、
# 訓練データの形状・列情報・欠損・基本統計量・目的変数（モース硬度）の分布を確認します。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import json
import inspect
import os
import time

# この変更以降の検証ランID（ログで pre-fix と区別）
RUN_ID = "post-fix"

#region agent log helper
def _agent_log(*, hypothesisId: str, location: str, message: str, data: dict | None = None, runId: str = "pre-fix"):
    """Write one NDJSON line for debug-mode evidence."""
    try:
        payload = {
            "id": f"log_{int(time.time() * 1000)}_{os.getpid()}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data or {},
            "sessionId": "debug-session",
            "runId": runId,
            "hypothesisId": hypothesisId,
        }
        with open(r"c:\Users\naoya\projects\portfolio\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
#endregion

# カレントが analyses/mohs_hardness/ のとき data/ を相対パスで参照
DATA_DIR = Path("data")
CRYSTALS_PATH = DATA_DIR / "Artificial_Crystals_Dataset.csv"
MINERAL_PATH = DATA_DIR / "Mineral_Dataset_Supplementary_Info.csv"

#region agent log H1
try:
    import sklearn  # type: ignore
    _sk_ver = getattr(sklearn, "__version__", None)
except Exception as _e:
    _sk_ver = None
_agent_log(
    hypothesisId="H1",
    location="analyses/mohs_hardness/mohs_hardness_analysis.py:imports",
    message="Environment versions",
    data={
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "sklearn_version": _sk_ver,
        "lightgbm_version": getattr(lgb, "__version__", None),
        "mean_squared_error_module": getattr(mean_squared_error, "__module__", None),
    },
    runId=RUN_ID,
)
#endregion

#region agent log H2
try:
    _mse_sig = str(inspect.signature(mean_squared_error))
except Exception as _e:
    _mse_sig = None
_agent_log(
    hypothesisId="H2",
    location="analyses/mohs_hardness/mohs_hardness_analysis.py:imports",
    message="mean_squared_error signature",
    data={
        "signature": _mse_sig,
        "has_squared_kw": (_mse_sig is not None and "squared" in _mse_sig),
    },
    runId=RUN_ID,
)
#endregion

# %% [markdown]
# ### 1. データの読み込み
# 使用データ: `Artificial_Crystals_Dataset.csv`（人工結晶）、`Mineral_Dataset_Supplementary_Info.csv`（鉱物・補足情報）。

# %%
if not CRYSTALS_PATH.exists():
    raise FileNotFoundError(
        f"{CRYSTALS_PATH} が見つかりません。"
        " data/ に CSV を配置してください。"
    )
df_artificial = pd.read_csv(CRYSTALS_PATH)
df_mineral = pd.read_csv(MINERAL_PATH) if MINERAL_PATH.exists() else None

# CSV 先頭列が行番号（0,1,2,...）になっているため落とす
# （Kaggle の CSV が先頭カンマで始まる形式だと "Unnamed: 0" 列として読み込まれる）
for _df_name, _df in [("df_artificial", df_artificial), ("df_mineral", df_mineral)]:
    if _df is None or _df.empty:
        continue
    first_col = _df.columns[0]
    if isinstance(first_col, str) and first_col.startswith("Unnamed"):
        _df.drop(columns=[first_col], inplace=True)

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
candidates = [c for c in df_artificial.columns if "mohs" in c.lower()]
if not candidates:
    candidates = [c for c in df_artificial.columns if "hardness" in c.lower()]
target_col = candidates[0] if candidates else df_artificial.columns[-1]
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
df_artificial.head()
#%%
df_mineral.head()
# %%
numeric_cols = df_artificial.select_dtypes(include="number").columns
numeric_cols_mineral = df_mineral.select_dtypes(include="number").columns
# %%
# df_artificialの散布図行列をCrystal structureで色分け
if "Crystal structure" in df_artificial.columns:
    if len(numeric_cols) > 1:
        sns.pairplot(df_artificial, vars=numeric_cols, hue="Crystal structure", palette="tab10")
        plt.suptitle("df_artificialの散布図行列（crystal structureによる色分け）", y=1.02)
        plt.show()
    else:
        print("df_artificial: 数値カラムが1つ以下のため、散布図行列は作成できません。")
else:
    print("df_artificialに 'crystal structure' 列がありません。")
#%%
# df_mineralのモース硬度を目的変数としてLightGBMで5分割交差検証による予測解析



# "Hardness" カラムを目的変数とする
target_col = [c for c in df_mineral.columns if "mohs" in c.lower() or "hardness" in c.lower()]
if target_col:
    target_col = target_col[0]
else:
    target_col = df_mineral.columns[-1]
print(f"df_mineral 目的変数として使用する列: {target_col}")

# 数値特徴量を説明変数とする
features = df_mineral.select_dtypes(include="number").drop(columns=[target_col], errors="ignore").columns.tolist()
if not features:
    raise ValueError("数値特徴量が見つかりません。")

X = df_mineral[features]
y = df_mineral[target_col]

# 欠損値補完（平均値で埋める）
X = X.fillna(X.mean())
y = y.fillna(y.mean())

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []
fold = 1

for train_idx, test_idx in kf.split(X):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # LightGBM用データセット
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': 42,
    }

    model = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train, lgb_valid],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10, verbose=False),
        lgb.log_evaluation(period=0),  # 学習ログを出さない
    ],
)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    #region agent log H3
    _agent_log(
        hypothesisId="H3",
        location="analyses/mohs_hardness/mohs_hardness_analysis.py:cv_rmse",
        message="About to compute RMSE via mean_squared_error(squared=False)",
        data={
            "fold": fold,
            "y_test_shape": getattr(getattr(y_test, "shape", None), "__repr__", lambda: None)(),
            "y_pred_len": int(len(y_pred)) if hasattr(y_pred, "__len__") else None,
        },
        runId=RUN_ID,
    )
    #endregion
    try:
        # sklearn>=? では squared が削除されているため、シグネチャで分岐して互換対応する
        if _mse_sig is not None and "squared" in _mse_sig:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            #region agent log H5
            _agent_log(
                hypothesisId="H5",
                location="analyses/mohs_hardness/mohs_hardness_analysis.py:cv_rmse",
                message="RMSE computed via mean_squared_error(squared=False)",
                data={"fold": fold, "rmse": float(rmse)},
                runId=RUN_ID,
            )
            #endregion
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            #region agent log H5
            _agent_log(
                hypothesisId="H5",
                location="analyses/mohs_hardness/mohs_hardness_analysis.py:cv_rmse",
                message="RMSE computed via sqrt(mean_squared_error) fallback (no squared kw)",
                data={"fold": fold, "mse": float(mse), "rmse": float(rmse)},
                runId=RUN_ID,
            )
            #endregion
    except TypeError as e:
        #region agent log H4
        _agent_log(
            hypothesisId="H4",
            location="analyses/mohs_hardness/mohs_hardness_analysis.py:cv_rmse",
            message="TypeError when calling mean_squared_error with squared",
            data={
                "error": repr(e),
                "mean_squared_error_module": getattr(mean_squared_error, "__module__", None),
                "mean_squared_error_name": getattr(mean_squared_error, "__name__", None),
                "signature": _mse_sig,
            },
            runId=RUN_ID,
        )
        #endregion
        raise
    rmses.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.4f}")

    # 最初のfoldだけ特徴量重要度を表示
    if fold == 1:
        importances = model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(features, importances), key=lambda t: t[1], reverse=True)
        print("特徴量重要度:")
        for f, imp in feat_imp:
            print(f"{f}: {imp:.2f}")

    fold += 1

print(f"\n5分割交差検証 RMSE (平均±標準偏差): {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
# %%
import lightgbm as lgb
print(lgb.__version__)
# %%
