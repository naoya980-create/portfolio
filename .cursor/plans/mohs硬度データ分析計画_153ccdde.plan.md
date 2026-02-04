---
name: Mohs硬度データ分析計画
overview: Kaggle の Mohs hardness データセットを取得し、既存の MNIST 分析と同様の構成で「shape と各種統計量の確認」を行う分析用スクリプト・ノートブックを追加する計画です。
todos: []
isProject: false
---

# Mohs 硬度データセットの shape・統計量確認計画

## 対象データセット

- **Kaggle**: [prediction-of-mohs-hardness-with-machine-learning](https://www.kaggle.com/datasets/jocelyndumlao/prediction-of-mohs-hardness-with-machine-learning)（jocelyndumlao）
- **内容**: 鉱物のモース硬度（1〜10）を回帰予測するためのデータ。化学組成・結晶構造などに基づく特徴量と目的変数が含まれる。
- **形式**: 典型的に `train.csv` / `test.csv`（および `sample_submission.csv`）が提供される想定。

## 前提・方針

- 既存の [analyses/mnist](analyses/mnist) と同様に、**analyses 配下に専用フォルダ**（例: `analyses/mohs_hardness`）を用意する。
- 分析コードは **Jupytext 連携の .py** とし、`build.bat` で .ipynb → .html まで一括ビルドする構成を踏襲する。
- 今回のスコープは **データ取得・読み込み・shape 確認・各種統計量の確認** までとする。

---

## 1. ディレクトリと依存関係の準備

- **作成**: `analyses/mohs_hardness/` フォルダ。
- **作成**: `analyses/mohs_hardness/requirements.txt`  
  - 必須: `pandas`, `numpy`  
  - 統計・可視化用: `matplotlib`, `seaborn`（既存 mnist と揃える）  
  - データ取得: `kaggle`（Kaggle API 用。未導入ならここで追加）
- データファイルの置き場所: `analyses/mohs_hardness/data/` を想定（Kaggle から取得した CSV をここに配置、または API でここにダウンロード）。

---

## 2. データの取得方法

- **推奨**: Kaggle API（`kaggle` パッケージ）でダウンロード  
  - コマンド例: `kaggle datasets download -d jocelyndumlao/prediction-of-mohs-hardness-with-machine-learning -p analyses/mohs_hardness/data --unzip`  
  - 初回は `~/.kaggle/kaggle.json` に API キー設定が必要。
- **代替**: ブラウザで手動ダウンロードし、ZIP を解凍して `data/` に配置。  
分析スクリプト側では「`data/` 配下の `train.csv`（および `test.csv`）を読む」とパスを固定する。

---

## 3. 分析スクリプトで実施する内容（shape・統計量）

以下の順で実装する。


| 項目        | 内容                                                                         |
| --------- | -------------------------------------------------------------------------- |
| **読み込み**  | `pandas.read_csv()` で `train.csv`（と必要なら `test.csv`）を読み、変数 `df_train` 等に格納。 |
| **Shape** | `df.shape` で (行数, 列数) を表示。`len(df)`, `len(df.columns)` も必要に応じて表示。          |
| **列情報**   | `df.dtypes`, `df.columns.tolist()` で列名・型を確認。                               |
| **欠損**    | `df.isna().sum()` または `df.isnull().sum()` で列ごとの欠損数。                        |
| **基本統計量** | `df.describe()`（数値列の count, mean, std, min, 25%, 50%, 75%, max）。           |
| **目的変数**  | 目的変数（Mohs hardness 列）の `value_counts()` や `describe()`、必要ならヒストグラムで分布確認。    |
| **メモ**    | データセット説明に「化学組成・結晶クラス等」とあるため、列名は実際の CSV で確認し、必要ならコメントでメモする。                 |


---

## 4. ファイル構成（案）

```
analyses/mohs_hardness/
├── data/                          # Kaggle から取得した CSV を配置
│   ├── train.csv
│   └── test.csv
├── mohs_hardness_analysis.py      # Jupytext 対応 .py（セルは # %% で区切り）
├── requirements.txt
└── build.bat                      # jupytext → ipynb → nbconvert → html
```

- **ビルド**: [analyses/mnist/build.bat](analyses/mnist/build.bat) を参考に、`mohs_hardness_analysis.py` → `mohs_hardness_analysis.ipynb` → `../mohs_hardness.html` のように出力する `build.bat` を用意する。

---

## 5. 実装時の注意点

- **列名**: Kaggle の実 CSV で先頭行を確認し、目的変数（例: `mohs_hardness` や `Mohs hardness` 等）を特定してから統計量・可視化の対象にする。
- **パス**: スクリプトは `analyses/mohs_hardness/` をカレントにしたときの相対パス（例: `data/train.csv`）で読むと、ビルドや Jupyter のカレントディレクトリと揃えやすい。
- **実行順**: データ取得（手動 or API）→ `pip install -r requirements.txt` → 分析 .py 実行（またはノートブックで実行）→ `build.bat` で HTML 化。

---

## 6. 成果物

- `mohs_hardness_analysis.py`: 上記「3. 分析スクリプトで実施する内容」をマークダウンセクションとコードセルで整理したファイル。
- 実行結果として、**データの shape** と **各種統計量（describe, 欠損, 目的変数分布）** がレポート可能な状態にする。
- 必要に応じて `analyses/mohs_hardness.html` をポートフォリオ用に生成する。

この計画で進めれば、まずは shape と統計量の確認までを再現可能な形で実施できます。続けて相関分析やモデル検討を行う場合は、このスクリプトを拡張する形で対応できます。