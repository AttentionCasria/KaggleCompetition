# Store Sales - Time Series Forecasting (LightGBM)

这是一个用于解决 Kaggle 上 **[Store Sales - Time Series Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting)** 比赛的时间序列预测项目。该项目主要基于传统的机器学习（Machine Learning）技术，特别是利用 LightGBM（梯度提升决策树）模型，结合精细的时间序列特征工程，来实现对商店未来销售额的准确预测，而非使用深度学习模型。

## 项目结构
仓库中包含多个 Jupyter Notebook，用于数据分析、特征构建与模型训练：
- `kaggle_test.ipynb`: 核心训练与预测脚本，包含了主要的数据合并流程、特征提取以及基于 LightGBM 的模型训练代码。
- `test.ipynb`, `origin.ipynb`, `xsyhhh.ipynb`: 数据探索（EDA）、模型调优及实验对比的其他 Notebook 分支。

## 核心流程（机器学习 Pipeline）

1. **数据加载与预处理**
   - 读取核心数据：`train.csv` 和 `test.csv`
   - 合并外部数据：商店元数据 (`stores.csv`)、交易记录 (`transactions.csv`)、节假日信息 (`holidays_events.csv`)、每日油价 (`oil.csv`)
   - 缺失值处理：比如油价的连续前向与后向填充 (ffill/bfill)。

2. **特征工程 (Feature Engineering)**
   这是本项目最核心的环节，用于捕捉时间序列数据中的规律：
   - **时间特征**：从日期中提取 年、月、日、周数、季度、星期几、是否周末、月初月末等。
   - **滞后特征 (Lag Features)**：构造 1天、7天、14天、28天 等历史销售数据，以捕获自相关性与周期性。
   - **滑动窗口特征 (Rolling Features)**：计算 7天、14天、28天 的滑动均值等统计量，捕捉近期趋势。
   - **衍生特征**：计算单次交易平均销售额 (`sales_per_transaction`)、油价滑动均值等。
   - **分类变量编码**：使用 `Scikit-learn` 的 `LabelEncoder` 对非数值类别（如商品类型、假日类型、城市、州）进行数值型编码。

3. **模型训练与预测 (Model Training)**
   - **目标值转换 (Log Transformation)**：对销售额 (`sales`) 进行 `log1p` 对数变换以稳定方差，缓解极端波动，在预测后使用 `expm1` 还原。
   - **模型选择**：使用强大的梯度提升树模型 `LightGBM` (GBDT)，其在处理表格型数据时训练速度快且效果优异。
   - **验证评估**：以 2017-06-15 或 2017-07-01 为界进行基于时间序列的验证集划分（防止未来数据穿越），采用 RMSLE（均方根对数误差）作为评估指标。

## 如何运行

1. 下载比赛数据集并放置于 `data/` 目录或修改 Notebook 中的 `DATA_PATH` 路径配置。
2. 安装环境依赖：
   ```bash
   pip install pandas numpy lightgbm scikit-learn
   ```
3. 运行 `kaggle_test.ipynb` 即可自动完成从特征工程到模型预测的全流程，并输出 `final_submission.csv` 提交文件。

## 环境要求
- Python 3.x
- Pandas, Numpy
- LightGBM
- Scikit-learn
# KaggleCompetition
