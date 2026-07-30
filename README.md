# Jittor 图学习挑战赛复现

本仓库整理了计图挑战赛热身赛与正式赛 Track 1 的代码、关键实验记录、复现数据和算法说明。正式赛最终方案不是单一图神经网络，而是面向时间有向图候选边的 Learning-to-Rank 系统：历史统计与序列模式负责“记忆”，正向/反向 TruncatedSVD 负责低秩图结构，XGBRanker 负责行内 100 个候选的最终排序。

## 核心结果

| 数据集 | 最终配置 | 本地 MRR |
|---|---|---:|
| dataset1 | v65 特征、去位置特征、双向 SVD32、1200 棵树、ranker/V18 = 0.95/0.05 | 0.8456168559 |
| dataset2 | v61 精简特征、去位置特征、双向 SVD32、850 棵树、纯 ranker | 0.5496001738 |
| 合计 | V69 独立可复现方案 | 1.3952170297 |

完整的算法推导、版本演进、特征体系、工程优化和复现逻辑见 `交付/计图挑战赛_代码复现与核心算法说明.docx`。

## 目录

```text
.
├─ gcn.py                              # 热身赛：两层 GCN、早停、多种子集成、配额后处理
├─ data/cora.pkl                       # 热身赛 Cora 数据
├─ formal_track1/
│  ├─ track1_solution_v62_dataset1_ranker.py  # 当前完整正式赛主线（含 v65/v67 profile）
│  ├─ track1_solution_v29_robust.py ...       # 关键版本演进源码
│  ├─ *_fusions*.py / fusion_*.py             # 秩融合与门控实验
│  ├─ downloads/track1_data.zip               # 正式赛原始数据
│  ├─ local_*_metadata.json / *.log           # 本地实验参数与指标
│  └─ result_v69_d1v65_1200_d2v62_local.zip   # 最终独立方案输出
├─ build_algorithm_doc.py              # 算法文档生成脚本
├─ 交付/                               # 最终算法说明文档
└─ UPLOAD_MANIFEST.md                  # 上传范围、排除项与大文件说明
```

## 环境

建议使用 Python 3.10 或 3.11。GPU 训练需要可用的 CUDA 与支持 GPU 的 XGBoost；没有 GPU 时可修改主脚本的 XGBoost 参数使用 CPU。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 热身赛复现

```powershell
python .\gcn.py --data_path .\data\cora.pkl --output_dir .\reproduced\warmup
```

模型先执行 `A+I` 的对称归一化，训练两层 GCN；使用验证集选择最佳轮数，再用 `train+val` 重训并进行多随机种子概率平均。类别配额后处理依赖特定测试分布，建议作为可开关实验对待。

## 正式赛复现

先解压数据：

```powershell
Expand-Archive .\formal_track1\downloads\track1_data.zip .\formal_track1\data -Force
```

运行最终独立方案：

```powershell
python .\formal_track1\track1_solution_v62_dataset1_ranker.py `
  --data-root .\formal_track1\data `
  --output-dir .\reproduced\formal_track1 `
  --datasets dataset1,dataset2 `
  --ranker-rows 130000 `
  --svd-dim 32 `
  --seed 20260525
```

主脚本会完成：时间切分与伪候选构造、历史索引、180 维原始特征、profile 选列、双向时间加权 SVD、XGBRanker 训练、验证 MRR 选权重、完整历史重建以及逐块生成提交。dataset1 和 dataset2 的最终 profile/树数在脚本的数据集配置中分别设定。

## 复现注意事项

- 每个查询组严格包含 100 个候选；XGBRanker 的 `group` 必须保持为 `[100, 100, ...]`。
- 验证切分必须按时间顺序，不能随机打乱边，否则会产生未来信息泄漏。
- 候选位置特征在最终方案中关闭，避免模型学习人工插入位置。
- 输出分数是每行 min-max 归一化后的排序分数，不是校准概率。
- V71–V94 的部分融合依赖未单独保留的外部/open 提交；完全自包含复现以 V69 为准。

## 资料完整性

GitHub 不适合作为 17.6GB 重复训练产物的对象存储。本仓库保留了产生结果所需的数据、源码、最终结果和实验元数据；重复 CSV、各中间版本 ZIP、缓存及密钥不进入版本控制。详见 `UPLOAD_MANIFEST.md`。
