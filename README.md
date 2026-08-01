# jittor-lc666-基于图学习的推荐任务

计图人工智能挑战赛正式赛 Track 1 可复现代码。任务是在时间有向图中，对每个查询给出的 100 个候选目标节点进行排序，使真实目标尽可能排在前列。

本仓库仅包含正式赛最终方案、复现依赖、审核说明和必要实验记录，不包含热身赛代码。

## 开源地址

- GitHub：`https://github.com/purplellllll/jittor-lc666-graph-learning-recommendation`
- GitLink：`https://gitlink.org.cn/llllllc/lc666`

GitHub 仓库名受平台限制只能使用 ASCII 字符，中文项目名保留在 README 与说明文档中。

## 最终方案

正式赛被建模为时间动态图上的候选边排序问题。完整流程为：

1. 按时间切分训练边，构造与正式测试一致的 100 候选验证行。
2. 对历史有向边建立计数、最近时间、转移、跳跃转移、节点窗口统计和测试候选分布索引。
3. 对正向图与反向图分别计算时间衰减的 TruncatedSVD 表示。
4. 为每个候选生成绝对统计、行内相对排名、序列模式、图结构和无标签测试分布特征。
5. 使用 XGBRanker 的 `rank:ndcg` 目标训练行内排序器，并以本地 MRR 选择最终混合权重。
6. 使用完整历史重建特征，分块输出 `dataset1.csv`、`dataset2.csv` 和 `result.zip`。

最终 V69 独立方案：

| 数据集 | 特征配置 | 树数 | SVD | Ranker 权重 | 本地 MRR |
|---|---|---:|---:|---:|---:|
| dataset1 | v65，167/180 维，不使用候选位置 | 1200 | 正/反向各 32 维 | 0.95 | 0.8456168559 |
| dataset2 | v61，105/180 维，不使用候选位置 | 850 | 正/反向各 32 维 | 1.00 | 0.5496001738 |
| 合计 | V69 | - | - | - | 1.3952170297 |

## 仓库结构

```text
.
├── code/
│   ├── main.py                 # 训练、验证、预测和打包的唯一入口
│   ├── run.sh                  # Ubuntu 一键复现脚本
│   ├── check_environment.py    # 环境与依赖版本检查
│   └── config.json             # 最终 V69 参数快照
├── docs/
│   ├── 复现说明.md
│   └── 正式赛算法与复现说明.tex  # 完整算法、优化逻辑与复现说明 LaTeX 源文件
├── records/
│   ├── final_metrics.json      # 最终本地指标与输出行数
│   └── reproduction.log        # dataset1 完整训练日志
├── requirements.txt
├── environment.yml
├── LICENSE
```

## 审核环境

与通知中的复现环境保持一致：

- Ubuntu 22.04
- CUDA 12.4
- Python 3.10
- Jittor 1.3.10.0

本方案的候选排序器使用 XGBoost；Jittor 版本仍按主办方审核镜像固定。完整 Python 依赖见 `requirements.txt`。

## 安装

```bash
sudo apt-get update
sudo apt-get install -y python3.10-dev g++ build-essential libomp-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python code/check_environment.py
```

## 数据

代码默认从主办方提供的地址下载 `track1_data.zip`，并解压到 `data/track1/`：

```text
data/track1/
├── dataset1/train.csv
├── dataset1/test.csv
├── dataset2/train.csv
└── dataset2/test.csv
```

也可以提前把官方数据压缩包放到 `data/track1_data.zip`。仓库不重新分发测试标签或任何外部标注。

## 一键复现

在仓库根目录执行：

```bash
chmod +x code/run.sh
bash code/run.sh
```

等价命令：

```bash
python code/main.py \
  --preset final \
  --data_dir data/track1 \
  --zip_path data/track1_data.zip \
  --output_dir outputs \
  --result_zip outputs/result.zip
```

输出：

```text
outputs/
├── dataset1.csv       # 61,051 行，每行 100 个分数
├── dataset2.csv       # 153,420 行，每行 100 个分数
├── metadata.json
└── result.zip
```

固定随机种子为 `20260525`。GPU 不可用时，XGBoost 会自动回退到 CPU `hist`，但耗时会显著增加。

## 无测试标签声明

- 训练过程只读取 `train.csv` 中的历史边标签。
- `test.csv` 仅提供 `src`、`time` 和 `c1...c100` 候选集合；代码不会读取或构造测试集真实目标。
- 测试候选频率、首次/末次出现时间和无标签共现统计属于转导式无监督特征，不包含测试标签。
- 验证集的真实目标完全由训练边按时间后移构造，避免未来信息泄漏。
- 最终方案不依赖未公开的外部提交；V71-V94 融合实验不属于本审核包。

## 资源与耗时

建议至少 16 核 CPU、32 GB 内存和 12 GB 以上显存。特征构建以 `float32`、排序键和分块预测控制内存；完整训练时间取决于 CPU、磁盘和 GPU，审核时以日志阶段标记为准。

## 说明材料

- 命令行与故障排查：`docs/复现说明.md`
- 完整算法、优化逻辑与复现说明：`docs/正式赛算法与复现说明.tex`
- PDF 说明和最终审核 ZIP 按当前整理结果另行补充，本次仓库提交暂不包含 PDF。

## 许可证

本项目采用 MIT License，详见 `LICENSE`。
