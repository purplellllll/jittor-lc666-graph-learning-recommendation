# jittor-lc666-基于图学习的推荐任务

计图人工智能挑战赛正式赛 Track 1 复现仓库。任务是在时间有向图中，对每个查询给出的 100 个候选目标节点排序，使真实目标尽可能靠前。

本仓库只保留正式赛方案，不包含热身赛代码。最终选定版本是 **V83 双分支高分方案**：A 榜成绩 **1.4591**。原先记录的 `1.3952170297` 是 V69 主模型分支的本地时间验证 MRR，不是最终榜单成绩。

## 仓库地址

- GitHub：<https://github.com/purplellllll/jittor-lc666-graph-learning-recommendation>
- GitLink：<https://gitlink.org.cn/llllllc/lc666>

GitHub 仓库名受平台限制使用 ASCII；中文项目名保留在 README 和说明文档中。

## 最终算法

最终系统由两个互补的候选排序分支组成。

### 1. 时间动态图排序主分支（V69）

1. 按时间切分训练边，构造与正式测试一致的 100 候选验证行；
2. 从历史有向边构建 pair 频次与时效、相邻转移、跳跃转移、节点时间窗口、最近序列和候选分布统计；
3. 在正向图和反向图上分别计算 32 维时间加权 TruncatedSVD；
4. 为每个候选生成绝对统计、行内归一化、倒数排名、z-score、序列模式和图结构特征；
5. 以每个查询行为 group，训练 `rank:ndcg` 的 XGBRanker，并与稳定图规则先验做行内分数校准；
6. dataset1 使用 v65/1200 棵树/混合系数 0.95，dataset2 使用 v61/850 棵树/混合系数 1.00。

### 2. 数据集感知的双分支路由（V83）

两个数据集的结构与稀疏度不同，统一权重并不最优。最终根据严格本地验证与候选实验确定：

| 输出 | 采用的排序分支 | 选择原因 |
|---|---|---|
| `dataset1.csv` | V69 时间动态图排序器 | 主分支在 dataset1 的时间验证更稳定，MRR 为 0.8456168559 |
| `dataset2.csv` | 互补预测分支 | 在 dataset2 上提供更强的结构互补性，直接保留其候选次序更稳健 |

最终路由可写成：

\[
S_d=\begin{cases}
S_{\mathrm{graph}}, & d=\mathrm{dataset1},\\
S_{\mathrm{comp}}, & d=\mathrm{dataset2}.
\end{cases}
\]

选择是在数据集层完成的，不对两个分支的原始分值强行平均，从而避免不同分数尺度破坏已经可靠的行内排序。开发阶段还比较了 reciprocal rank、RRF、Borda、rank product、归一化加权和置信度门控；最终 V83 的分数据集路由最简单、最稳定，也与高分产物的逐文件指纹一致。

## 结果与版本边界

| 记录 | 数值 | 含义 |
|---|---:|---|
| V69 dataset1 本地 MRR | 0.8456168559 | 主分支严格时间验证 |
| V69 dataset2 本地 MRR | 0.5496001738 | 主分支严格时间验证 |
| V69 本地合计 | 1.3952170297 | 只用于主分支模型选择 |
| V83 A 榜 | **1.4591** | 最终双分支提交成绩 |

榜单成绩与本地 MRR 来自不同评价数据，不能混写或直接相减。完整记录见 `records/final_metrics.json`。

## 仓库结构

```text
.
├── code/
│   ├── main.py                 # V69 主分支：训练、验证、预测和打包
│   ├── ensemble.py             # V83：双分支路由、全量校验和最终打包
│   ├── run.sh                  # 最终高分版本一键入口
│   ├── run_primary.sh          # 仅复现 V69 主分支
│   ├── check_environment.py
│   └── config.json             # 最终 V83 参数与产物指纹
├── docs/
│   ├── 复现说明.md
│   └── 正式赛算法与复现说明.tex
├── records/
│   ├── final_metrics.json
│   ├── artifact_audit.md
│   └── reproduction.log
├── requirements.txt
├── environment.yml
└── LICENSE
```

## 环境

- Ubuntu 22.04
- CUDA 12.4
- Python 3.10
- Jittor 1.3.10.0
- XGBoost 2.1.4

```bash
sudo apt-get update
sudo apt-get install -y python3.10-dev g++ build-essential libomp-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python code/check_environment.py
```

## 输入准备

官方数据目录：

```text
data/track1/
├── dataset1/train.csv
├── dataset1/test.csv
├── dataset2/train.csv
└── dataset2/test.csv
```

也可将官方 `track1_data.zip` 放到 `data/track1_data.zip`，主程序会自动解压。

逐字节复现 V83 还需要归档的互补预测分支，放置为：

```text
data/complementary_result.zip
```

该压缩包根目录必须包含 `dataset1.csv` 与 `dataset2.csv`，归档 SHA256 应为：

```text
974582aa2f8f156bbe7f539487450e3e1d7ca75cfaec6027ee2999ad53fdb461
```

集成代码只读取其中的候选分数，不读取测试标签；其内部训练过程不由当前仓库重新声明。该边界通过固定文件指纹、行数和列数显式记录，避免把不可验证部分误写成主模型训练代码。

## 一键复现最终高分版本

```bash
chmod +x code/run.sh code/run_primary.sh
bash code/run.sh
```

若互补分支位于其他位置：

```bash
COMPLEMENTARY_RESULT_ZIP=/path/to/complementary_result.zip bash code/run.sh
```

流程先训练 V69 主分支，再由 `code/ensemble.py` 生成最终：

```text
outputs/
├── primary_v69.zip
├── high_score/
│   ├── dataset1.csv
│   ├── dataset2.csv
│   └── ensemble_metadata.json
└── result.zip
```

只复现 V69 主分支：

```bash
bash code/run_primary.sh
```

只执行最终路由与产物校验：

```bash
python code/ensemble.py \
  --primary_result outputs/primary_v69.zip \
  --complementary_result data/complementary_result.zip \
  --output_dir outputs/high_score \
  --result_zip outputs/result.zip
```

最终归档会检查 ZIP CRC、成员名、完整行数、每行 100 个有限浮点数，并输出 SHA256。已归档高分产物的 SHA256 为：

```text
5b9e61fd7ebdb20178b45ae787c2498eaa5ac019081a31ff21ef2475acdc7e6b
```

该值是历史归档容器指纹。重新打包会改变 ZIP 成员时间戳，因此整包 SHA256 可以不同；真正决定预测版本的是两个 CSV 成员指纹，程序默认强制核对：

```text
dataset1.csv  829b6877d8a4663f512005e4d1d431d906991c517bb3708e03a99b161cdb7ee5
dataset2.csv  a56c257f97710bbe8714fcd6d23b1fb4e5b7ea528d2404011457d452f044fb4a
```

## 合规说明

- 主分支只使用公开训练历史和测试文件中的 `src`、`time`、`c1...c100` 候选集合；
- 验证标签来自训练边的严格时间后移切分，不使用隐藏测试标签；
- 测试候选频率和共现统计属于无标签转导特征；
- 第二预测分支作为固定、可校验的模型输出输入，集成层不把它伪装成 `main.py` 训练得到的结果；
- 最终说明将 V69 本地验证与 V83 榜单成绩分开报告。

## 文档

- 快速复现与故障排查：`docs/复现说明.md`
- 完整算法、公式、优化演进和复现逻辑：`docs/正式赛算法与复现说明.tex`

## 许可证

MIT License，详见 `LICENSE`。
