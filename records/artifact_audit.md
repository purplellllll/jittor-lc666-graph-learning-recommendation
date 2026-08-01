# 最高分版本产物审计

审计时间：2026-08-01（Asia/Shanghai）

## 结论

- 榜单：Jittor-7 Track 1，stage 585，A 榜；
- 队伍：lc666；
- 已核实成绩：**1.4591**；
- 仓库最终配置：**V83 数据集感知双分支路由**；
- V69 的 `1.3952170297` 只保留为主分支本地时间验证结果。

## 最终产物

| 项目 | 值 |
|---|---|
| 本地归档 | `formal_track1/result_v83_d1v69_d2open.zip` |
| 文件大小 | 52,230,691 bytes |
| ZIP SHA256 | `5b9e61fd7ebdb20178b45ae787c2498eaa5ac019081a31ff21ef2475acdc7e6b` |
| ZIP CRC | passed |
| 根目录成员 | `dataset1.csv`, `dataset2.csv` |

ZIP 容器哈希包含成员时间戳；重新打包后整包 SHA256 可以变化。候选预测版本以成员哈希为准。

## 成员级来源验证

| 成员 | 路由来源 | 行数 | 列数 | SHA256 |
|---|---|---:|---:|---|
| `dataset1.csv` | V69 主分支 | 61,051 | 100 | `829b6877d8a4663f512005e4d1d431d906991c517bb3708e03a99b161cdb7ee5` |
| `dataset2.csv` | 互补预测分支 | 153,420 | 100 | `a56c257f97710bbe8714fcd6d23b1fb4e5b7ea528d2404011457d452f044fb4a` |

比对结果：

- V83 的 `dataset1.csv` 与 `result_v69_d1v65_1200_only_tmp.zip` 中的成员逐字节相同；
- V83 的 `dataset2.csv` 与归档互补分支中对应成员逐字节相同；
- 两个成员均完成全行校验，每行恰好 100 个有限浮点数；
- `code/ensemble.py` 默认把上述成员哈希作为最高分版本门禁。

## 其他候选的处理

V84/V86 对 dataset1 做倒数名次校准，V88--V90 使用置信度门控，V92--V94 使用 Borda、RRF 或归一化加权。这些文件仍保留在本地历史实验目录中用于追溯，但仓库默认入口、配置、指标与说明均已统一指向 V83，不再把 V69 或其他实验候选标记为最终产物。

## 日志边界

`records/reproduction.log` 是 V69 dataset1 的完整训练与预测日志。末尾的打包失败来自该次隔离运行没有同时生成 dataset2，不影响已经成功写出的 dataset1 成员；最终双分支产物完整性以本审计、`records/final_metrics.json` 和 `ensemble_metadata.json` 为准。
