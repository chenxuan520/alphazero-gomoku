# AlphaZero Gomoku — iter440-evolved-64x8 验收报告

日期：2026-08-24
分支：`iter440-evolved-64x8`
路线：**纯 AlphaZero 自对弈**（无任何教师信号；行为/π/z 全部来自当前 64x8 网络自身）

## 1. 训练概要

- 基线：冻结 `runtime/final_iter440.net`（32x4，生产 stable/deep）
- 起点：Net2Net 精确扩张 32x4→64x8（711,597 参数，推理零差异已验证）
- 数据：`--init-buffer` 导入 iter440 的 200,000 条纯自对弈 replay + 30 轮新自对弈
  （每轮 80 局 × 600 sims，全程 `teacher_targets: 0`）
- 优化：全新优化器，lr=1e-4，100 steps/iter × batch 128，共 3000 步
- 内部 gate（600 sims，temp6）：0.550 → 0.550 → 0.600↑ → 0.625↑ → 0.475 → 0.425
  （best 定格 iter20；后续 10 轮按 latest 继续进化）

## 2. 外部验收（候选 = `runtime_v2/evolved64x8_pure/checkpoint.latest.30.net`）

统一口径：arena 颜色逐局轮换、确定性逐局种子、temp-moves 6、dir-eps 0；
对照 `runtime/final_iter440.net`。

| 预算 | 局数 | 比分 | 胜率 | 95% CI | 结论 |
|---|---:|---:|---:|---|---|
| 48 sims（3 种子 62300/73100/84200） | 360 | 203:157 | **56.4%** | [51.3%, 61.5%] | ✓ 显著优势（CI 不含 50%） |
| 96 sims（2 种子 62400/62410） | 200 | 107:93 | 53.5% | [46.6%, 60.4%] | ✓ 正向（合并 48s 后 CI [52.2%, 58.5%]） |
| 600 sims | 40 | 26:14 | **65.0%** | [50.2%, 79.8%] | ✓ 预算越高优势越大 |

gauntlet 回归（48 sims，每级每色 10 局，同色同种子）：

| 模型 | L6 黑/白 | L7 黑/白 | 合计 |
|---|---|---|---|
| iter440 | 90% / 10% | 100% / 0% | 20/40 = 50% |
| latest.30 | 70% / 80% | 100% / **100%** | **35/40 = 87.5%** |

- L6 黑色侧 70% vs 90%：n=10 下为噪声范围；白色侧 80% vs 10%、L7 白色 100% vs 0%
  修复的是 iter440 有文献记录的白色侧结构性弱点。
- 候选 SHA256：`28be769f201ea12fc8aab4a489cbe9e0698a52e9c33c449928845acb79719ddf`
  （另见 `runtime_v2/evolved64x8_pure/SHA256SUMS.txt`）

**判定：全部验收门禁通过；latest.30 在联合证据下明确强于冻结 iter440。**

## 3. 过程教训（用于下次）

1. temp0 确定性对局在近等网络间退化为"先手必胜"表演（12/12 全黑胜），
   不能用于强度裁决；验收协议必须带 temp-moves≥6 的开局多样性。
2. 内部 gate 的移动靶 best 与最终对 iter440 的外部强度可能反向
   （best.20 48-sims 外部 50.8% 反而低于 latest.30 的 56.4%）——
   内部 gate 只用于止损与快照，强度宣判一律以外部对冻结基线的大样本 results 为准。
3. `gate_threshold_` 为 float，精确等于阈值的 gate（如 22:18）不会触发升迁；
   影响可忽略（平线不升迁反而更保守），后续可改 double。
4. 长尾对局决定单轮时长（200 步上限下个别局 ~2h），128s 预算下 about 1.6-2h/轮。

## 4. 产物清单

- 候选权重：`runtime_v2/evolved64x8_pure/checkpoint.latest.30.net`（64x8）
- 备选：`checkpoint.best.20.net`（内部 gate 冠军）
- 训练目录自洽可 resume（`latest.current = 30 20 30`）
- 实验日志：`runtime_v2/eval_*.log`

## 5. 浏览器推理实测（发布前关键约束）

生产引擎 = 浏览器内 JS 推理（labs.js 实验台，sims 用户可选 12/24/48/96/120）。
同机同引擎 Chrome 实测单步耗时（每点 ≥8 手、含开局多样化）：

| 模型 | 48 sims | 96 sims | 12 sims(extrap.) |
|---|---:|---:|---:|
| iter440 (32x4) | 3.7s | 8.0s | ~1s |
| latest.30 (64x8) | 33.5s（9.0×） | 51.8s（6.5×） | ~8s |

强度-预算曲线（vs iter440，同预算对比）：12 sims 52.9% → 48 sims 56.4% →
600 sims 65.0%。**优势随预算扩大，低预算下收敛到接近平手。**

结论：
- **deep 通道换 latest.30 有数据支撑**（65% @600，语义就是高预算慢思考）。
- **stable 直接换 64x8 会破坏体验**（3.7s→33.5s @48），且低 sims 下强度
  优势未达到统计显著；除非引擎提速（WebGPU/批量化，预计 5-10× 空间）
  或训练出小尺寸的强模型。

## 6. 未执行（待确认）

- 生产 `channels/stable.json` / `deep.json` 未改动；模型资产仓库未推送。
- 发布动作在 `~/self/llm/models/alphazero-gomoku/`（CF Worker，public/ 资产），
  本机无 wrangler CLI，部署需用户授权或代为执行。
