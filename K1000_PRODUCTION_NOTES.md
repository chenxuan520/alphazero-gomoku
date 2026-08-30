# k1000 迭代复现与生产晋升记录（2026-08-27 收官）

本文档衔接 `TRAINING_NOTES.md`（iter440 主线）。iter440-evolved-64x8 的实验历史见
`EVOLVED64X8_ACCEPTANCE.md` 与 `.agents/plans/iter440-evolved-64x8.md`；
本文是 k1000 分支的全部信息与最终生产切换的完整档案。

> 若只看结论：当前 `stable`/`deep` 通道生产模型 = `k1000-best80-19ea8c34.net`
> （v1.1.0，发布传递 2026-08-27），fast 通道不变。

## 1. 链路总览

1. iter440 → Net2Net 32x4→64x8 精确扩张（函数等价，已验证 5 局面 0 diff）
2. 64x8 双配方（600-sim / 900-sim）均撞上同一 55.8–56.4% @48 外部平台——
   瓶颈是 **replay 信息量**（π 标签质量），不是容量
3. 转 k1000：32x4 冠军真身续训，自对弈 1000 sims/手（高信息量标签）
4. iter80 起金色快照出现；全门禁打满后由 best80 接任生产

## 2. k1000 训练复现命令

```bash
nohup ./bin/alphazero train \
  --run-dir runtime_1000sim \
  --iterations -1 --resume 0 --cache 1 \
  --init-model runtime/final_iter440.net \
  --init-best-model runtime/final_iter440.net \
  --init-buffer runtime/checkpoint.buffer.440.bin \
  --workers 48 --games-per-iter 80 --sims 1000 \
  --train-steps 200 --batch 128 --train-threads 16 \
  --lr .0003 --wd .0001 --value-weight 2 --hard-fraction .3 \
  --buffer 200000 --max-moves 200 --temp-moves 6 \
  --seed-hard-prob .3 --cpuct .8 --dir-eps .25 --dir-alpha .3 --fpu 0 \
  --gate-every 5 --gate-games 40 --gate-threshold .55 \
  --save-buffer-every 5 --trunk 32 --blocks 4 --seed 84400 \
  > runtime_1000sim/process.log 2>&1 &
```

- 数据与原 440 完全同源：行为/π/z 全部自产，`teacher_targets: 0` 可在
  `train.log` 的 `selfplay_done` 行核验。
- 平台期出现的迭代数 beat 推断：iter20 前的 0.55×3 是冷启动保守，iter110+
  才稳定 0.55+。

## 3. k1000 门禁链（内置 40 局 vs 冻结 440）

| 区间 | 典型读数 | 升迁 |
|---|---|---|
| iter5-15 | 0.45~0.55 | 平台期 |
| iter20/25 | 0.55 | ↑ best20 |
| iter45 | 0.575 | ↑ best45 |
| iter55 | 0.65 | ↑ best55 |
| iter60 | 0.525 | — |
| iter65 | 0.55（精确平线，float 阈值不升迁） | — |
| iter70 | 0.65 | ↑ best70 |
| iter75 | 0.50 | — |
| iter80 | 0.60 | ↑ best80 ← 冠军出身 |
| iter85 | 0.575 | ↑ best85 |
| iter90 | 0.525 | — |
| iter95/100 | 0.45 | — |
| iter105 | 0.60 | ↑ best105 |
| iter110/115/120 | 0.45/0.50/0.55 | — 死区起点 |
| iter123 | （run 自停于此） | 收官 |

教训：40 局 gate 噪声带宽约 ±10pp；**外部多局同档验证才能定性**（见 §4）。

## 4. 外部验收复现（全部命令）

头对头双色轮换、确定性逐局种子。开局噪声即 `--dir-eps .25 --temp-moves 6`：

```bash
./bin/alphazero arena --model-a MODEL --model-b runtime/final_iter440.net \
  --games 50 \       # 每档 50 局
  --sims {48|96|128|512} --workers 20 --seed FIXED --deterministic-games 1 \
  --temp-moves 6 --dir-eps .25
```

v1-v7 全门禁（game-old 的 7 档启发式，双色各 50 局）：

```bash
./bin/alphazero gauntlet --model MODEL --levels 1,2,3,4,5,6,7 --color both \
  --games 50 --workers 16 --sims {48|512} --seed FIXED
```

## 5. 最终结果（best80 / best85）

### 5.1 对 iter440 头对头（带开局噪声，每档 50 局）

| sims | best80 | best85 |
|---|---:|---:|
| 48 | 66% | **70%** |
| 96 | 56% | 58% |
| 128 | 56% | 56% |
| 512 | **70%** | 62% |
| 合计 | **62.0%** | **61.5%** |

追加无噪声 48-sim 大样本（120 局）：best80 65.0%、best85 58.3%；合并
65.3% vs 61.8%，两个 CI 重叠 → 互为镜像。

### 5.2 对 v1-v7 gauntlet（双色各 50 局）

best80：48sims 合计 **674/700 = 96.3%**，L6 黑90/白62、L7 100/100；
512sims 合计 **695/700 = 99.3%**，L6 100/94、其余全 100。

best85：48sims 合计 **668/700 = 95.4%**，L6 黑76/白66、L7 100/100；
512sims 除 L3 白 90 外全 100。

iter440 历史基线（10 局扫读）：L6 黑90/**白10**、L7 黑100/**白0** ——
白侧硬伤两代全部修复。

### 5.3 生产裁剪点

- best80 = 生产 stable+deep（头对头略强、48 sims 不缺格）
- best85 = 镜像备胎，已入 release 不部署

## 6. 踩坑记录（复现者必读）

1. **gate 阈值 float 陷阱**：`config.gate_threshold_` 是 float；
   `atof("0.55")` 存入后是 `0.5500000119…`，而 `22/40` 精确 double 为
   `0.5500000000000000444` —— 精确平线判负、best 不升迁。**这个偏保守的
   行为反而是好事**：平线不升迁，避免了噪声升上去后再回来守不住。
   想改双精度但不必动已收官分支。
2. **我的 arena `--sims-b` 双侧错峰**：用于回答"新模型 @48 相当于老模型
   @多少 sims"。结论：latest.30(64x8) @48 ≈ iter440 @75-85 sims。
   注意异步发布物测试的时候 B 侧是老模型,A 侧固定。
3. **trainer retention 会在下一轮发布时删除旧 latest.***：任何外测引用要在
   使用前复制到 `runtime_v2/candidates/<name>-<sha8>.net` 并 `chmod a-w`，
   否则跑到一半模型文件被删。
4. **best 也遵循同一 retention**：升迁是 rolling 更替换名,不是追加。升迁 born
   后立刻锁进 vault 再用。
5. **确定性种子 + 无噪声 arena 在近等模型间退化为"先手必胜"表演**：
   双色各一半全黑胜是正常读数；必须带 temp-moves + dir-eps 才有分辨力。
6. **40 局内置 gate 单点噪声可达 0.55→0.35→0.625 波动**，永远对齐到
   多档外部验证再下结论。
7. **eval 高耗时集中在 gate**：40 局 @600/900 sims 约 7 倍于一轮自对弈。
   对长跑路线，gate 是主要开销。
8. **长尾局决定一轮时长**：200 手天花板下个别可达 2h+；80 局/48 workers
   队列摊薄是标准做法。
9. **wins=25 losses=15 这种读数** —— loss > win 不要看错方向; color split
   里的 black_wins 配合 totals 一眼校验符号方向。
10. **模型文件部署后 CF Asset 还会先 404/旧版**，用 `?cb=<ts>` 强制重读。
11. **同 batch 发布工具 `promote_stable.py`** 会把 immutable 资产 + manifest 
    原子切换 + 回读校验；失败自动回滚。不要手写 manifest。

## 7. 生产晋升复现（promote_stable.py 原样走）

```bash
cd ~/self/llm/models/alphazero-gomoku
export PATH="$HOME/.nvm/versions/node/v22.22.1/bin:$PATH"   # 让 npx wrangler 可见
export CLOUDFLARE_API_TOKEN=...        # 环境里已有

python3 tools/promote_stable.py \
  --model /data00/home/lingchen.judy/self/alphazero-gomoku/runtime_v2/candidates/k1000-best80-19ea8c34.net \
  --label k1000-best80 --release v1.1.0 \
  --variant "k1000 iter80 gate champion; 62 percent vs iter440 across sims" \
  --channel stable --deploy
# deep 同样一条命令换 --channel deep
```

脚本完成：资产复制 + manifest 原子更新 + `wrangler deploy` + 线上回读校验。
2026-08-27 实际执行输出版本：stable=`3d47edaa-…`、deep=`dbb08db0-…`。

## 8. 引擎与前端（与模型解耦）

- v2 并行引擎内容寻址：`alphazero-gomoku-v2-8ec18530.js`，SRI 已入
  labs.js / gobang-web index.html
- 部署仓库：`~/self/llm/models/alphazero-gomoku`（Worker `alphazero-gomoku-model`）
  用 `promote_stable.py --deploy` 触发，或手动 `npx wrangler deploy`
- 第 25 章试玩、`gobang.011203.xyz` 都读 `channels/stable.json`，无需再改页面

## 9. 生产现状快照（2026-08-27）

- stable → k1000-best80-19ea8c34.net（v1.1.0）
- deep → 同上
- fast → fast-iter4-1bbd8634.net（历史蒸馏专用）
- 引擎：v2-8ec18530（worker 虚拟损失批处理）
- 回滚兜底：manifest 改回 `iter440-66aa74b7.net` 即可，资产不可变仍在货架上
