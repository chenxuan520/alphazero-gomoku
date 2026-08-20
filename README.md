# alphazero-gomoku

15×15 自由规则五子棋的 AlphaZero 式自我对弈强化学习项目，纯 CPU 训练。

**状态：主线已在 iter440 收口。** 纯自我对弈（无 teacher 动作/价值标签；
hard curriculum 仅用手写三/四连检测重采样自有错题）共完成
440 个唯一 iteration、33,920 局去重自对弈（含重启实际 35,000 局）、约 133 万
局面和 87,280 个 Adam step。最终网络 32ch/4blocks、191,853 个序列化参数。

- 最终训练快照：`runtime/final_iter440.net`
  (`66aa74b70cd2a73b1c61616df13aaa4a61073d3c5cdf10c1979084212827b2c4`)
- 收口时 gate best：`runtime/best_at_stop_iter435.net`
- 正式网页稳定模型：iter330（低预算综合表现更稳）
- 完整验收、横评、错误修复、停止理由见 `TRAINING_NOTES.md`
- GitHub：`https://github.com/chenxuan520/alphazero-gomoku`
- 全部模型：GitHub Release `v1.0.0`（模型不进入 git 历史）
- 独立低模拟次数实验见 `FAST_POLICY_DISTILLATION.md`；未通过 48/96/600
  双预算门禁前不会替换 v1.0.0 或线上 iter330。

复现验收：
```bash
./bin/alphazero gauntlet --model runtime/final_iter440.net \
    --levels 1,2,3,4,5,6,7 --games 10 --workers 24 --sims 800

# 只测模型执白/黑(例如 L7 后手 20 局)
./bin/alphazero gauntlet --model runtime/best_at_stop_iter435.net \
    --levels 7 --games 20 --workers 8 --sims 96 --color white
```

训练栈完全建立在 deeplearning 仓库的 WIP 组件副本之上（`lib/` 下：
`FloatTensor4D` / `BatchedConv2D` / `BatchNorm2D` / `ResidualBlock2D` /
`FloatLinear` / `PolicyValueResNet` / `PolicyValueLoss` / `FloatAdamW` /
`ThreadPool`），不依赖任何外部 ML 库。

## 训练原理（AlphaZero 主循环）

```
自我对弈(MCTS+当前网络) → 样本 (s, π, z) 进回放池 → 训练网络 → 与历史最优打擂台(胜率≥55%晋级) → 循环
```

- **网络**：策略-价值双流 ResNet。默认 4 卷积 stem + 4 残差块（trunk 32 通道），
  策略头输出 225 格 logits，价值头 tanh 输出 [-1,1]（行棋方视角）。约 19 万参数。
- **棋盘编码**：4 个 15×15 平面 = 己方子 / 对方子 / 上一手位置 / 行棋方颜色。
- **MCTS**：PUCT 选边 + 根节点 Dirichlet 噪声；先手手数内按访问数分布温度采样，之后贪心。
- **训练目标**：`PolicyValueLoss` = 合法手 masked 策略交叉熵 + 价值 MSE；AdamW
  （权重衰减只作用于 conv/linear 权重，BN/bias 不衰减）。
- **数据增广**：棋盘 8 对称（4 旋转 × 镜像），采样时随机取一种同时变换平面与策略目标。
- **评估缓存**：自对弈同轮内共享局面缓存，键包含行棋方、上一手和完整棋盘；
  并行对局的重复开局直接命中，权重更新即失效。

最终主线配方：

| 阶段 | 默认量 |
| --- | --- |
| 自对弈 | 80 局 × 48 worker，每步 600 次 MCTS 模拟，上限 200 手截断 |
| 训练 | 200 步 × batch 128（recency + hard mining + 8 对称），AdamW lr=1e-3 |
| 评估 | 每 5 轮：vs 随机棋手 20 局 + 与 best.net 擂台 20 局 |
| 保存 | latest/best/replay 版本化 bundle，由 `latest.current` 原子三元组提交 |

主训练绑定 CPU0-55；低预算横评占 CPU56-63。最终阶段约 40–55 分钟/iteration。

## 构建与测试

```bash
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
cd .. && ./bin/test_az        # 4411 checks, 0 failed
```

## 命令

```bash
# 最终正式配方（项目已收口；仅在复现实验时重启）
./bin/alphazero train --run-dir runtime --workers 48 --games-per-iter 80 \
    --sims 600 --train-steps 200 --batch 128 --lr .001 --wd .0001 \
    --value-weight 2 --buffer 200000 --max-moves 200 --temp-moves 6 \
    --seed-hard-prob .3 --cpuct .8 --dir-eps .25 --dir-alpha .3 --fpu 0 \
    --gate-every 5 --gate-games 20 --gate-threshold .55 --save-buffer-every 10

# 断点续训：同参数重跑即可（读取 latest.current 指定的 latest/best/replay bundle）
# --no-resume 冷启动；--no-cache 关评估缓存

# 模型 vs 随机棋手（交替执色）
./bin/alphazero eval --model runtime/best.net --games 20 --sims 100 --workers 8

# 两个模型打擂台
./bin/alphazero arena --model-a runtime/final_iter440.net \
    --model-b runtime/best_at_stop_iter435.net --games 50 --sims 48

# 显式开启严格有界子树复用（默认关闭，方便 A/B）
./bin/alphazero arena --model-a runtime/iter405.net --model-b runtime/iter360.net \
    --games 50 --sims 48 --reuse-tree 1

# 人机对战（你是白棋 O，输入 "行 列"）
./bin/alphazero play --model runtime/best.net --sims 100

# 网络前向性能 / 规模
./bin/alphazero bench --concurrent 40 --iters 30
./bin/alphazero info

# 原版 game-old 七档 AI 的两两排名（需同级 game-old 仓或显式环境变量）
GAME_OLD_FRONTEND=~/self/game-old/gobang-web/frontend \
  node tools/js_engine/rank_pair.js 6 7 20 4242
```

## 监控训练

所有结构化日志追加到 `runtime/train.log`（JSON 行，同时打到 stderr）：

```bash
tail -f runtime/train.log
grep eval_random runtime/train.log   # 看 vs 随机胜率变化（弱网默认 0%，起来后应攀升）
grep '"phase":"gate"' runtime/train.log  # 看擂台晋级记录
```

关键字段：
- `avg_moves`：自对弈平均局长。随机初期顶着 200 手上限，变强后会显著缩短（几十手出胜负）——这是最直观的早期变强信号。
- `policy_loss`：从 ln(225)≈5.42 一路降。
- `cache_hit_rate`：评估缓存命中率，开局重复局面多时 30%+ 正常。
- `gate`：`challenger_wins/best_wins/draws/rate`，`promoted` 表示新网络晋级为 best。

## 文件布局

```
lib/            框架组件（自 deeplearning 仓 WIP 拷贝，勿删勿改同步方向）
src/game/       Gomoku 规则/编码/8对称
src/mcts/       MCTS (PUCT + Dirichlet 根噪声)
src/train/      评估器(INetEvaluator/缓存) / 回放池 / 自对弈 / 擂台 / 训练器
test/           单元测试(规则/编码/对称/MCTS 必杀局面)
runtime/        训练产物：版本化 latest/best/replay bundle、latest.current、train.log
runtime.out     nohup stdout
```

## Release 模型包

仓库只跟踪源码与文档。`v1.0.0` Release 提供：

- `alphazero-gomoku-all-models-v1.0.0.zip`：训练过程中保存的全部 `.net` 快照；
- `alphazero-gomoku-final-v1.0.0.zip`：iter440 最终快照、iter435 gate best、
  线上 iter330 以及模型清单/校验和。

Replay Buffer（约 860MiB）与 optimizer state 不是模型权重，不上传 Release。

## 设计取舍备忘

- **FPU reduction 默认 0**：>0 时在值几乎平坦的早期局面会锁死探索（首个被访问的边永远领先）。
- **终局价值约定**：赢的局面返回 -1（按"该走棋的一方视角"的约定，终局时没有下一手，行棋方即输家视角），保证 MCTS 回溯逐层翻转的符号一致。
- **worker 网络副本**：组件内部线程池不可多实例并发复用，所以每个自对弈 worker 持有独立网络副本（拷贝含 BN running statistics，不只可训练参数）。
- **无 resign**：v1 未做认输加速，弱网阶段价值信号不可靠，先靠 max-moves 截断。
- **MCTS 子树复用默认关闭**：显式开启后真实落子继承 child subtree；node/edge
  有硬预算，触顶丢树 fresh 重建。批准版 L6@48 耗时 187s→122s。
- **低预算 policy 蒸馏**：v1.1.0 fast 模型只更新 policy head，冻结 iter440
  trunk/value/BN；L6@48(reuse) 总胜率72%，L6@96为86%，L7@48/96均
  100%，并在600 sims下分别40:0击败冻结 iter440 与晋升快照iter360。
  完整方案与门禁见
  [`FAST_POLICY_DISTILLATION.md`](FAST_POLICY_DISTILLATION.md)。
- **生产选模**：8模型、1400局、48-sim全循环中iter440以210:140/60%排名
  第一，fast iter4以198:152/56.6%第三；因此`stable/deep`用iter440，
  `fast`通道保留蒸馏版，避免把外部对手专项优势误当成全代际优势。
- **崩溃安全 checkpoint**：latest/best/replay 三者准备并 fsync 后，只通过一次
  `latest.current` rename 提交；平台 monitor 用 nonce 让 trainer 在完整轮边界暂停，
  不再外部 kill。
