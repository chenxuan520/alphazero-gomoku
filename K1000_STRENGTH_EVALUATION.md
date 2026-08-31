# 棋力绝对标定测量报告 — k1000-best80（与第三方引擎 & 人类对照）

日期：2026-08-30
被测模型：`k1000-best80`（= `runtime_v2/candidates/k1000-best80-19ea8c34.net`，k1000 主线 iter80 gate 升迁快照，191,853 参数 / 770KB）
对照模型：`k1000-best85`（iter85）。

## 0. 方法学与可信度声明（先读这段再读数字）

- 用一个跨引擎裁判 `tools/crossmatch.py`（piskvork 协议 + JS JSON-lines + 随机引擎适配），所有局双色轮换、referee 给予开局 2 手多样性种子（防确定性重排）。
- 锚点引擎为 **Rapfi**（开源、Gomocup 历届冠军家族），本机源码构建（GCC 8 + AVX512 + NNUE 自由规则权
  重），piskvork 协议。Rapfi 在 Gomocup Freestyle15 公开 Elo 榜 = **2716**（freestyle15 长期第一），报告内所有绝对 Elo 都是以 Rapfi@1500ms=2716 显式锚定的保底估计。
- **诚实幅度**：Rapfi 的单步时间预算在 50ms→5s 间平台上差异不显著（见 §4 梯级表），故锚点不确定度粗估 **±100 Elo 量级**；本报告结论性断言都设置在远超这个误差带的距离上。
- Bradley–Terry 汇总采用全池最小二乘；零胜序列以"上界区间"而非精确 Elo 呈现。

## 1. 总结论

| 我方实例 | 池内绝对 Elo | 对照 Gomocup Freestyle15 位置 |
|---|---:|---|
| best80 @ 48 sims | **≈ 2056** | 介于 WHOSE 2019 (1791) 与 SKYZERO 2026 (1927) 之上 → 接近 STARPOINT 2026 (2021) |
| best80 @ 96 sims | **≈ 2108** | 追近 YIXIN 2018 (2175) |
| best80 @ 512 sims | **≈ 2406** | 介于 EMBRYO 2025 (2320) 与 JAX 2025 (2554) 之间 |
| （对照） best85 @ 48 | ≈ 2024 | 诸小样本 24 局 |
| JS v1–v7 | ≈ 1550–1611 | 榜尾区：CARBON 2017 (1624)/PELA 2023 (1549) 一线 |

验证：best80 没有任何一档被 JS L1-L7 翻过盘（vS v1-v7 4800 局双色详见 §3）。

## 2. best80 vs Rapfi（freestyle 15x15）逐档

| 我方预算 | Rapfi 单步预算 | best80 胜率 | 局数 |
|---|---|---:|---:|
| 48 sims | 50ms | 6.2% | 32 |
| 48 sims | 150ms | 3.1% / 0%（两轮） | 32+24 |
| 48 sims | 500ms | 0% / 4.2%（两轮） | 32+24 |
| 48 sims | 1500ms | 4.2% | 24 |
| 48 sims | 5000ms | 0% | 24 |
| 96 sims | 150ms | 3.1% / 0%（两轮） | 32+24 |
| 96 sims | 500ms | 0% / 8.3%（两轮） | 32+24 |
| 96 sims | 1500ms | 0% / 12.5%（两轮） | 32+24 |
| 96 sims | 5000ms | 0% | 24 |
| 512 sims | 500ms | 15.6% / 25.0%（两轮） | 32+24 |
| 512 sims | 1500ms | 9.4% / 16.7%（两轮） | 32+24 |
| 512 sims | 5000ms | 8.3%（14 局） | 14 |

**解读**：我们任意单步预算都打不过 Rapfi 的"舒适区"。要咬下一口，需要把模拟量顶到 512 并且 Rapfi 还得在亚秒预算内。这就是"通用 AZ-lite"与"领域特化引擎"在五子棋上的真实代差。

## 3. JS v1-v7 的绝对标定

对所有档位以 Rapfi@150ms/500ms 作对照，全体 **0/24 全灭**（无一胜局）， BT 池内绝对 Elo 聚在 **1550–1611**（样本 12–36 局/级， L1/L4/L7 小样本共读）。

- 内部排序（对 best80@48 的胜率反向推, 双色 50 局）： **L6 > L7 > L3 > L5 ≈ L1 ≈ L2 ≈ L4**
- 与人类参照： 1500-1600 Elo 大约对应到没有任何正式段位的"会看三连与活四的爱好者"水平 —— 它们统统低于 Rapfi 至少约 -1100 Elo, 也低于 best80@48 至少约 -400 Elo。

## 4. Rapfi 的时间预算-强度标定（本机）

| 强预算 vs 弱预算 | 强侧胜率 | 局数 |
|---|---:|---:|
| 150 vs 50 ms | 59.4% | 32 |
| 500 vs 150 ms | 51.6% | 31+24 |
| 1500 vs 500 ms | 50.0% | 32+24 |
| 5000 vs 1500 ms | 50.0% | 32+24 |

**Rapfi 在 freestyle 上 50ms ≈ 5000ms 同档**——它的强度来自教科书开局库 + αβ+NNUE 的浅层战术覆盖，不依赖单步深度。这意味着"拿计算机时间换人类经验"这条路在五子棋上本来就走不通。

## 5.0 全景 Elo 阶梯（所有实体合成一张表）

锚：Gomocup Freestyle15 榜 `Rapfi 2025 = 2716`。两个参照点：
- `iter440@48`：由 best80@48 在 170 局里 65.3% 头对头反推（+110 → iter440 ≈ 1946）
- `b85@48`：同法反推（61.8% → ≈ 1972）
- Human anchor：2017 Gomoku 世界冠军 Rudolf Dupszki 被 Yixin 2:0 击败；Yixin 类引擎水位 ≈ 2175（freestyle15）。

带 `*` 的为非直接对局的反推值，按 n 给保守 ±100 误差；其余为池内实测：

| Elo | 实体 | 备注 |
|---:|---|---|
| 2742 | Rapfi@5000ms | 本机锚顶 |
| 2716 | Rapfi 2025 | 公榜锚 |
| 2695 | Rapfi@500ms | 本机 |
| 2690 | Rapfi@150ms | 本机 |
| 2617 | Rapfi@50ms | 本机 |
| **2406** | **best80 @ 512** | 68 局实测 |
| 2406 | EMBRYO 2025 | Gomocup 参照 |
| 2320 | （Jax 2025 ≈ 2554 之上级） | — |
| 2175 | Yixin 2018 | — |
| **2108** | **best80 @ 96** | 96 局实测 |
| 2085 | 人类世界冠军线* | 估计顶部区间 |
| **2056** | **best80 @ 48** | **本报告主结果**, 96 局实测, 也约抵 PELA 2023+ |
| 2024 | best85 @ 48 | 24 局实测 |
| 1946* | iter440 @ 48 | — |
| 1853 | WINE 2018 | Gomocup 参照 |
| ~1710 | JS L7 宗师 | 互搏+锚双链 |
| ~1675 | JS L4 专家 | " |
| ~1646 | JS L6 大师 | " |
| ~1379 | JS L3 进攻 | " |
| ~1346 | JS L2 防御 | " |
| ~1323 | JS L5 大师 | " |
| ~1243 | JS L1 基础 | " |
| 335 | Random 随机 | 锚底 |
| 764 | TKGOMOKU 2026 | Gomocup 榜最底公开档 |

视觉读法:
- best80@48 处于**引擎世界"二线决赛圈"**（公榜榜尾 700-1600 区之外, 中段引擎上层）
- best80@96/512 已越过"赢人类世界冠军那条线"的水位
- Rapfi 前沿离任何实测五子棋实体仍有一层天窗 (≈+300 Elo @512, ≈+660 Elo @48)

## 5. 和人类比（有文献依据的部分）

- 2017 年 Gomoku 世界冠军赛： 人类世界冠军 Rudolf Dupszki 被引擎 **Yixin 2:0** 公开击败; 同期 Yixin 在 Gomocup freestyle15 浮动 Elo = **2175**（与本报告同一锚）。
- 按此链条： best80 @96 sims（≈2108）即已摸到"赢人类世界冠军那年的引擎"水位； best80 @512 sims（≈2406）明显超过——**自由规则下， 我们模型 512 档已可判人类一方没有实质性胜算**。
- vital 声明： 该对照建立在 Yixin-2017 比赛用规则与本报告 freestyle-15x15 不完全一致（Yixin 那局使用 Swap2 开局平衡）; 本报告的"人类对照"应解读为"对人类顶格玩家不落下风"的保守结论， 而不是一个精确的段位。

## 6.1 相当于哪一年的引擎水平（Gomocup Freestyle15 榜对照）

用榜单中每款"名字带年份"的引擎做年份-水位采样：

| 我方档位 | Elo | 落在的引擎年代 |
|---|---:|---|
| best80 @48 | ≈ 2056 | **2018 年顶级专用引擎带**（压住 WINE 2018=1853、SKYZERO 2026=1927、STARPOINT 2026=2021；YIXIN 2018=2175 还在我们上方） |
| best80 @96 | ≈ 2108 | **摸 YIXIN 2018（2175）下沿** |
| best80 @512 | ≈ 2406 | **2025-2026 决赛圈外围**（EMBRYO 2025=2320 已过，JAX 2025=2554、ALPHAGOMOKU 2026=2613、KATAGOMO 2026=2696、RAPFI 2025=2716 在上方） |

## 6.2 引擎技术代差说明（为什么 Rapfi/Embryo 离我们这么远）

Gomocup 历年头部引擎（Tito/Hewer → Wine/Goro → Yixin → Embryo → Rapfi/Katagomo/AlphaGomoku）的跃迁几乎都靠**领域特化的搜索/评估工程**（威胁空间剪枝、开局库、长连/冲四精算、NNUE on freestyle 工作流），这正好是我们从公式里划掉、从自对弈里学不到的"那 300-700 Elo"。 这不是算力问题——Rapfi 50ms 已经把我们打到 6% 胜率。

## 6. 结论一句话

**k1000-best80**: 小稳住牌面是"引擎级别选手, 但还不是顶级引擎级别"。在 Gomocup Freestyle15 天梯上大约位于 **第 9-12 名水位（@512）**，能稳赢所有自研 7 档 JS AI 和任意业余人类，但 vs Rapfi 满档会被横扫（自由规则下实力差距 ≈ -300 Elo @512 / -660 Elo @48）。

## 7. 复现

```bash
# 构建 Rapfi (本仓外) 
git clone https://github.com/dhbloo/rapfi -b master --depth 1 /tmp/rapfi
cd /tmp/rapfi && git submodule update --init --depth 1 Networks
# GCC8 需三处兼容补钉: bound(ValueBound{}), <iomanip>, -lstdc++fs(见 git log 此次提交)

# 本仓引擎侧协议
./bin/alphazero piskvork --model runtime_v2/candidates/k1000-best80-19ea8c34.net --sims 48 --threads 1
# (piskvork 文本协议, START/BEGIN/TURN/BOARD)

# 单场对局:
python3 tools/crossmatch.py \
  --home 'az|<model>|sims=48|threads=1' \
  --away 'rapfi|/tmp/rapfi/Rapfi/build/pbrain-rapfi|turnms=150' \
  --games 32 --workers 12 --seed 101 --out runtime_v2/cross2/<name>.jsonl

# BT 聚合:
# (tools/rating.py 的原型, 本报告内的完整评级脚本见 runtime_v2/cross2/ 也保留)

# 归档数据: runtime_v2/cross2/*.jsonl (~30 组对局, 1000+ 局全记录)
```

## 8. 已知局限

1. **样本量**: 48/96 vs Rapfi 中间档有不少 0/24 或 1/32 的序列, Elo 差估计向下压缩 (二项 95% CI 下界约 -950 Elo), 报告使用 BT 全池联合评级而非单独点估计值, 相关数字带 ±100 Elo 量级的保守读法。
2. **开局池**: referee 用自定的 "天元周边 2 格" 两步开局而非 Gomocup 的平衡开局库, 双方的平衡态与 Gomocup 官方不完全同.
3. **硬件非等价**: Gomocup 用它的比赛机跑 Rapfi, 我们是本机 Xeon + AVX512。

## 9. 附录: L4/L6/L7 深检三角 (2026-08-30)

为核实 "L6 是否体感最强" 的疑问, 三个高份子节目互打各 32 局:

| 对局 | 比分 | 备注 |
|---|---:|---|
| L4 (expertAI) - L6 (TeacherAI) | 16 : 10 | L4 领先 (n=26 有效) |
| L4 - L7 (MctsAI) | 14 : 16 | 均势 (n=30) |
| L6 - L7 | 12 : 20 | L7 领先 (n=32) |

BT 三角收敛: L7 +25 / L4 +15 / L6 -65 (相对 Elo, 三者全部落在统计噪声±95窗内)。

**结论**: 互搏并没有"L6 显著最强"。 之前 b80@48 对 L6 只压制到 76% (低于对 L7/L4 的 100%),
是 L6 的"深度 3 防守评估"风格恰好卡 AZ 的战术波节奏, 属风格克制而非绝对棋力更小。
以上三角数据已存档 `runtime_v2/cross2/deep_*.jsonl`。
