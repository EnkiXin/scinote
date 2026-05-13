# ExpVid Experiments — Progress Log

更新时间：2026-05-13  
模型：Qwen2.5-VL-7B-Instruct  
硬件：RTX 4090 48GB (C1) + RTX 4090 24GB (C3) on featurize.cn

---

## 实验时间线

| 日期 | 事件 |
|---|---|
| 2026-05-11 | 启动 C1 (video-only) 评测 |
| 2026-05-12 | 启动 C3 (video+ASR) 和 C2 (ASR-only) 评测 |
| 2026-05-12 | C2 完成 L1，发现 ASR 泄漏（C2 > C3） |
| 2026-05-12 23:00 | 启动 Two-stage v1 实验 |
| 2026-05-13 00:00 | 升级到 Two-stage v2（task-aware prompt） |
| 2026-05-13 03:00 | C3 跑 L3 OOM，patched 成 8 帧 |
| 2026-05-13 13:00 | L3 在 C1 和 C3 同时跑（进行中） |

---

## C1: video-only baseline (Qwen2.5-VL-7B)

### L1（已完成）

| Task | Accuracy | n_valid | OOM 数 |
|---|---|---|---|
| materials | **34.28%** | 1199 | 67 |
| tools | **35.66%** | 1091 | 39 |
| operation | **64.43%** | 908 | 30 |
| quantity | **48.78%** | 658 | 43 |
| **平均** | **45.79%** | 3856 | 179 |

### L2（已完成）

| Task | Accuracy |
|---|---|
| sequence_generation | 45.08% (F1, 论文用 Jaccard ~21%) |
| sequence_ordering | 54.35% |
| step_prediction | **2.14%** ⚠️ 极低（所有模型都难） |
| video_verification | **17.38%** ⚠️ 接近随机 |

### L3（跑中）

- experimental_conclusion: 进行中
- scientific_discovery: 排队

---

## C3: video + ASR

### L1（已完成）

| Task | Accuracy | vs C1 |
|---|---|---|
| materials | 89.16% | +54.9pp |
| tools | 79.50% | +43.8pp |
| operation | 96.69% | +32.3pp |
| quantity | 96.0% | +47.2pp |
| **平均** | **90.34%** | **+44.6pp** |

→ ASR 加上后 L1 几乎 ceiling，强烈暗示 **ASR 泄漏**

---

## C2: ASR-only（没视频）

### L1（已完成）

| Task | C2 (ASR only) | vs C3 |
|---|---|---|
| materials | **91.47%** | +2.3pp |
| tools | **87.79%** | +8.3pp |
| operation | **97.23%** | +0.5pp |
| quantity | **96.15%** | +0.2pp |
| **平均** | **93.16%** | **+2.8pp** |

🚨 **关键发现**：**纯 ASR（没视频）比 video+ASR 还高**

意味着：
1. L1 任务的答案完全在 ASR 文本里
2. 视觉信号在 ASR 已知时反而是噪声
3. ExpVid L1 的设计存在严重答案泄漏（标注流程从 ASR 抽取实体）

→ 论文里 Table 2 没考虑这个，所有 video-only 评测都是在测"对实体的视觉确认"，不是真正的视觉理解

---

## Two-stage Prompt 实验

### v1（generic note，2026-05-12 23:00）

让 Qwen2.5-VL 生成通用观察笔记：
```
{materials_visible, tools_in_use, actions, quantities, setting, fine_details}
```
然后用笔记答题。

| Task | v1 | C1 baseline |
|---|---|---|
| materials | 32.7% | 34.3% |

**结果失败**：笔记过于通用（"brownish substance"），无法 disambiguate 具体材料名。

### v2（task-aware note，2026-05-13 00:00）

针对每个任务定制 prompt：
- materials → 强调具体科学名称、可读标签、容器形态
- tools → 强调具体仪器名、distinguishing features、品牌型号
- quantity → 强调显示屏数字、刻度、计数
- operation → 强调精确动作动词、目标对象
- L2 (sequence/step/verify) → 强调步骤序列、当前状态
- L3 (analysis/discovery) → 全面长视频观察

### v2 结果（vs C1 baseline）

| Task | v2 | C1 | Δ | n |
|---|---|---|---|---|
| materials | 38.78% | 34.28% | **+4.5pp** ✅ | 49 (小测) |
| tools | 32.33% | 35.66% | -3.3pp | 1126 |
| operation | 52.03% | 64.43% | -12.4pp ❌ | 936 |
| quantity | 31.95% | 48.78% | -16.8pp ❌ | 698 |
| sequence_generation | 21.29% | 45.08% | -23.8pp | 70 (OOM 多) |
| sequence_ordering | 56.52% | 54.35% | +2.2pp | 69 (OOM 多) |
| step_prediction | 2.61% | 2.14% | +0.5pp | 230 |
| **video_verification** | **30.54%** | **17.38%** | **+13.2pp** 🏆 | 239 |
| experimental_conclusion | 跑中 | 跑中 | — | — |
| scientific_discovery | 排队 | 排队 | — | — |

### v2 关键洞察

**👍 v2 真正赢的地方：video_verification +13.2pp**
- 这是 ASR 不能直接帮的任务（要找出哪个步骤**没出现**）
- 任务感知 prompt 让模型聚焦"哪些步骤被执行了"
- 两阶段拆解 perception → reasoning 显示出价值

**👎 v2 输的地方：operation / quantity**
- 笔记 prompt 太聚焦反而丢了 context
- 比如 quantity 要"count things"，笔记只记数字丢了上下文
- → 暗示：固定 task-aware prompt 不够，**需要训练学到该记什么**

---

## 错误处理与已知问题

### OOM 问题

- L1/L2 在 24GB C3 服务器上偶发 OOM（特定长视频）
- L3 在 24GB GPU 上完全跑不动（每个 sample OOM）→ patched 成 8 帧

### 解决方案

```python
# C3 (24GB) 上的 patched 参数
if task_type == "fitb":  # L3
    fps, max_frames = 0.2, 8     # 8 帧，覆盖 ~40s 间隔
else:
    fps, max_frames = 1.0, 32    # L1/L2 用 32 帧

max_pixels = 360 * 420  # 别压更低，会触发 min_pixels 错误
```

---

## 下一步

### 短期（等 L3 跑完）

- [ ] 收集 C1 video-only 完整 L3 baseline
- [ ] 收集 C2/C3 的 L3 数据
- [ ] 收集 v2 Two-stage 的 L3 数据
- [ ] 完整三向对比 C1 vs v2 vs C3，看 L3 上 ASR 泄漏程度（预期：L3 较低，因为答案来自 paper 不在 ASR 里）

### 中期（训练）

- [ ] 申请 SciVideoBench 数据访问
- [ ] 申请 FineBio 数据访问
- [ ] 下载 EPIC-KITCHENS 标注
- [ ] 写 counterfactual SFT 数据生成 pipeline
- [ ] LoRA 微调 Qwen2.5-VL-7B 学"做笔记"

### 长期

- [ ] Cross-benchmark transfer 验证（SciVideoBench/EPIC 训 → ExpVid 测）
- [ ] Domain transfer 矩阵（biology / chemistry / medical / cooking 互换）
- [ ] RL/DPO on top of SFT（counterfactual reward）

---

## 当前实验状态（2026-05-13 13:00 JST）

| 实验 | 状态 |
|---|---|
| C1 L1+L2 | ✅ 全部完成 |
| C1 L3 | 🔄 进行中 (patched) |
| C2 L1 | ✅ 完成 |
| C3 L1 | ✅ 完成 |
| Two-stage v2 L1+L2 | ✅ 8/10 完成（有 OOM 噪声） |
| Two-stage v2 L3 | 🔄 进行中 (8 帧 patched) |
