# ExpVid Experiments — Visual Note-Taking for Scientific Video Understanding

实验性 repo：在 [ExpVid benchmark](https://huggingface.co/datasets/OpenGVLab/ExpVid) 上探索**绕开 ASR 答案泄漏**的视觉推理方法。

原始 benchmark 介绍见 [README_ExpVid_Paper.md](./README_ExpVid_Paper.md)。

---

## 🎯 我们做的事

ExpVid 论文设计了一套科学实验视频 benchmark，但我们发现：

> **L1 任务的标签是从 ASR 文本里抽取的实体（材料/工具/数量/动作名）。**  
> 把 ASR 喂给模型答题，相当于直接给模型看答案。

为了在**不依赖 ASR** 的前提下做真正的多模态视觉推理评测，我们：

1. **复现 video-only baseline (C1)** — Qwen2.5-VL-7B 直接看视频
2. **对照实验 ASR-only (C2) 和 video+ASR (C3)** — 量化 ASR 泄漏到底有多严重
3. **提出 Two-stage prompting 方法**（v1 / v2）— 让 VLM 先生成结构化笔记，再用笔记答题，全程不喂 ASR

---

## 📊 主要发现

### 1. ASR 泄漏在 L1 是 50pp 级别的

| Task | C1 (video) | C3 (video+ASR) | **C2 (ASR-only)** |
|---|---|---|---|
| materials | 34.3% | 89.2% | **91.5%** |
| tools | 35.7% | 79.5% | **87.8%** |
| operation | 64.4% | 96.7% | **97.2%** |
| quantity | 48.8% | 96.0% | **96.2%** |
| **L1 平均** | **45.8%** | **90.3%** | **93.2%** |

**关键观察：** 纯 ASR（没视频）比 video+ASR 还高 3pp。说明 L1 评测本质是 ASR 阅读理解，视觉信号在端到端 VLM 里反而是噪声。

### 2. Two-stage 笔记法在 video_verification 上 +13.2pp

| Task | C1 baseline | Two-stage v2 (task-aware notes) | Δ |
|---|---|---|---|
| materials | 34.3% | **38.8%** | +4.5pp ✅ |
| tools | 35.7% | 32.3% | -3.4pp |
| operation | 64.4% | 52.0% | -12.4pp |
| quantity | 48.8% | 32.0% | -16.8pp |
| sequence_ordering | 54.4% | 56.5% | +2.2pp |
| step_prediction | 2.1% | 2.6% | +0.5pp |
| **video_verification** | 17.4% | **30.5%** | **+13.2pp** ✅ |

**关键观察**：  
- **video_verification** 是 ASR 不能帮的"诚实"视觉任务（要找视频里没出现的步骤），两阶段笔记法让 7B 模型从 17% 涨到 30%
- L1 的 operation/quantity 上下降，说明 task-aware 笔记**信息聚焦过头**反而丢了 context
- 后续方向：用 **counterfactual ablation** 自动学"该记什么不该记什么"

### 3. step_prediction 是真正的视觉硬骨头

论文里所有 < 7B 开源模型都 < 5%（Qwen2.5-VL-72B 仅 0.3%），我们的所有方法也都接近随机。ASR 没法帮（要预测未来），真正考验视觉时序理解能力。

---

## 🛠️ 评估脚本

| 脚本 | Condition | 说明 |
|---|---|---|
| `evaluate.py` | C1 (video-only) | 复现 ExpVid 论文标准 video-only 评测 |
| `evaluate_asr.py` | C3 (video + ASR) | 把 ASR 也喂给 VLM |
| `evaluate_asr_only.py` | C2 (ASR-only) | 纯文本评测，量化泄漏上限 |
| `evaluate_twostage.py` | Two-stage v1 | 通用 prompt 笔记 + 答题 |
| `evaluate_twostage_v2.py` | Two-stage v2 | **Task-aware** prompt 笔记 + 答题（推荐） |

### 快速运行

```bash
# 安装依赖
pip install transformers torch huggingface_hub av qwen-vl-utils

# 跑 v2 Two-stage 在 L1
python evaluate_twostage_v2.py --task all_level1 --output results_twostage_v2

# 跑特定任务（带 limit 快速验证）
python evaluate_twostage_v2.py --task video_verification --limit 100

# resume（跳过已完成的任务）
python evaluate_twostage_v2.py --task all --output results_twostage_v2 --resume
```

### 硬件需求

| Level | Min GPU | 建议 |
|---|---|---|
| L1 (short clips, 32 frames) | 24GB | RTX 3090/4090 |
| L2 (medium clips, 32 frames) | 24GB | 同上 |
| L3 (long videos, 8-12 frames patched) | 24GB | C3 服务器跑过 |
| L3 (32 frames full) | 48GB+ | RTX A6000 / H100 |

---

## 📁 项目结构

```
expvid/
├── evaluate.py                    # C1: video-only baseline
├── evaluate_asr.py                # C3: video + ASR
├── evaluate_asr_only.py           # C2: ASR-only
├── evaluate_twostage.py           # Two-stage v1 (generic note)
├── evaluate_twostage_v2.py        # Two-stage v2 (task-aware) ⭐ 推荐
├── note_eval.py                   # ExpVid 原作者的评估代码
├── transcribe.py                  # ASR 转录（如需）
├── README_ExpVid_Paper.md         # ExpVid 原论文 README
├── PROGRESS.md                    # 实验进度日志
├── PROJECT_CONTEXT.md             # 详细项目背景（给跨机协作）
└── results/                       # 评测结果 JSON（小文件保留）
```

---

## 🚀 下一步研究方向

### Counterfactual SFT for Note-Taking

**思路**：
- Stage 1 让 GPT-4o 生成 over-complete 视觉笔记（多字段 JSON）
- 对每个字段做 ablation：移除后下游 LLM 还能答对吗？
- 标记 ESSENTIAL / REDUNDANT / HARMFUL fields
- 训练 Qwen2.5-VL-7B 生成"causally minimal but sufficient"笔记

**训练数据**（cross-benchmark，避免污染 ExpVid）：
- SciVideoBench (231 视频，与 ExpVid 仅 10 个重叠)
- FineBio (生物实验，与 ExpVid 同领域)
- EPIC-KITCHENS (程序性视频，跨域 transfer)
- HowTo100M science slice

**测试**：ExpVid full 390（dedup 后 380），zero-shot transfer。

详见 [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md)。

---

## 🌟 引用原 benchmark

```bibtex
@article{xu2025expvid,
  title={ExpVid: A Benchmark for Experiment Video Understanding \& Reasoning},
  author={Xu, Yicheng and Wu, Yue and Yu, Jiashuo and others},
  journal={arXiv preprint arXiv:2510.11606},
  year={2025}
}
```

