# Project Context — ExpVid & CRAG Experiments

**生成时间**：2026-05-13  
**用途**：让在新电脑上启动的 Claude Code 快速接上下文。  
**用法**：新机器上 Claude 启动后，第一句说 "请先读 PROJECT_CONTEXT.md 了解我们之前的工作"。

---

## 1. 两条并行的研究线

### 线 A: CRAG (movie recommendation) — 阶段性结论：**负面**

**项目位置**：`/Users/enki/crag2/`（Mac 本地）

**目标**：用 Knowledge Graph (KG) 增强 CRAG 论文的 4-step movie recommendation pipeline。

**论文 baseline**（Zhu et al., CIKM 2025）：
- R@10 = 0.1715, NDCG@10 = 0.1065 on REDDIT-V2 CIKM split

**做了什么**：
1. **Case analysis** (`analyze_kgsoft_peritem.py`): 发现 KGsoft 的提升 95.7% 来自 Step4 reranking，Step1 候选生成贡献 < 5%
2. **协同图构建** (`build_collab_graph.py`): 41K train conversations → 8,982 movies / 220K edges
3. **PPR 异构图** (`build_ppr_graph.py`): 159,839 nodes (movies + persons + genres + keywords + collections), 1.1M edges, 3.1MB sparse npz
4. **Query-driven retrieval** (`evaluate_kg_querydriven.py`): 完全弃用 CF，用 PPR 检索候选
5. **多变体实验** (`evaluate_kg_variants.py`):
   - V1: PPR α=0.3, entity_scale=0.5
   - V2: V1 + Step4 prompt 注入 KG paths
   - V3: V2 + Step3 prompt 也注入 KG paths
   - V4: V3 + LLM-based entity extraction

**结果**（aligned eval 与论文同 protocol）：

| Method | n | Step3 R@10 | Step4 R@10 | NDCG@10 |
|---|---|---|---|---|
| Paper CRAG | 4564 | — | 0.1715 | 0.1065 |
| Our CRAG Baseline | 4564 | 0.1695 | 0.1726 | 0.1035 |
| KGsoft (Step1 KG only) | 4564 | 0.1703 | 0.1677 | 0.1021 |
| V2 Full (Step4 KG) | 4564 | 0.1595 | 0.1576 | 0.0955 |
| V3 Full (Step3+4 KG) | 4564 | 0.1300 | 0.1505 | 0.0888 |

**关键发现**：
- V3 在 200 样本上 R@10=0.1793 看似超过论文，**全量验证回归 0.1505**，是采样运气
- PPR 检索 < CF (BBsim) 本身
- Step3 注入 KG paths 反而干扰 LLM 生成
- **结论：这条线收手**

---

### 线 B: ExpVid (scientific video understanding) — 进行中，有 promising 信号

**项目位置**：`/Users/enki/expvid/`（Mac 本地）

**目标**：在 ExpVid benchmark 上评测 video-only 与不同 condition 的差异。

**Benchmark 信息**：
- **ExpVid** (Xu et al., ICLR 2026): 390 视频，7,800 QA，3 级任务（L1 perception, L2 procedural, L3 reasoning）
- 数据来源：JoVE 期刊视频（科学实验录像）
- HF: `OpenGVLab/ExpVid`

**做了什么**：

#### 三个 condition 在 Featurize 服务器上跑：

| Condition | 视频 | ASR | 服务器 | 状态 |
|---|---|---|---|---|
| C1 (video-only) | ✅ | ❌ | C1 (port 26795, 48GB GPU) | L1+L2 完成，L3 跑中 |
| C2 (ASR-only) | ❌ | ✅ | C3 (port 44002) | L1 全部完成 |
| C3 (video+ASR) | ✅ | ✅ | C3 (port 44002) | L1 全部完成 |

#### L1 结果（已完成）：

| Task | C1 (video) | C3 (video+ASR) | C2 (ASR only) |
|---|---|---|---|
| materials | 34.3% | 89.2% | **91.5%** |
| tools | 35.7% | 79.5% | **87.8%** |
| operation | 64.4% | 96.7% | **97.2%** |
| quantity | 48.8% | 96.0% | **96.2%** |

**惊人发现**：**C2 (纯 ASR 文本，没视频) 比 C3 (video+ASR) 还高！**

**原因分析**（详见 ExpVid 论文附录 G）：
ExpVid 的 L1 标注流程是从 ASR 句子里抽取实体（materials/tools 名称），所以**问题答案就在 ASR 里**。
- L1 是 ASR 阅读理解，不是真正的多模态理解
- L2 部分任务（特别是 step_prediction）是真正的视觉任务
- L3 答案来自 paper（不在 ASR 里），是干净的多模态推理评测

#### Two-stage prompt 实验（v1 → v2）：

**思路**：Stage 1 让 VLM 看视频生成 structured note，Stage 2 用 note 答题（不喂 ASR）。
**目的**：在不靠 ASR 的前提下，看视频能否逼近 C2/C3。

| Variant | materials | tools | operation | quantity | video_verification |
|---|---|---|---|---|---|
| C1 baseline | 34.3% | 35.7% | 64.4% | 48.8% | 17.4% |
| v1 (generic note) | 32.7% | — | — | — | — |
| **v2 (task-aware note)** | **38.8%** | 32.3% | 52.0% | 32.0% | **30.5%** |

**v2 关键发现**：
- ✅ **video_verification +13.2pp**（17→30%）是真正的视觉任务，两阶段笔记真起作用了
- ✅ materials +4.5pp
- ❌ tools/operation/quantity 下降（笔记 prompt 太聚焦丢了信息）
- L3 还没跑（C3 服务器 24GB 显存不够，patched 成 8 帧后正在重跑）

#### 当前在跑的任务（2026-05-13 12:30 JST）：

1. **C3 服务器**: Two-stage v2 跑 L3 (experimental_conclusion + scientific_discovery)，8 帧 patched，正常进行
2. **C1 服务器**: video-only L3，8 帧 patched 后正常进行
3. 预计 1.5-2 小时跑完

---

## 2. 下一步计划

### 已经讨论清楚的方向

#### Counterfactual SFT for Note-Taking
- 用 GPT-4o 生成 over-complete note → 字段级 ablation → 训练 model 生成 causally minimal note
- 用 reference: ExpVid 论文 Appendix G 公开了所有标注 prompt（可以直接用 DeepSeek-R1 / GPT-4o 生成训练数据）

#### 训练数据集（用户已确认想要 cross-benchmark transfer，不污染测试）
1. **SciVideoBench** (231 视频, 1000 MCQ, 与 ExpVid 仅 10 个重叠，已验证) — 申请访问 + HuggingFace
2. **FineBio** (32 ppl × 7 protocol, 14.5h, biology) — 申请访问 + GitHub aistairc/FineBio
3. **EPIC-KITCHENS** (100h kitchen, 90K segments) — 公开下载
4. **HowTo100M** (136M YouTube, lab/science slice) — 公开下载

#### 硬件
用户已租了 48GB 4090，跑 Qwen2.5-VL-7B LoRA 完全够（QLoRA 可上 32B）。

---

## 3. 用户偏好（已记到 memory）

- **自主决策**：长链任务 / 夜间任务全部默认 yes，不反复问确认（详见 `~/.claude/projects/-Users-enki-expvid/memory/feedback_autonomous.md`）
- **使用语言**：中文沟通
- **关注点**：方法 novelty + benchmark 数字 + 与论文对比

---

## 4. 关键代码文件 (Mac 上)

```
/Users/enki/crag2/
├── analyze_kgsoft_peritem.py    # case analysis
├── build_collab_graph.py         # collaborative graph
├── build_ppr_graph.py            # PPR heterogeneous graph
├── evaluate_kg_querydriven.py    # query-driven (PPR) eval
├── evaluate_kg_variants.py       # multi-variant eval (V1-V5)
├── eval_aligned.py               # paper-aligned scoring
└── libs/kg_paths.py              # MovieKG class

/Users/enki/expvid/
├── evaluate.py                   # C1 video-only
├── evaluate_asr.py               # C3 video+ASR
├── evaluate_asr_only.py          # C2 ASR-only
├── evaluate_twostage.py          # v1 generic note
├── evaluate_twostage_v2.py       # v2 task-aware note
└── results_twostage_v2/          # latest results
```

---

## 5. 给新 Claude 的建议

如果用户接下来想：

- **继续 ExpVid two-stage 实验**：可以读 `/Users/enki/expvid/evaluate_twostage_v2.py` 改进 prompts
- **开始 Counterfactual SFT 训练**：等 ExpVid L3 数据出来后开始，先调研 SciVideoBench/FineBio 申请进度
- **看 CRAG 最终结果**：直接读 `/Users/enki/crag2/eval_aligned.py` 跑评估
- **想用 RL 训练 noter**：参考 Rank-GRPO 论文 (arXiv:2510.20150)，但需要先 SFT warm-start

**对新机器（Linux server `ci-l-2x6nxb4`）**：
- 该机器只是用户的另一台工作机
- 主项目在 Mac 上
- 但可以在新机器上：写代码、调研、planning，不需要重复 Mac 上的实验

---

## 6. 实验数字快速 reference

```
ExpVid benchmark Qwen2.5-VL-7B baseline (论文 Table 2):
  L1 avg: 42.6%, L2 avg: 24.6%, L3 avg: 23.3%

我们 C1 video-only:
  L1 avg: ~44%, L2 avg: ~29% (Generation 用 F1 而非 Jaccard)
  L3: 跑中

我们 C2/C3 ASR augmented (L1):
  90%+ 全部，确认 ASR leakage

Two-stage v2 (task-aware prompt, no ASR):
  video_verification: 17% → 30% (+13pp, 真正的视觉任务赢了)
  materials: 34% → 39% (+4.5pp)
  其他 L1/L2 任务 mixed
```

完。
