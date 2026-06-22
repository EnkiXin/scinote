# REPORT — 统一公平框架终审结果(唯一结果汇总)

**更新**:2026-06-12 · 代码:commit `ecfab199` 起(6 处 harness bug 已修,见 [HARNESS_AUDIT_2026-06-11.md](./HARNESS_AUDIT_2026-06-11.md))
**契约**:`scripts/unified_harness.py` — 同 uid 题集 / 每题一次 32 帧抽取 / 同 BUILDERS / greedy / 修复后解析打分 / 条件单变量 / rep2 独立进程噪声底
**聚合**:`python -m tools.aggregate_unified`(配对 sign test + 跨尺度 interaction)→ `results_unified/aggregate_summary.json`

---

## 1. 一句话结论

**在干净的测量口径下,没有任何文本干预(注入笔记或自生成推理)在任何尺度、任何基准上显著提升;
所有真实效应都是伤害,且伤害位置由"模型×基准"暴露的失败模式决定**——7B 的脆弱程序性作答被
自己的推理带偏(ExpVid CoT −3.39pp),72B 的先验压倒被任何自由文本喂大(SciVB 注入与 CoT
双双 −8.72pp)。旧的"注入帮弱模型、伤强模型"规模反转叙事**两半都被修正**:"伤强"只在 SciVB
成立,"帮弱"在公平口径下不显著。

## 2. 主矩阵(全量,c0/cot/c1,rep2 噪声底)

### ExpVid L2(程序理解,n=608:seq_ordering+seqgen+video_verification+step_prediction)

| | 7B | 72B |
|---|---|---|
| c0 | 0.2846 | 0.3648 |
| cot | 0.2495(**−3.51pp,p≈0,真伤害**) | 0.3835(+1.86pp,p=0.45,零) |
| c1 注入 | 0.3032(+1.86pp,p=0.87,零) | 0.3716(+0.67pp,p=0.63,零) |
| 噪声底 | +0.17pp,9/608 翻转 | −0.09pp,9/608 翻转 |

### ExpVid L3(科学推理,n=137:experimental_conclusion+scientific_discovery)

| | 7B | 72B |
|---|---|---|
| c0 | 0.1864 | 0.2818 |
| cot | 0.1580(**−2.84pp,p=0.003,真伤害**) | 0.2531(−2.88pp,p=0.28,零/方向负) |
| c1 注入 | 0.1695(−1.69pp,p=0.08,边缘) | 0.2616(−2.02pp,p=0.18,零) |
| 噪声底 | −0.13pp,4/137 翻转 | −0.38pp,3/137 翻转 |

> **拆 L2/L3 暴露的对冲**(原 n=745 合并表掩盖):72B 的 CoT 在合并口径下是"零(+0.99pp)",实为 **L2 +1.86pp(外化计算帮程序任务)与 L3 −2.88pp(长视频推理被带偏)对冲**;c1 注入在 72B 上同样 L2(+0.67)/L3(−2.02)反向。7B 则 L2(−3.51)/L3(−2.84)两级都被 CoT 真伤害。L3 绝对分对两模型都显著低于 L2(7B 0.186 vs 0.285、72B 0.282 vs 0.365),与论文"L3 最难"一致。加权回合并:7B c0 608×0.2846+137×0.1864→0.2665、72B→0.3495,与原合并值一致。

### SciVideoBench(n=218)

| | 7B | 72B |
|---|---|---|
| c0 | 0.2569 | 0.4174 |
| cot | 0.2523(−0.46pp,p=1.0,零) | 0.3303(**−8.72pp,p=0.020,真**) |
| c1 注入 | 0.2569(±0.00pp,p=1.0,零) | 0.3303(**−8.72pp,p=0.00031,真**) |
| 噪声底 | 0.00pp,0 翻转 | 0.00pp,0 翻转 |

### 跨尺度 interaction(7B 效应 − 72B 效应,pp)

| | ExpVid | SciVB |
|---|---|---|
| cot | −4.38(7B 伤得更重) | +8.26(72B 伤得更重) |
| c1 | +1.03(≈无) | +8.72(72B 伤得更重) |

### 72B ExpVid 分任务(CoT 的任务结构)

| task | n | c0 | cot | c1 |
|---|---|---|---|---|
| steppred | 145 | 0.041 | **0.124**(×3) | 0.069 |
| seqgen | 161 | 0.446 | **0.498** | 0.465 |
| mc | 302 | 0.477 | 0.447 | 0.467 |
| fitb | 137 | 0.282 | 0.253 | 0.262 |

### 7B ExpVid 分任务(CoT 伤害的来源)

| task | n | c0 | cot | c1 |
|---|---|---|---|---|
| seqgen | 161 | 0.423 | **0.308**(主要伤害源:推理后弃答/带偏) | 0.412 |
| mc | 302 | 0.348 | 0.311 | 0.371 |
| fitb | 137 | 0.186 | 0.158 | 0.169 |
| steppred | 145 | 0.000 | 0.055 | 0.041 |

## 2.5 全模型对比

### A. 历史多模型 sweep:笔记注入效应(C1−C0,修复解析重打分,positional 配对 sign test)

> 口径:旧 sweep harness(布线经审计为净:单变量、note 缓存零错位),解析/打分按修复后代码重算
> (`tools/rescore_fixed.py`)。7B 行来自统一公平框架(§2),口径略不同,仅作方向参考。

| 模型 | ExpVid L2/3 (n=745) | ExpVid L1 (n=4035) | SciVB (n=218) |
|---|---|---|---|
| Qwen2.5-VL-3B | +1.83(p=.24,零) | +0.64(p=.24,零) | +0.92(p=.69,零) |
| MiMo-VL-7B | +0.26(p=.68,零) | **+1.78(p=.013,正)** | −1.38(p=.70,零) |
| InternVL3-8B | +0.92(p=.41,零) | −1.26(p=.059,边缘) | −1.38(p=.68,零) |
| Qwen2.5-VL-7B* | +1.21(p=.49,零) | — | 0.00(p=1.0,零) |
| Qwen2.5-VL-72B | −0.34(p=.24,零) | **−4.19(p≈0,负)** | **−10.55(p=2e-5,负)**(公平口径 −8.72,p=3e-4) |

**读法**:12 个历史格子里只有 3 个显著——MiMo-L1 微正、72B-L1 与 72B-SciVB 真负。
**"注入帮小模型"在修复口径下基本是噪声**(3B 的 +1.83 不显著);伤害只属于最强模型,
且只在两处。配合 §2 的 interaction:效应由"模型×基准"的失败模式决定,不是尺度单调。

### B. 各模型 C0 基线水平(修复解析,绝对值供选型参考)

| 模型 | ExpVid L2/3 | ExpVid L1 | SciVB |
|---|---|---|---|
| Qwen2.5-VL-3B | 0.218 | 0.393 | 0.211 |
| InternVL3-8B | 0.253 | 0.439 | 0.294 |
| Qwen2.5-VL-7B* | 0.267 | — | 0.257 |
| MiMo-VL-7B | 0.286 | 0.437 | 0.252 |
| Qwen2.5-VL-72B | 0.351 | 0.517 | 0.417 |

### C. 论文公开数字(外部口径:帧数/打分协议不同,**不可与本地直接比**)

| 模型 | ExpVid L2 | ExpVid L3 | SciVB overall |
|---|---|---|---|
| GPT-5 | **57.5** | **56.4** | — |
| Gemini-2.5-Pro | 53.8 | 47.9 | **64.3** |
| Gemini-2.5-Flash | ~50 | 44.1 | 46.4 |
| InternVL3-78B(最强开源) | 41.9 | 37.7 | 38.8 |
| Qwen2.5-VL-72B(论文口径) | 35.9 | 30.6 | 20.3(768 帧) |
| Qwen2.5-VL-7B | 24.6 | 23.3 | — |
| GPT-4o | — | — | 24.9 |
| 人类(非专家/研究生) | 42.1 (L2) | — | 17.4 |
| 随机 | — | — | 10.0 |

**读法**:本地 72B C0(ExpVid 0.35)与论文口径一致;SciVB 本地 41.7 vs 论文 20.3 不可比
(帧预算/抽取协议差异,见审计)。与 SOTA 的差距(L2/3 约 −21~−26pp)主要是原生 long-CoT
推理能力(CoT 给 Gemini-1.5-Pro +21.1pp,给 Qwen 是零或负),prompt 管线闭合不了——
论文不卖 SOTA,卖机制定界。

## 3. 五个机制级发现(论文资产)

1. **自由文本伤害通道无关**(72B SciVB):自生成推理与外部注入笔记**逐点同为 −8.72pp**——
   伤害不来自文本的来源,来自文本的存在(prior-override 通道)。
2. **CoT 不是均匀的好/坏,是任务结构化的**(72B ExpVid):外化计算受益的任务(steppred ×3、
   seqgen +5.2pp)对冲近平局判别受损的任务(mc −3.0,"两项相同→选 A" 8v0 翻错 p=0.008),净零。
3. **CoT 伤害随尺度反向迁移**:7B 在 ExpVid 被 CoT 伤(−3.39,p≈0:推理诱发弃答/带偏,
   seqgen −11.5pp),72B 免疫;SciVB 反过来。"reason more"对两个尺度都不是免费的。
4. **"注入帮弱模型"不复现**:7B c1 在两个基准上均不显著(+1.21 p=0.49 / 0.00 p=1.0)。
   历史 +3.12pp(p=.0019)是旧口径的产物;公平口径下注入对 7B 也只是无害,不是有益。
5. **测量学**:greedy 全管线近乎确定(SciVB 双尺度零翻转、ExpVid ~1.6%);963 条 CoT 输出
   0 缺 marker;任何 ≥1pp 效应可读。旧仓库同一"72B C0"五个值漂移 10-12pp 的时代结束。

## 4. 新旧结论对照

| 旧结论(坏口径) | 公平框架终判 |
|---|---|
| 注入帮 7B(+3.12)伤 72B(−4.19)= 规模反转 | 两半皆改:7B 增益不显著;72B 伤害只在 SciVB(−8.72)与 L1(历史 sweep −4.19,布线已审计为净) |
| CoT 大幅伤害(7B −10pp / 72B −5.7pp) | 7B ExpVid −3.39 真(弃答行为,非截断伪迹);72B ExpVid 零(任务结构化);SciVB 72B −8.72 真 |
| 完美 oracle-KG 仍 −33pp(形式非内容) | 实验作废(选样回归+非真 oracle+解析 bug) |
| V9 KG 有害、时序边无用 | 绝对值作废;KG 注入在跑干净版(§5) |
| 弃答门(Gate-C)是唯一不伤 72B 的路径 | Gate-C 实为解析失败门;真实弃答路径从未被测(v10 的 P1/P3 将首次测试) |

## 5. KG 注入终审 + 注入家族全光谱(2026-06-12 完成,72B,G0 gate 判定)

同契约同进程配对(`run_unified_72b_kg.sh`):c0 / kg(V9 状态机图全量多视图渲染,中位 3229 字符)
/ kgs-placebo(字段名 bug 致只注入状态 ID 符号串,中位 51 字符——**意外的安慰剂臂**)
/ kgs2(修复后真稀疏:按题挑 2 条真实图事实,中位 206 字符)。

### 注入家族全光谱(vs 同进程 c0,配对 sign test)

| 注入物(字符量级) | ExpVid (n=745) | SciVB (n=218) |
|---|---|---|
| 无(c0) | 0.348 | 0.417 |
| 符号占位串(~51)| −0.09pp(p=.51,零) | **−6.88pp(p=7e-4)** |
| 真稀疏 2 条事实(~190) | −0.40pp(p=.45,零) | **−7.80pp(p=3e-3)** |
| prose 笔记(~800) | +0.18pp(p=.24,零) | **−8.72pp(p=3e-4)** |
| 全图渲染(~3229) | +0.37pp(p=.90,零) | **−10.09pp(p=2e-4)** |
| (参照)自生成 CoT | +0.99pp(p=.95,零) | **−8.72pp(p=.020)** |

### 判读(G0 gate:注入家族盖棺)

1. **ExpVid:五连平线**——从空白到全图,整体效应全部为零;唯一会动的是任务内部结构
   (kg 给 steppred +6.9pp p=.031、seqgen +4.5;mc −4.0 p=.058 对冲)。瓶颈与注入内容的
   数量、质量完全正交。
2. **SciVB:存在性主导的剂量曲线**——51 字符的**无内容符号串**就 −6.9pp,真稀疏 2 条事实
   (~190 字符)−7.8pp,prose −8.7,全图 −10.1,只随长度温和加深。**伤害的大头是"注入文本
   的存在"本身**(presence-dominated):占位符已占了全图伤害的 68%(−6.9/−10.1),从占位符
   到全图、内容增加 60 倍只多 −3.2pp。旧"form-not-content"主张以 n=218、p=7e-4 的安慰剂证据
   复活(且比当年 jove 轶事干净得多)。
3. **G0 判定:注入家族(prose/原子/稀疏/全图,含未建的 KB/RAG)在 72B 上正式盖棺**;
   v10 按零注入路线全速执行。KG 的价值定位为:结构任务的机制证据(steppred ✓)+
   v10 的不确定性账本/裁决依据/帧选择器(§见 V10 计划 3.5/3.6)。
4. KB 检索不立项:从未有可用实现(V8"KB"为图像接地库,假验证 bug + 0.1% 命中,线闭),
   且 presence-dominated 伤害 + generic-prior override 双重指向负期望。

## 5.5 各方法实现细节(近期测试的每个条件,代码级)

所有条件共享同一答题契约(`scripts/unified_harness.py` + `evaluate_c0_test_split.py` 的 BUILDERS):
每题抽 **32 帧均匀采样**(`extract_frames`,一次抽取全条件复用),`MAX_PIXELS = 360×420 = 151200`
像素/帧,greedy 解码(`do_sample=False`),按 task 给答题 token(mc=8 / seqgen=96 / steppred=16 / fitb=96)。
注入物统一放进 prompt 的 `_ctx_block`:`"Visual notes:\n{note}\n\n"`,位于 video token 与 Question 之间;
note=None 时该块为空。条件之间**只差这个槽位放什么**。输出过修复后的解析器
(`parse_mc_aj` 分层解析 / `score_fitb` 多分隔符)和打分器。

### c0 — 基线(无注入)
note=None,`Visual notes:` 槽位为空。32 帧 + 题干 + 选项直接进 72B,按 task 的 BUILDER 组 prompt
(mc 带 `Answer (A/B/.. only):` 结尾、seqgen 要求空格分隔步号、steppred 要求单整数、fitb 要求 ` | ` 分隔)。
这是冻结基线 R0。每个干预条件在**同进程**内对同一题也跑一遍 c0 → 严格配对、消除跨进程漂移。

### cot — 自生成推理(对照,非注入)
**实现**(`scripts/unified_harness.py:60-90, 287-294`):**不调任何外部工具、不动 note 槽位、不增加任何输入**,只在 c0 的 BUILDER prompt 上做一处改动 —— `add_cot_suffix(messages, item)` 深拷贝 c0 的 messages,在 user content **最后一个 text 块**末尾追加(verbatim,英文原文):

> `\n\nFirst reason step by step about what the video actually shows and how it bears on the question. Keep the reasoning under 150 words — do NOT enumerate every protocol step. Then end with ONE line exactly of the form:\nFINAL ANSWER: <{fmt_hint(item)}>`

其中 `fmt_hint(item)` 按 task 给格式提示:mc=`the SINGLE correct option letter (A, B, C, ...)`、seqgen=`the space-separated step numbers (e.g. '3 4 5')`、steppred=`ONLY the step NUMBER of the next step (a single integer)`、fitb=`the value for each blank, separated by ' | '`。

- **生成**:`vlm.generate(..., max_new_tokens=args.cot_tokens)`,`cot_tokens` 默认 **1536**(c0/c1 答题用 `ANSWER_TOKENS={mc:8,seqgen:96,steppred:16,fitb:96}`;修复前 cot 用 768 会截断长 seqgen 的推理→弃答,审计后放宽)。frames / system prompt / options / 32 帧抽取**与 c0 逐字相同**,唯一变量 = 这段推理后缀。
- **解析**(`parse_cot`,审计修复版):marker 正则 `_MARKER = (?:FINAL|EXACT)?\s*ANSWER\s*[:：]`。① **mc**:有 marker → `extract_final(raw)` 取 marker 后内容,无 marker → 用全文;再交 `parse_for_task`(mc 用 `parse_mc_aj` **自底向上**扫第一个合法选项字母,避免抓到推理中途 "heated to 60°C" 的 'C')。② **非 mc**:有 marker → 取 marker 后内容;无 marker → 取 verbose 输出**最后一非空行**。这条修复是关键:旧版从推理文本头部抓首字母,系统性误判。

### c1 — prose 笔记注入(ProtoNote C1_fixed,行内重建)
**实现**(`scripts/unified_harness.py:244-264, 295-298` + `evaluate_c0_test_split.py:60-61`):答题前先由**同一个答题模型自己**生成一段散文视觉笔记,塞进 prompt 的 `Visual notes:` 槽位,再走与 c0 **完全相同**的答题路径。`build_c1_note(item, vp, q, uid)` 流程:

1. **缓存**:笔记按 uid 缓存到 `notes_dir/<uid>.md`,命中直接读(保证可复现 + 同进程配对)。
2. **task 路由**:`classify_task(item)` 给题分类 → `tools_for_task(task)` 返回该 task 路由的工具集(不是所有题都调所有工具)。
3. **调工具(同一个 72B/7B 自己看视频)**:
   - `visual_inspect(video_path=vp, query=_VIS_QUERY)`,`_VIS_QUERY`(verbatim)= `"In 1-2 sentences, describe the key actions, materials, and any visible labels/quantities."` → 产出 `NoteEntry(section="Visual")`;
   - `ocr(video_path=vp, focus_query=q[:160])`(用题干前 160 字符做 OCR focus)→ `NoteEntry(section="OCR")`。
   - 每条经 `NoteBuffer.append_entry(uid, NoteEntry(...))` 累积。
4. **渲染**:`nb.render_for_llm(uid, question_context=q)` 把条目渲成带小节标题(`## Visual` / `## OCR`)的散文(中位 ~800 字符,每条带硬编码置信 0.85/0.80);无条目则空串→note=None。
5. **注入并作答**:`answer(item, frames, note, tt)` 调 `BUILDERS[tt](item, frames, note, ...)`,`_ctx_block(note)` = `"Visual notes:\n{note}\n\n"`,**插在 video token 与 `Question:` 之间**;note=None 时该块为空(退化成 c0)。frames / system / options / 解析器(`parse_for_task`)/ 答题 token 全与 c0 相同。

**单变量公平性**:c0 / cot / c1 三条在**同一进程、同一 uid、同一 32 帧抽取、同一 BUILDERS、同一解析打分器**下跑,唯一差异是 `Visual notes:` 槽位放什么(c0=空、c1=自生成散文笔记)或 user 文本是否追加推理后缀(cot)。这正是 §1「干净测量口径」的来源,使 §2 的 cot/c1 effect 可归因为单一干预。

### kg — 全图注入(V9 状态机知识图谱,多视图渲染)
答题前由同一个 72B 建图(缓存到 uid),两次 VLM 调用:
1. **Stage 1.1**(`run_stage1_1`):看 32 帧 → 输出结构化实体清单(`canonical_name` / `type`
   Material·Tool·Operator / `estimated_quantity` / `core_role` / `first_appearance` 时间);
2. **Stage 1.2**(`run_stage1_2`,本轮 `ledger=[]` **未启用 OCR**):为每个实体追踪状态变化,
   抽出 operations——每个 op 有 `action`(动词短语)/ `timestamp` / `duration` /
   `input_states → output_states`(状态转移,物质流骨架)/ `confidence`(硬编码 1.0);
3. `render_multi_view_kg(kg, ["procedural","conceptual","quantitative","hypothetical"], include_edges=True)`
   渲染成四视图 Markdown + 时序边,合成 `# Knowledge Graph (multi-view)` 文本块(中位 **3229 字符**,
   截断至 ≤4000);
4. 整块放进 `Visual notes:` 槽位作答。

### kgs-placebo — 稀疏注入(字段名 bug,意外安慰剂)
本应注入按问题词面挑选的 2 条图事实,但 `_kg_sparse_facts` 读错了 v9 字段名
(读 `description`/`name` 而非 `action`/`canonical_name`),实际注入的是无内容的状态 ID 符号串
("Operation: (inputs: E1_S1; outputs: E1_S1)",中位 **51 字符**)。**作为零内容安慰剂臂保留**——
正是它揭示了 SciVB 的 presence-dominated 伤害。

### kgs2 — 真稀疏注入(字段修复后)
`_kg_sparse_facts` 修复:从 kg 取所有 op 的 `action` 文本和实体的 `canonical_name`+`visual_features`,
按与问题的词面 token 重叠排序,取 top-2(中位 **~190 字符**),如
"- Operation: Preparing the samples (at ~1s)\n- Operation: Loading samples into electrophoresis cartridge (at ~5s)"。
放进槽位作答。这是 PI 2026-06-01"稀疏检索注入"方向的首次正确执行。

### P2.6-A OCR grounding(接地产出测量,非答题条件)
检验"启用从未用过的 V9 OCR ledger 能否接地图"。`p26_ocr_grounding_pilot.py`:
1. `OCRLedgerBuilder` 对 8 关键帧逐帧 OCR(`generate_image`),得带时间戳的屏幕文本 ledger;
2. 用 ledger 重建图(`run_stage1_1` → `run_stage1_2(ledger=ledger)`,这次**启用** OCR 对齐);
3. 确定性三值接地标记(必须能失败):实体名的判别 token 是否出现在 ledger / 估计数量是否对上
   ledger 数字 token;与盲建图缓存 diff 数量修正数。
⚠️ 指标偏松(`qty=1` 通配 + 单 token 匹配),宽松 38.7% / 严格地板 13.5%,真值待严格重测。

## 5.7 P2 wave-1:感知回看终判(2026-06-13,72B,负结果)

检验"不确定性触发→帧重看(更密/重聚焦)"能否帮 72B。三臂全 text-free(只交付原始帧,
note=None),对照同进程/冻结配对基线,McNemar on discordant 触发子集:

| 臂 | 干预(单变量) | rw | base | net | McNemar | 判定 |
|---|---|---|---|---|---|---|
| **arm-H** | 64帧 vs 32帧(均 direct,hedge-fired mc,n=78) | 0.218 | 0.231 | −1.28pp | +2/−3,p=1.0 | 零 |
| **arm-SV** | 窗口聚焦 vs 均匀(均 32帧,SciVB windowed,n=77) | 0.429 | 0.455 | **−2.60pp** | +8/−10,p=0.81 | **微负** |
| **arm-Step** | 64帧 vs 32帧(均 frontier+1,steppred,n=145) | 0.186 | 0.152 | +3.45pp | +12/−7,p=0.36 | 正向不显著 |

**结论(R2 感知层证伪)**:
1. **帧数翻倍(32→64)不帮 72B**:arm-H 干净版零(−1.28pp)、arm-Step 仅 +3.45pp 不显著。
2. **帧重聚焦反而有害**:arm-SV 同帧数、只把帧从均匀挪到 cited window,**−2.6pp**——
   强模型需要全局覆盖,不是局部细节。"question-conditioned 帧重选"假设被干净证伪。
3. **欠定位尾救不回**:arm-Step 的 26 题 ≥6 步欠定位子集,64 帧下 0.038 vs 0.000——
   几乎仍全死。更多帧不修复感知失败。
4. **唯一正向是结构,不是感知**:arm-Step 的 +3.45pp 来自 frontier-head 的 prompt 重构
   (问"最后步骤"再 +1),与帧无关,且 n=145 不显著。
5. **wave-2(arm-G/arm-GK,verified-KG 注入)被 gate 掉**:启动条件是 arm-H 越过噪声底,
   未达成 → PI 的 verified-KG 注入假设不再单独投入(注入家族已盖棺 + 感知回看证伪,
   双重负先验)。

**这把诊断的 input-bottleneck 视角反转了**:40% 的 TEMPORAL+PERCEPTION 误差**不是**靠
"喂更多/更准帧"能修的——直接干预证明 72B 在 32 均匀帧上已用尽可提取的感知信号,
瓶颈是模型对所见的利用能力(capability),不是输入带宽。

**v10 幸存版**:R1(任务路由,免费 ~+2pp)+ R3 的 prose-free 结构裁决头(frontier+1 给
steppred +3.4pp;vv 头 P4 待测)。R2 感知层与注入家族均已证伪。叙事收敛为:**72B 的天花板
不被感知(帧)、注入(文本)、自身推理(cot 净零)移动;只有确定性的答案抽取重构在
结构绑定任务上给小幅增益。**

## 5.8 盲测对照:视频净贡献(2026-06-13,72B,无视频只给问题)

`scripts/blind_no_video.py` — 同模型/解析/打分,prompt 去掉 video block,只给问题(+选项)。
视频净贡献 = c0(带视频) − blind。

| | blind | c0(视频) | **视频净贡献** |
|---|---|---|---|
| **ExpVid mc**(seq_ordering+verification,n=302) | 0.454 | 0.477 | **+2.3pp** |
| ExpVid fitb(n=137) | 0.132 | 0.282 | +15.0pp |
| **ExpVid seqgen**(n=161) | 0.029 | 0.446 | **+41.8pp** |
| ExpVid steppred(n=145) | 0.000 | 0.041 | +4.1pp |
| ExpVid 整体(n=745) | 0.214 | 0.350 | +13.5pp |
| **SciVB mc**(n=218) | 0.289 | 0.417 | +12.8pp |

**关键发现:这解释了 P2 感知回看为什么失败(§5.7)**:
1. **ExpVid mc 95% 盲测可答**(0.454/0.477,视频仅 +2.3pp)。sequence_ordering(4 选,随机
   25%)+ video_verification 本质是**选项判别/语言先验题,不是视频感知题**——模型盲测就接近
   上限,**总共只有 2.3pp 视频信息可加**,所以 arm-H(hedge mc 回看)注定无效:它要修的池子
   里根本没有视频 headroom。
2. **seqgen 是真视频绑定**(+41.8pp,blind 近 0)——这恰是之前 cot/kg 结构注入唯一帮上忙的
   任务。逻辑闭合:**视频绑定任务(seqgen)吃结构化视频提取;非视频绑定任务(mc)既不缺
   视频信息也无从被感知/结构干预帮助。**
3. **SciVB blind 28.9% ≫ 10% 随机**:概念题有大量语言先验/教科书可答性;视频真贡献 +12.8pp,
   但 arm-SV 证明这 12.8pp 不能靠帧重聚焦兑现(需全局覆盖)。
4. 论文含义:"为什么 72B 不被感知干预移动"的最终机制 = **任务的视频依赖度两极分化**——
   要么近乎无视频 headroom(mc),要么视频绑定但 32 均匀帧已饱和(seqgen/fitb)。两种情况下
   "加帧/重聚焦"都无增益。视频净贡献谱(+2.3 到 +41.8pp)本身是论文一张主图。

## 6. 下一步

[V10_REASONING_GRAPH_PLAN.md](./V10_REASONING_GRAPH_PLAN.md):零注入阶梯(确定性任务路由 ⊕
question-conditioned 帧重选 ⊕ prose-free 结构裁决器),试点 P0-P8 带 GATE,8/1 冻结数字,
目标 ICLR 2027。

## 7. 复现

```bash
bash scripts/run_unified_72b.sh      # 72B 主矩阵
bash scripts/run_unified_7b.sh       # 7B 主矩阵
bash scripts/run_unified_72b_kg.sh   # 72B KG 条件
python -m tools.aggregate_unified    # 聚合 + 检验
python -m tools.rescore_fixed        # 历史结果修复口径重打分
```
