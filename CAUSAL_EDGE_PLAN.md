# 因果边注入实施计划(operation 结构化 → 物质流因果 → MLLM+校验)

> 来源:用户 2026-05-31 下发。严格按 STEP 顺序执行,每步 VERIFY + GATE,gate 不过禁止下一步、停下报告。每步完成在 LOG 追加记录。

## 前提(执行者先确认,不满足就停下来报告)
- [x] grounding metadata 缓存 bug 已修(comprehension 显示真实值,非恒 0)。— 2026-05-31 已修 + 验证
- [ ] entity 去重已生效(无单视频 entity 数异常膨胀)。
- [ ] diagnose.py 可用(按 chunk_filename+line_index 配对,出 confusion matrix)。
- [ ] 固定小集(50-100 样本,覆盖 mc/seqgen/steppred/fitb)已确定,记录 id 列表,本计划所有实验都用同一批。

## 总原则
- 严格按 Step 顺序。每步有 VERIFY + GATE,gate 不过禁止下一步,停下报告。
- 每次改动后跑 diagnose.py,禁止只看总分下结论。
- 任何一步"加了反而更差"且无法改善 → 按该步 ROLLBACK,停下报告,不要硬撑。
- 每步完成在末尾 LOG 追加记录。

---

## STEP 1 — operation schema 升级为带 input/output 的结构化事件 [因果的原料]
当前 operation 是泛动词、无 input/output,无法支撑物质流/因果。重新设计 schema:
```
operation: {
  type: <受控词表: transfer/mix/heat/separate/measure/dispense/wait/...>,
  verb: <原始动词>,
  input_entities:  [entity_id, ...],
  output_entities: [entity_id, ...],
  t_start, t_end,
  description
}
```
改 Stage 1 operation 抽取 prompt:为每个操作输出 input/output_entities(引用已抽 entity id),尽量给受控词表 type。

**VERIFY(小集):** 带非空 input/output 的 operation 比例;抽查 5 样本人工核 separate/mix 类;对比升级前 operation 总数、非泛动词比例。
**GATE:** ①带 input/output 比例显著上升 + "No operations" 大幅减少;②diagnose.py 分数无明显退化;③若 operation 抽取仍很差 → 停下报告(地基)。
**ROLLBACK:** 新 prompt 致抽取崩溃 → 回退 prompt,报告"Stage1 无法可靠抽 input/output"。

## STEP 2 — 时序边 [低风险确定收益 + 探针:Stage4 会不会用边]
相邻/重叠 operation 连有向时序边(before / overlaps),source=temporal, reliability=high,渲染进 notes_md,改 Stage4 prompt 说明如何用时序关系。
**VERIFY:** diagnose.py 对比 STEP1 基线,重点 seqgen/procedural。
**GATE:** ①procedural/seqgen 正向或持平、其他无损 → 过;②⚠️探针:连时序边都对 procedural 毫无影响 → 先查 Stage4 prompt 是否真用了边,修好仍无效则报告(核心假设"Stage4 用图结构"存疑,STEP3 慎重)。
**ROLLBACK:** 掉点 → 简化渲染 / 只留 before 链,重测。

## STEP 3 — 物质流因果边 [可验证因果,先做]
从 STEP1 的 input/output 自动推:A 的 output_entity ∈ B 的 input_entities 且 A 时序先于/接近 B → A --[produces_input_for]--> B, source=material_flow, confidence=high。用去重后 entity id 匹配。渲染(与时序边分开标注),改 Stage4 prompt 用因果链。
**VERIFY:** 推出边数 / 每视频几条;抽查 5 样本人工核;diagnose.py 重点 counterfactual。
**GATE:** ①counterfactual 提升 → 核心结果之一;②边太少 → 回 STEP1 加强 output 抽取或接受覆盖有限靠 STEP4 补;③掉点 → ROLLBACK。
**ROLLBACK:** 掉点(input/output 抽错致错边)→ 收紧匹配 / 只留高置信边,重测。

## STEP 4 — MLLM 推断因果 + 物质流校验 [核心 novelty]
72B 模型读结构化 operation 序列推因果边(边/关系类型/理由/confidence,source=llm_inferred)。物质流校验:A 的 output ∈ B 的 input → 升 confidence(verified_by_material_flow);否则降/阈值丢弃(unverified)。合并三类边分层渲染。
**VERIFY:** LLM 推边数 / verified vs unverified;抽查 unverified 是否更像幻觉;diagnose.py 重点 counterfactual+整体;**关键 ablation:(a) 只用 verified 边 vs (b) 全部 LLM 边**。
**GATE:** ①用 verified 边后 counterfactual/整体提升 → 成功;②verified-only ≥ all-llm-edges(证明校验有用),否则如实报告;③掉点 → 回退 verified-only → 仅 STEP3 → 报告。
**ROLLBACK 分级:** 全部 LLM 边 → 仅 verified → 仅物质流 → 仅时序,每级跑 diagnose 确认。

## 全局停止条件
任何 STEP 的 GATE 显示"加了反而更差"且 ROLLBACK 后仍无改善 → 停止,生成完整诊断(分数+confusion matrix+失败 case 分类),报告等决策。

## 关键探针
- STEP 2 是伪装成功能的探针:连零幻觉的时序边都不能提分 → "Stage4 利用图结构"假设可能不成立,先解决"Stage4 为何不用边",别硬上 STEP3/4。
- STEP 4 的 verified-only vs all-edges 对比 = 核心卖点"物质流校验约束 MLLM 幻觉"的试金石,务必认真做。

## LOG
- [x] ORACLE-KG 判别实验 (2026-06-01) — 10 个分层样本(选 KG 伤害大的),同 7B 同帧同 renderer,只变 KG 内容:
  - **C0 裸视频=0.566, C2 自动KG=0.215, C_oracle 完美KG=0.236**。
  - **C_oracle ≤ C0(−33pp)且 ≈ C2(+2pp)→ 形式问题,非质量问题**:连正确/去重/物质流连好的完美 KG 都比裸视频差,修上游救不回。
  - 机制:失败是真·答错(锚定/误导,非解析假象)——例 steppred_2693:C0 答对的下一步,C_oracle 被 KG 的 operation 列表带偏成错误步骤。输出未截断(<400tok)。
  - 例外:**seqgen** C_oracle=0.453 ≫ C2=0.051(逼近 C0 0.548)——正确 KG 对序列任务确实远胜错误 KG。
  - caveat:n=10 且选了 KG-harm 子集(绝对 −33pp 被选择偏倚放大;全 80 集 KG 净伤害 −3.5pp);只测 7B;只测冗长多视图形式。
  - 脚本:scripts/v9_build_oracle_kg.py(72B 从 oracle_note 构建)、scripts/v9_oracle_answer.py;数据 results_protonote_v9/oracle_kg/。
  - **决策(用户)**:换形态 → **KG 当稀疏检索(question-conditioned 注入 1-2 条相关事实),不整块灌**。STEP 3/4 仍冻结。

- [x] DECISION 2026-05-31 — 计划改建在 **V9**(用户定);V9 已有去重+结构化 operation。
- [x] PREREQ (前提核验, 2026-05-31) —
  - metadata 缓存 bug:V8 已修;V9 本就无此 bug(_refresh_metadata 实时重算)。
  - 固定小集:已建 `tools/fixed_small_set_expvid.json`(80 条,fitb/mc/seqgen/steppred 各 20)。
  - diagnose.py:**仍缺**(待建,按 sample_id 配对 + 分任务 confusion matrix)。
  - entity 去重(=STEP0):V9 **已做好**(同轨聚合/异轨拆分;218样本 max 15 entity,0 膨胀)。
- [x] STEP 0 (entity 去重) — V9 已实现,验证通过(无膨胀)。
- [~] STEP 1 (operation schema) — V9 **结构已具备**(input/output_states,均 6op/10state),但:
  ① 计划要 entity 级 input/output,V9 是 **state 级** → STEP3 需改用 state_id 匹配;
  ② state_id 无存在性校验、空 op 计数;③ 无受控词表 type。
- [!] PROBE (STEP2 前置, 2026-05-31) — **Stage4 不消费图结构(no_entity_list_only)**:
  state_graph 构建但从不被读;Stage4 只喂 entity 名册 + operation 扁平串;gate_kg
  对 ~40% 题跳过整块 KG;n_transmutations=0 全样本。→ 边在当前 Stage4 下近乎零效果。
  另:V9 整体比 V8 低 ~4-5pp(scivb 均~21% vs ~25-26%)。
- [x] STEP 2 (时序边, 受控探针 2026-05-31) — 在固定 80-set 上,同一抽取只换 prompt(C0 无KG / C1 KG无边 / C2 KG+时序边),贪心:
  - 均分:C0=0.191, C1=0.156, C2=0.174。**KG 本身有害(C1<C0 −3.5pp);时序边只把 C1 拉回一点(+1.8pp)但仍低于无KG基线。**
  - 有边子集(40/80):C1 0.158→C2 0.171(+1.25pp),7↑/6↓ = 噪声级。
  - 分任务:mc 0.50/0.45/0.55;seqgen 0.211/0.170/0.143(边更差);steppred 0.05/0/0;fitb ~0。
  - 诊断:STEP1 VERIFY 通过(**100% op 有非空 input/output_states**);但 ops/样本 4.5、**50% 样本 <2 op 无边**;**n_transmutations=0**(实体级物质流为空)。
  - **GATE 判定:未通过。** 零幻觉时序边都不能稳定提分、且 KG 净有害 → 瓶颈是 **Stage4 消费(KG 对 7B 答题模型有害)**,不是缺边。按计划:STEP3/4 暂缓,先解决"为何 KG 有害/Stage4 不用结构"。
  - 改动:stage4 加 include_edges(默认False,向后兼容);新增 scripts/v9_step2_probe.py、tools/diagnose.py。
- [ ] STEP 2 (时序边) —
- [ ] STEP 3 (物质流因果) —
- [ ] STEP 4 (LLM 因果 + 校验) —
