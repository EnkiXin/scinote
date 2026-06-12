# v10 设计终审建议(2026-06-12,致 PI)

依据:HARNESS_AUDIT_2026-06-11.md 终审矩阵(72B,unified contract,results_unified/,2026-06-11 19:00 完成)+ 四个设计提案 + 两轮对抗批评。所有 pp 估计均已按批评者核实后的数据缩水;被批评者抓出的证据引用错误已在文中改正。

---

## 1. 四个范式的排序判决

**Paradigm A(零注入感知控制器)——强,推荐作为 SciVB 主攻 + 论文建设性章节的最低可行单元。**
它是四个设计里唯一完全绕开已证实的文本伤害通道(SciVB 上 cot 与 c1 注入同为 −8.72pp,p=0.020/0.00031,零噪声翻转)的方案,且 cot 伤害恰好集中在带时间戳的桶(range 0.458→0.354,single 0.421→0.276),感知是那里唯一没用过的杠杆。但两处必须修:(a) 批评者实测 7/96 range 题引用的时间戳**超出本地 mp4 时长**(其中 4 题 c0 已答对),原 trust gate 只查窗宽不查 `t_hi≤duration`,这是直接回归洞,必须加越界即回退 c0 的守卫;(b) "SciVB 零噪声底→3 个净翻转即可读"是范畴错误——零翻转底只约束确定性重跑,改帧分布是真干预,n=96 的二项 SE≈5pp,正确检验是 discordant 对上的 McNemar,且需 n≥172(全部带时间戳题)起跑。另注意:原稿声称复用 `average_rate` 的"逐字节相同 decode 循环"不准确——现有 extract_frames 走 `stream.frames` 计数、从不算时间戳,秒→帧号映射是新代码,且需对 `stream.frames=0` 的容器加 `fps×duration` 守卫。诚实估计:**SciVB +1.5~+3pp(约 30% 概率 ≈0,但 0 本身是干净的机制结论);ExpVid motion-energy 臂 0±噪声,严格 gate 在 SciVB 成功之后**。成本最低(~1 天,pilot 0.5 GPU-hr)。

**Paradigm B(post-hoc 结构 verifier,prose-free override)——可行,推荐作为 ExpVid 结构头的核心机制。**
四个设计中机制最稳、pilot 最可读(steppred n=145 零翻转,override 集合=discordant 集合,override precision 直接可读,~0.3 GPU-hr)。两处证据修正:(a) off-by-one 池实测 cot 下 |δ|=1 为 **37/145**(δ=−1:25,δ=+1:12),不是 42/145;(b) "oracle KG 3/3 seqgen 胜出"是 **oracle(gold 派生)结构**的结果,同实验里 auto(模型自建)结构并不优于 c0——B 在推理期建的是 auto 结构,这条 crumb 只给出 B 够不到的上界,不直接支持 B。整个收益压在一个未测量的量上:frontier 定位调用的准确率。批评者实测给出关键校准事实:**answer-only 枚举模式 frontier 中位超出 +7 步,而 cot 模式中位超出 0**——frontier 调用必须从 cot 校准的 decode 构建,并在 pilot 里直接测 frontier 准确率。另需做廉价消融"cot 答案盲目 +1"(零额外调用):若它追平完整管线,grounded 调用就是浪费算力。诚实估计:**verifier 对 per-task-best-base 的边际 +1~+2.5pp;steppred 0.124→0.16~0.19;vv +0~+1pp(false-absence 风险,严格次级)**。

**Paradigm C(presence-set + 确定性算术头)——弱,作为独立范式淘汰,枚举组件并入 B。**
核心乐观假设"max(S) 比精确集合匹配容易"被数据直接证伪:seqgen 自身题目上 max(S) 精确命中仅 **34.4%**,且 66 vs 39 系统性 overshoot;按原稿规定的 SEQGEN_SYSTEM answer-only 模式建集,max(S)+1 会超出 gold ~8 步,**比 cot steppred 0.124 还差,大概率自杀于自己的 pilot gate**。"seqgen 0.498"是 partial-credit F1,与 frontier 定位是两个量,"11x gap"框架不成立。其声称的 +7.53pp 约虚高一倍。但它有两件真资产:(a) contiguity-verify-then-abstain 守卫(seqgen gold 161/161 连续)是真正的下行保护;(b) "把失败任务重构成模型相邻强能力 + 确定性头"这个思想本身最有新颖性。**判决:不独立存活;其枚举(改用 cot 校准 decode)+ contiguity 守卫并入 B 的 contradiction+strength gate,合并体严格优于两者单独。**

**Paradigm D(typed schema CoT + SFT)——死(作为范式),仅 OPTION_DIFF 组件以末位 gated 条件存活。**
三个理由:(a) 旗舰干预瞄错了靶——A-collapse 是 **cot 诱发的症状**(cot 错题 pred=A 33/49),c0 上根本不存在(c0 错题 A13/B10/C7/D4 均衡,acc 0.773),所以 seqord 上没有结构杠杆,schema 的任务只是"别把 cot 的 −10pp 伤害带回来",这不是收益;且 collapse 的根因是感知失败("选项近乎相同")而非格式缺失,schema 索要 divergence 只会拿到幻觉的 divergence。(b) 2048-token 的 schema 文本仍是 answer context 内的生成文本,离被证实 −8.72pp 的通道最近,"绕开 prior-override"的声称在四案中最弱。(c) Phase-2 SFT 是另一篇论文的体量,塞不进 ICLR 2027 窗口。诚实估计 0~+2.5pp 且高方差,vv/steppred 臂与 B 重复。**唯一正交资产是 seqord OPTION_DIFF 臂(B/C 都不碰 seqord),作为最后一个单独 gated 条件保留:acc<0.74 或 pred=A 占比>0.45 即杀,只有 ≥0.79 才并入。**

---

## 2. 推荐的 v10 组合架构:单变量阶梯(Ladder)

批评者找到的 synergy 与我的综合一致,但与任务提示里猜的"D 为核心"不同——**核心是 A(感知)⊕ 合并后的 B/C(prose-free 结构 verifier),共享一个确定性 router;D 只剩一个末位组件**。架构表述为冻结契约下的单变量阶梯,每级只动一个声明变量,从而把感知收益、路由收益、结构收益、格式收益相互隔离:

```
R0  c0(冻结基线,results_unified 已有)
     ExpVid 0.3496 / SciVB 0.4174

R1  route:确定性 per-task 路由(零模型调用,benchmark 自带 task 字段)
     {seqord, fitb, 全部SciVB, 全部L1} → c0 原样
     {steppred, seqgen, vv} → cot
     绑定证据:SciVB cot=c1=−8.72(真);fitb cot −2.9;seqord cot −10(A-collapse);
              steppred cot 3x、seqgen +5.2 —— 路由是机制先验,不是调参
     诚实标价:测得 +3.53pp 但 sign-p=0.056、在本测试集上选出、含 seqgen 5.6% partial-credit 噪声;
              held-out 折价记 ~+2pp。这个脆弱性全梯只计价一次,不许四个组件重复计入。

R2  route + zoom(A 层,感知通道,零文本)
     SciVB 臂:regex 解析 mm:ss 窗 → 守卫(2s≤宽≤90s 且 t_lo≥0 且 t_hi≤duration,
              stream.frames=0 时用 fps×duration)→ 8 global + 24 in-window 重解码(无 seek 索引循环);
              守卫不过 → 逐项回退 c0。另跑一个预算增量臂(32 global + 24 window = 56 帧)
              把"密集"与"少 global"解耦。
     ExpVid 臂(gate 在 SciVB 臂成功后):motion-energy 活动段密集重解码,喂给 R3 的 frontier 调用。
     绑定误差池:79% SciVB 题引用 mm:ss 而 32 均匀帧在被引窗内期望 ~0.8 帧;
              cot 伤害集中在 timestamped 桶 = 感知是该处未用杠杆。

R3  route + zoom + struct(合并 B/C 的结构头,只输出字母/整数,零散文)
     steppred(n=145,主靶):cot 校准的 frontier 调用(STRICT JSON last_observed + evidence 文本)
       → 确定性矛盾触发(committed ≠ frontier+1)→ token-F1>0.4 强度门 → override = frontier+1;
       解析失败/集合非连续(C 的 contiguity 守卫)→ 保持 cot 答案。
       绑定误差池:cot 下 δ=−1 共 25 题 + δ∈{−2,−3} 共 22 题;cot 模式 frontier 中位超出 0(已校准)。
     vv(n=152,次靶,gate 在 steppred 通过后):per-step present/absent 覆盖向量 + margin≥2 门
       → argmin 覆盖;0 或 >1 候选时用 D 借来的 pairwise 判别(OPTION_DIFF 式)收窄到 ≤3 个歧义步。
       绑定误差池:vv c0 0.184 近随机、mc oracle 天花板 +29pp;false-absence 由 R2 的密集帧直接缓解
       ——这是最强的跨设计 synergy:更好感知 → 更准 present-set → 更准缺步推导。

R4  + seqord OPTION_DIFF(D 的孤臂,最后跑,投机性,默认不进 headline)
     kill:acc<0.74 或 pred=A>0.45;并入:≥0.79。
```

**诚实的合计预期(对 c0):ExpVid +3~+6pp(routing ~+2~3.5 脆弱 + struct 边际 +1~+2.5 + zoom-ExpVid 0~+1.5),SciVB +0~+3pp(zoom),绝不是四案各自 headline 相加的 +5~+7.5pp——那个加法对共享的 routing 地板和结构任务 headroom 重复计了价。** SciVB 在 R1/R3/R4 全程不被触碰(c0 0.4174 是该处最优条件),是设计上的"干预被正确扣留"对照组。

---

## 3. 与"构造 reasoning graph"原始想法的关系

**保留了什么。** 你的四步直觉——建图、验证、找因果、用图推理——其中**"验证"是被证据支持得最强的一步,它在 v10 里从配角升为主角**:R3 的全部机制就是"图对已承诺答案的一个离散量做裁决"。支持它的证据有三条:(a) hedge 语言 23.8pp 判别力(hedged 27.9% vs 51.7%)说明模型自己"知道"哪里不稳,可被结构化裁决利用;(b) precision-gate 项目的幸存结论——强模型只被 abstention 帮助,而不被任何注入的中间产物帮助——和"verifier 不是 injector"完全同构;(c) oracle 结构在 3/3 seqgen 上胜出 → **在答案本身就是结构的任务上,结构内容有信号**(审计保留的边界结论)。图的本体也保留了:节点=题目自带的 protocol 步骤,边=时序先后/在场关系,接地=视频帧——这就是一个 scene/protocol graph,只是瘦身成了任务需要的最小结构。

**为什么变形。** 三处变形,各有一条硬证据逼迫:(a) **图不再以文本形式进入 answer prompt**——终审矩阵的机制级发现是自生成推理与外部注入在概念题上共享同一 prior-override 失败通道(双双 −8.72pp,SciVB 零噪声),所以图只能以"一个字母/一个整数的 override"或"帧分布的改变"作用于答案,任何散文化的图都被这条通道吞掉;(b) **"找因果关系"暂缓**——干净的时序边消融 C2−C0=−1.07pp≈0(审计修正后唯一站得住的边结论),且基底测量显示 ExpVid 50% 视频不足 2 个操作、material-flow transmutation=0,MECD 式因果发现在这两个基准上没有可测的肥沃土壤;它是 v11 候选,不是 v10 组件;(c) **"用图推理"变成"从图确定性地推导"**——因为推理文本通道已被证实有害,而确定性算术头(frontier+1、argmin coverage)把 LLM 从最终裁决中移出,这是 SG-VLM("limited gains, no verification")没做的那一步。

**你的哪个直觉被证据支持。** 最核心的一个:**"模型需要先把视频接地成结构、再核对,而不是直接一口报答案"——在答案即结构的任务上(steppred/vv/seqgen)这是对的**,cot 在这些任务上 3x/+5.2/+4 的提升和 oracle 结构 3/3 的胜出都是它的证据。被证据修正的是作用点:接地的瓶颈在**感知**(32 均匀帧覆盖不到被引窗、frontier 定位 34% 上界),不在推理;所以 v10 把"建图"的前半段做成了帧重解码(R2),把后半段做成了 prose-free 裁决(R3)。一句话:**你的图活了下来,但它从"喂给模型的输入"变成了"管住模型的输出"。**

---

## 4. 实验路线(全部接入 scripts/unified_harness.py,一条件一 flag 一 builder)

判读总则:每个条件对其参照臂做 n=745/218 全集 paired sign test(干预性条件用 McNemar on discordant);ExpVid 噪声底 1.6% 翻转/−0.15pp,SciVB 重跑底为零但**干预条件按二项方差读,不许借用零翻转底**;每个最终入选条件加 `--tag rep2` 独立进程复跑。

| 步 | 条件名 | 内容 | n | GATE(对照) | GPU | 日历 |
|---|---|---|---|---|---|---|
| P0 | `v10_frontier_calib` | 测量步:cot 式 frontier JSON 调用,直接测 last_observed 命中率(同时记 answer-mode 对照) | 145 | frontier 命中 <40% → R3 主靶降级、提前转 vv+R2 联合 | 0.3 GPU-hr | 6/15–6/17 |
| P1 | `v10_steppred` + 消融 `cot_plus1` | B 核心:contradiction + F1 门 override;并跑零调用"盲+1"消融 | 145 | vs cot 18/145:≤20 杀;≥26 续;21–25 读 override precision(<50% 杀);若 cot_plus1 追平 → 砍 grounded 调用 | 0.3 GPU-hr | 6/17–6/20 |
| P2 | `c2_zoom`(SciVB) | 先做 5 题帧 dump 目检时间戳对齐(**先于全量**);带 t_hi≤duration 守卫;全部 172 个 timestamped 题 | 172 | McNemar on discordant,预注册;c0=76/172;杀:净负;续:discordant 胜负 ≥2:1 且 p<0.05;灰:加预算增量臂(56 帧)再判 | 0.5 GPU-hr | 6/18–6/24 |
| P3 | `c2_zoom_b56` | 预算增量臂(32 global+24 window),解耦"密集 vs 少 global" | 172 | 只用于归因,不进 headline | 0.5 GPU-hr | 同上 |
| P4 | `v10_vv` | vv 覆盖头(margin≥2 + OPTION_DIFF 式歧义收窄),**仅 P1 续后跑** | 152 | vs cot 34/152:≤37 杀;≥42 续 | 0.4 GPU-hr | 6/24–6/28 |
| P5 | `zoom_expvid` | motion-energy 活动段密集重解码,**仅 P2 续后跑** | 745 | vs c0 全集 sign test,须 >1.6% 翻转底 | 3–4 GPU-hr | 7/1–7/5 |
| P6 | `v10_full`(R3 全梯) | route+zoom+struct 合体,R1/R2/R3 各级单独入表 | 745+218 | 全集 vs c0 与 vs R1:R3−R1 边际须 p<0.05;rep2 复跑,任务级增益须超该任务 rep2 翻转率 | ~5 GPU-hr ×2(含 rep2) | 7/6–7/15 |
| P7 | `v10_seqord_diff` | D 孤臂,最后跑 | 150 | <0.74 或 pred=A>0.45 杀;≥0.79 才并入 | 0.3 GPU-hr | 7/15–7/18 |
| P8 | holdout 路由确认 | 若可凑 held-out clip 池,复确认 R1 路由方向(只需方向,不需量级) | — | 方向反转 → headline 改报 R3−R1 边际、弃 R1 计价 | ~2 GPU-hr | 7 月下旬 |

总算力 <25 GPU-hr,在 8×H200 上完全不构成约束;关键路径是 gate 之间的串行依赖(P1→P4,P2→P5),全程 5–6 周。

**日历对齐 ICLR 2027**(abstract 约 2026-09 中旬、full paper 约 09 下旬):6 月下旬完成 P0–P2 三个决定性 pilot;7 月中完成 R3 全梯 + rep2;**8/1 冻结全部数字**,8 月写作(阶梯消融表 + 分析资产章节),9 月初内审,踩线提交。每步完成后按惯例更新 PROGRESS.md 并 push。若 P1 和 P2 双双被杀(最坏情形),论文退守纯分析叙事 + "感知与结构干预均被正确证伪"的负结果章,仍然成文,日历不变。

---

## 5. 论文策略

**论题(一句话):在强 VLM 上,自生成推理与外部注入文本共享同一条 prior-override 失败通道;结构只有在该通道之外——作为感知控制器和 prose-free 结构裁决器——才能帮助模型。**

四块资产拼成一篇:

1. **机制章(已在手,论文级发现)**:form-vs-content——c1 注入与 cot 在 SciVB 上**逐点同为** −8.72pp(p=0.00031/0.020,零噪声),证明伤害不来自文本的来源而来自文本的存在;这是对 CoT-harm 文献的精化:伤害是通道性的,不是 CoT 特有的。
2. **结构化章(已在手)**:CoT 的任务结构——steppred 3x / seqgen +5.2 对冲 mc −3.0 / fitb −2.9 净零;A-collapse 的 8v0 翻转证据(p=0.008)给出"近平局判别 + 文本通道 = 位置先验坍缩"的具体机制。加 scale-inversion:7B–72B interaction 三基准全显著(+2.65/+3.48/+9.17pp)——注入是弱模型的拐杖、强模型的噪声。
3. **建设性章(v10 阶梯)**:把 1、2 的机制翻译成干预——文本通道有害 → 走感知通道(R2)与离散裁决通道(R3)。阶梯消融表(c0→route→+zoom→+struct→±OPTION_DIFF)本身就是论文主表,每级一个变量、每级对照噪声底。**A 即使打出 0 也完成叙事闭环**("感知不是概念题瓶颈"同样是机制结论);R3 打正则给出"图只作为 verifier 有效"的建设性证据。SciVB 全程不动,作为"干预被正确扣留"的对照,堵住 over-claim。
4. **测量学章(差异化卖点)**:同仓库五个互斥的"72B C0"(漂移 10–12pp > 任何方法效应)、被审计推翻的 oracle-KG −33pp、被修复的六类解析 bug、冻结契约 + 实测噪声底——作为"VLM 干预研究的测量卫生"方法论贡献。这一章使我们对 SG-VLM 式"limited gains"文献的批评有牙齿:没有冻结契约和噪声底,那些 limited gains 不可解释。

与 SOTA 的关系要诚实写:本地 72B(~0.35/0.42)对 GPT-5(ExpVid L2 57.5)/Gemini-2.5-Pro(SciVB 64.3)差距巨大,论文不卖 SOTA,卖的是**在受控开放权重模型上对"结构/推理干预何时、为何、以何种形式有效"的机制定界**——这恰好是闭源 thinking-model 报告给不出的东西。

**最后一条风险声明**:全梯最大的单点脆弱性是 R1 路由的 +3.53pp(p=0.056、测试集选出)。论文中 routing 与 struct/zoom 边际必须分列,headline 以 R3−R1 边际和 R2 的 McNemar 为准;若 P8 holdout 不可得,R1 在正文降格为"机制先验下的固定系统配置"而非测得收益。

相关文件:`/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/HARNESS_AUDIT_2026-06-11.md`(审计)、`scripts/unified_harness.py`(冻结契约,所有新条件的接入点)、`results_unified/`(R0 冻结基线与 rep2 噪声底)、`evaluate_c0_test_split.py` / `evaluate_unified.py`(复用的 BUILDERS/解析/打分)。