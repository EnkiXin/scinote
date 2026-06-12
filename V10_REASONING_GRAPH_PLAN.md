# V10 计划 v2(2026-06-12 更新,致 PI)

> v1(四范式设计 + 对抗评审的完整论证)收录于本文附录;本版整合:双尺度公平矩阵终审、
> KG/kgs 注入实验(中期)、PI 的 verified-KG 假设与置信回看思路 → 合并为双触发器回看实验。
> 配套阅读:[REPORT.md](./REPORT.md)(结果)· [HARNESS_AUDIT_2026-06-11.md](./HARNESS_AUDIT_2026-06-11.md)(口径)

---

## 1. 论题与目标

**论题**:在强 VLM 上,自生成推理与外部注入文本共享同一条 prior-override 失败通道;
结构只有在该通道之外——作为**感知控制器**(决定看哪里)和 **prose-free 裁决器**(只输出
字母/整数的核查)——才能帮助模型。

**目标**:不卖 SOTA(与 GPT-5/Gemini-2.5 的 21-26pp 差距是原生 long-CoT 能力)。卖的是
机制定界 + 一个诚实的建设性阶梯:ExpVid +3~6pp、SciVB +0~3pp(批评者缩水后的估计),
每一级对照实测噪声底。目标会议 **ICLR 2027**(9 月底截稿)。

## 2. 设计的硬约束(公平框架终审,全部 n=745/218、配对检验、实测噪声底)

| 事实 | 对设计的含义 |
|---|---|
| 自由文本在任何格子都不增益;72B SciVB 注入=CoT=−8.72pp;7B ExpVid CoT −3.39 | 一切组件不得向 answer context 添加自由文本 |
| CoT 任务结构化:steppred ×3、seqgen +5.2 ↔ mc −3.0、fitb −2.9 | 确定性 per-task 路由是免费的第一级 |
| KG 注入(中期 n=300):整体零,mc −9.3 / seqgen +4.5,稀疏版两头抹平 | 结构内容只在"答案即结构"任务有信号;注入形态到此为止 |
| hedge 语言判别力:ExpVid mc 23.8pp(oracle 上限 +29pp);SciVB 仅 7.6pp | 回看的触发信号在 ExpVid mc;SciVB 走 mm:ss 窗口缺帧触发 |
| 数字自评置信恒高(72B 4.93/5;V8 元素置信是 0.90/0.70 假数) | 数字置信不得作为任何门;实验中记录它以正式埋葬 |
| 79% SciVB 题引用 mm:ss 窗,32 均匀帧窗内期望 ~0.8 帧;TEMPORAL+PERCEPTION ≈40% 误差 | 感知(回看)是最大且唯一未用的杠杆 |
| 图稀:ExpVid ops 中位 2;V8"验证"曾是假验证(USE_AS_IS) | 核查必须是感知动作(重解码确认),不是重贴标签;门槛不得"全有或全无" |

## 3. 架构:单变量阶梯(每级只动一个声明变量)

```
R0  c0 冻结基线(已有):ExpVid .3496 / SciVB .4174(72B)
R1  + route   确定性 per-task 路由(零模型调用):
              {steppred, seqgen, video_verification} → cot;其余(含全部 SciVB)→ c0
              标价 ~+2pp(测得 +3.53 但 p=.056、测试集选出,全梯只计价一次)
R2  + rewatch 不确定性触发的回看(零文本,核心层,见 §4 P2):
              触发 → 定位时间窗 → 密集重解码(8 global + 24 in-window)→ 原始帧重答
              守卫:2s≤窗宽≤90s、t_lo≥0、t_hi≤duration、不过则逐项回退
R3  + struct  prose-free 结构裁决头(只输出字母/整数):
              steppred:cot 校准 frontier 调用 → 矛盾触发 → F1>0.4 强度门 → override=frontier+1
              vv:per-step 覆盖向量 + margin≥2 → argmin;歧义时 pairwise 收窄
R4  ± OPTION_DIFF(seqord 孤臂,投机,kill: acc<0.74 或 pred-A>0.45)
```

图(scene/KG)在阶梯中的角色:**R2 的不确定性账本 + R3 的裁决依据,永不注入**——
除非 P2 的 arm-GK 推翻这一条(见下)。

## 4. 实验序列(全部接 `scripts/unified_harness.py`,一条件一 flag;判读对照 rep2 噪声底)

| # | 实验 | 内容 | GATE(kill / continue) | 成本 | 日历 |
|---|---|---|---|---|---|
| G0 | **KG 终判**(在跑) | c0/kg/kgs × ExpVid+SciVB @72B | kgs 在任一基准转正 → R3 加注入候选;否则注入家族盖棺(含 KB/RAG) | 已投 | 6/12 晚 |
| P0 | frontier 校准 | cot 模式 JSON 调用测 last_observed 命中率(同录 answer-only 对照) | 命中 <40% → R3 主靶降级,转 P2 联合 | 0.3 GPU-hr | 6/13-15 |
| P1 | steppred override | R3 核心 + "盲+1" 零调用消融 | vs cot 18/145:≤20 杀 / ≥26 续;盲+1 追平 → 砍 grounded 调用 | 0.3 GPU-hr | 6/15-17 |
| **P2** | **双触发器回看**(合并 PI 两案 + 原 c2_zoom) | 共享"触发→密采→原始帧重答";三臂:**arm-H** hedge 触发(ExpVid mc 池)/ **arm-G** 图元素不确定触发(行为性:双提取不一致+hedge,逐元素受约束二值核查,每题≤3 元素)/ **arm-GK** = arm-G + 核查后图注入(PI 的 verified-KG 完整版)。全臂记录数字置信(预期无判别,正式埋葬)。SciVB 臂用 mm:ss 缺帧触发(=原 c2_zoom,n=172,先 5 题目检对齐) | 触发子集 McNemar:纠错>误伤且 p<.05;**arm-GK>arm-G → "确定性是 KG 缺失成分"成立;arm-GK≈arm-G → 图是脚手架,感知承重** | ~6 GPU-hr | 6/17-24 |
| P3 | 预算增量臂 | 56 帧均匀(解耦"密集 vs 少 global") | 仅归因,不进 headline | 0.5 GPU-hr | 同上 |
| P4 | vv 覆盖头 | margin≥2 + pairwise 歧义收窄(P1 续后) | vs cot 34/152:≤37 杀 / ≥42 续 | 0.4 GPU-hr | 6/24-28 |
| P5 | ExpVid 回看臂 | motion-energy 活动段密采喂 R3(P2 续后) | 全集 sign test 超 1.6% 噪声底 | 3-4 GPU-hr | 7/1-5 |
| P6 | 全梯 R1→R3 | 各级单独入表 + rep2 复跑 | R3−R1 边际 p<.05;任务级增益超该任务 rep2 翻转率 | ~10 GPU-hr | 7/6-15 |
| P7 | OPTION_DIFF | seqord 孤臂 | <0.74 或 pred-A>0.45 杀;≥0.79 并入 | 0.3 GPU-hr | 7/15-18 |
| P8 | holdout 路由确认 | 若可凑 held-out 池,只验方向 | 反转 → headline 改报 R3−R1 边际 | ~2 GPU-hr | 7 月下旬 |

总算力 <25 GPU-hr;关键路径是 gate 串行(P0→P1→P4,P2→P5),全程 5-6 周。

## 5. 假设登记表(PI 的思路 → 哪个臂裁决)

| 假设(PI 原话大意) | 裁决臂 | 当前先验 |
|---|---|---|
| "graph 中不确定的元素去确定,全部确定后才使用 KG" | P2 arm-GK vs arm-G | 注入半受 G0/c1 压制;核查循环半被感知证据支持 |
| "根据置信度判断要不要回看 video" | P2 arm-H(触发器换 hedge;数字置信同录埋葬) | 回看方向强支持;数字置信已证无信号 |
| "根据置信度判断要不要 RAG" | 不立项(−8.72 通道 + kgs 无害无用 + prior-override 供弹) | 放弃;留一行论文证据 |
| "时序/物质流因果边" | 暂缓为 v11(基底:50% 视频 <2 ops,transmutation=0) | 待 P2/P5 把感知层垫起后再评估 |

## 6. 日历与论文

- **6/12 晚** G0 → **6/13-24** P0-P3 → **7 月** P4-P8 → **8/1 冻结全部数字** → 8 月写作 →
  9 月初内审 → ICLR 2027 提交。
- 论文四章:①机制(自由文本伤害通道无关,c1≡cot 逐点 −8.72)②任务结构(CoT/KG 同形的
  得失表 + 8v0 翻错)③建设性(R0→R4 阶梯消融主表 + P2 假设裁决)④测量学(审计 + 冻结契约
  + 噪声底方法论)。最坏情形(P1、P2 双杀)退守"分析 + 干预被正确证伪"叙事,日历不变。
- 全梯最大单点脆弱性:R1 路由 +3.53pp(p=.056,测试集选出)——headline 以 R3−R1 边际
  与 P2 的 McNemar 为准;P8 不可得则 R1 降格为"机制先验下的固定配置"。

---

## 附录:v1 四范式判决摘要(完整论证见 git 历史 `357dcb5a`)

- **A 感知控制器(零注入回看)— 强**:唯一完全绕开文本伤害通道;须加 t_hi≤duration 越界守卫;
  SciVB +1.5~3pp(30% 概率 ≈0,0 也是干净机制结论)。
- **B 后验结构裁决 — 可行**:机制最稳、pilot 最可读;frontier 必须用 cot 校准模式
  (answer-only 中位偏 +7 步,cot 偏 0);边际 +1~+2.5pp。
- **C 算术头 — 并入 B**:max(S) 命中仅 34.4%,独立必死;contiguity 守卫与枚举组件并入。
- **D typed-schema CoT — 死**:A-collapse 是 cot 诱发症状(c0 上不存在),schema 无靶;
  2048-token schema 文本离 −8.72 通道最近;仅 OPTION_DIFF 孤臂存活(P7)。
