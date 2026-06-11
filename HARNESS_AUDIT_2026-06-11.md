# Harness 审计 + 修复 + 统一公平框架(2026-06-11)

**背景**:PI 质疑"所有方法在 72B 上都失败"可能是代码 bug。三个多智能体审计/诊断 workflow
(共 ~60 个 agent)对全部评测代码做了数据流追踪 + 独立重打分 + 对抗复现。结论:**评测/解析层
确有多处 bug,实验布线本身基本干净;五个支柱结论两个被推翻、一个被改写、两个成立。**

## 1. 五个支柱结论的判定

| # | 结论 | 判定 | 关键证据 |
|---|---|---|---|
| K1 | 注入伤 72B(−4.2)帮 7B(+3.1)规模反转 | **改写** | 跨基准拼接:−4.19 是 L1(p≈2e-10,真,集中 l1_operation −12.7),+3.12 是 L2/L3;同基准:L2/L3 72B **−0.35,p=1.0(无效应)**;SciVB −10.55(p≈3e-4,真) |
| K2 | H2 证伪:precision 原子≈prose 同样伤 72B | **布线干净但统计为零** | n=75 仅 6 个 discordant pairs,McNemar p=0.219;pilot 比全集偏易 8pp、MDE~8-10pp;Gate-C"弃答有效"叙事塌(见 bug#6) |
| K3 | CoT 大幅有害(7B −10pp 级) | **7B 大半伪迹;72B mc 真** | 7B seqgen 16/20 CoT 在 768 token 内未到 FINAL ANSWER 记 0;72B mc CoT 40% vs direct 80%(8v0 翻错,p=0.008,"两项相同选 A"式 tie-break)真实 |
| K4 | 完美 oracle-KG 仍 −33pp(形式非内容) | **作废** | ① mc 解析器从 '60°C' 抓 'C'(57613 三臂同答 D,oracle 臂记 0);② "oracle" 从未见 gold、per-option evidence 含错误选项、50969 编码错序主动误导;③ n=10 按前轮 KG-harm 选样(prior c2 全 0)→回归均值 |
| K5 | V9 probe:KG 有害、时序边无用 | **绝对值作废** | CoT prompt 配简答 scorer:同 fixed-80 标准 C0=0.31 vs probe C0=0.19;fitb+steppred 40/80 无信号;修正解析后 MC 子集 C0 0.50/C1 0.45/C2(KG+边)0.60 **方向反转**;C1-vs-C2(边消融)干净且≈0 这条仍立 |

**横切发现**:"72B C0" 在仓库至少 5 个值(21.6/33.6/35.13/42.94/45.3,因 prompt/帧数/子集/scorer 而异),
基线漂移 10-12pp > 任何方法效应;`ANALYSIS_72B_REASONING_ERRORS.md` 的 "CoT 使 26.7→33.6" 是
跨基线伪迹(26.7 是 7B 的);"贪心解码不稳定"被反驳——63/65 重复条目预测不同实为 **sample_id
碰撞**(SciVB 65 个 id 对应两道不同题)。干净的部分:precision-gate 条件布线、C1 sweep 单变量
对照、note 缓存 md5 对齐(4466 文件零错位)、greedy do_sample=False。

## 2. 确认的 bug 与修复(全部已修 + 回归测试通过)

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `evaluate_c0_test_split.py:parse_mc_aj` | 取全文第一个 `\b[A-J]\b`,从 '60°C'/'N/A' 抓错字母 | 分层解析:marker 后 > 纯字母裁决行(自底向上)> 首行打头 > "answer is X" 短语 > 带 `(?<![0-9°/\w])` 防护的兜底 |
| 2 | `evaluate_unified.py:parse_mc` | 同上([A-D] 版) | 委托给修复后的 parse_mc_aj |
| 3 | `evaluate_unified.py:score_fitb` | 只按 `\|` 切分,模型用 ;/,/换行 → 第 2+ 空全记 0(0/137 条含管道) | `_split_fitb_pred`:\| > ; > 换行 > 非数字逗号 |
| 4 | `scripts/cot_ablation.py:extract_final` | 只认 "FINAL ANSWER";'EXACT ANSWER:' 被错过;'N/A' 流向解析成 A | 接受 FINAL/EXACT/裸 ANSWER;refusal→空 |
| 5 | `scripts/cot_ablation.py` 预算 | cot 768 截断(7B seqgen 16/20)、direct 32 截断长列表 | 默认 1536/64;fitb 提示加 ' \| ' 分隔说明 |
| 6 | `scripts/precision_gate.py:250` + `_parse_assessment` | CONFIDENCE 解析失败默认 0.0 < TAU=3 → P2"弃答门"只在解析失败时触发(模型从不输出 1-2 分) | `conf_parsed` 区分无信号 vs 低置信;`p2_on = conf_parsed and conf < TAU` |
| 7 | `protonote/data/loaders.py:load_test_split` | SciVB 65 个 sample_id 被两道不同题共用,join/缓存静默串题 | 每条加 `uid = sample_id + '#' + sha1(question)[:8]` |
| 8 | `scripts/v9_*`(probe/oracle) | CoT prompt 配简答 scorer + C0 与 KG 臂 footer 不一致 + oracle 非 gold 接地 | 未单修(实验作废);重做须经统一框架 |

## 3. 修复口径重打分历史结果(`tools/rescore_fixed.py` → `results_rescore_fixed/summary.json`)

- CoT 消融(stored→fixed,overall):72B ExpVid cot .337→.393 / direct .394→.444(差距 −5.1pp 仍真);
  7B ExpVid cot .129→.160 / direct .232→.271;7B SciVB cot .200→.217 / direct .333(7B seqgen 截断
  伪迹部分仍混杂,以统一框架重跑为准)
- 72B CoT-C0 全集:ExpVid .337→**.358**(fitb .012→.127,分隔符修复 +11.5pp);SciVB .349 不变
- C0-vs-C1 sweep(修复解析后不变,审计结论确认):L2/L3 −0.34(p=.24)/ L1 −4.19(p≈0)/ SciVB −10.55(p=2e-5)
- oracle-KG n=10:修复后 c0 .483 / auto .222 / oracle .200 —— 但该实验因选样+非真 oracle 不可救,作废

## 4. 统一公平框架(`scripts/unified_harness.py`)

单一测量契约,一切钉死:同 uid 题集、每题一次 32 帧抽取、同 BUILDERS、greedy、修复后解析/打分;
条件只差一个声明变量:**c0**(无 note)/ **cot**(+逐步推理后缀,≤150 词,FINAL ANSWER 行,
marker-aware 抽取,1536 tok)/ **c1**(同模型行内构建 C1_fixed prose note,uid 缓存)。
`--tag rep2` 独立进程重跑 c0 = 全管线可重复性噪声底。

**进行中的矩阵(本日启动,`scripts/run_unified_72b.sh`,nohup)**:
72B × {ExpVid 745, SciVB 218} × {c0, cot, c1} + rep2(c0),双实例(GPU 0,1 / 3,4)分 chunk,
输出 `results_unified/72b_*_{main,rep2}_chunk*.jsonl`,日志 `results_unified/logs/`。
判读规则:任何条件间差异须 > rep2 噪声底且 sign-test 显著才算效应。

## 5. 待办(矩阵跑完后)

1. 聚合 main vs rep2 → 噪声底;c0/cot/c1 配对 sign test → K1/K3 在公平框架下的最终判定
2. 若 c1 在 L2/L3 仍≈0:K1 的"伤害"只在 L1 与 SciVB,论文叙事按此收敛
3. 因果图提案的三个试点(oracle-72B 须 gold 接地重建 + 稀疏注入 + 后验弃答打 ExpVid mc hedge 池)
   全部走统一框架
