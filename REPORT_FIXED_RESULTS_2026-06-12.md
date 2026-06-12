# 修复后结果报告(Fixed-Harness Results Report)

**日期**:2026-06-12 · **代码版本**:commit `ecfab199` 起(六处 harness bug 修复)
**评测契约**:`scripts/unified_harness.py`(统一框架:同 uid 题集、每题一次 32 帧抽取、同 BUILDERS、
greedy、修复后解析/打分;条件间只差一个声明变量)
**审计背景与 bug 细节**:见 `HARNESS_AUDIT_2026-06-11.md`;方法路线:见 `V10_REASONING_GRAPH_PLAN.md`

---

## 1. 摘要

对全部评测代码审计后确认并修复 6 处 bug(MC 解析器、fitb 打分器、CoT 抽取/截断、
置信门、SciVB sample_id 碰撞),随后在统一公平框架下重跑 72B 全量矩阵
(ExpVid L2/L3 745 + SciVideoBench 218,条件 c0/cot/c1,另跑独立进程 rep2 测噪声底)。

**三个 headline 结论(修复后口径):**

1. **ExpVid L2/L3 上一切干预皆零**:CoT +0.99pp(p=0.95)、笔记注入 +0.18pp(p=0.24),
   均在噪声内。旧"CoT 大幅伤害 72B"在该基准上是 harness 伪迹。
2. **SciVideoBench 上自由文本伤害为真且通道无关**:CoT −8.72pp(p=0.020)与笔记注入
   −8.72pp(p=0.00031)**逐点同伤**,而 rep2 噪声底为字面零(218 题零翻转)。
   模型自己生成的推理文本 = 外部注入的笔记文本,共享同一条 prior-override 失败通道。
3. **CoT 在 ExpVid 上是任务结构化的,不是均匀的好/坏**:steppred ×3(0.041→0.124)、
   seqgen +5.2pp 对冲 mc −3.0pp、fitb −2.9pp,净效应为零。

## 2. 修复清单(详见审计文档 §2)

| 位置 | 问题 → 修复 |
|---|---|
| `parse_mc_aj`(两处) | 取全文第一个字母,从 '60°C'/'N/A' 抓错 → 分层解析(marker 后 > 裁决行自底向上 > 防护兜底) |
| `score_fitb` | 只认 `\|` 分隔(0/137 条输出含管道,第 2+ 空全记 0)→ 多分隔符 + 非数字逗号 |
| `extract_final` | 漏 'EXACT ANSWER:';'N/A'→选项 A → 全 marker 变体 + refusal→空 |
| CoT/direct token 预算 | 768/32 截断 → 1536/64 + fitb 提示加分隔符说明 |
| 置信门 | 解析失败默认 0.0 被当低置信触发 → `conf_parsed` 区分无信号 |
| SciVB 数据 | 65 个 sample_id 对应两道不同题 → `uid = sample_id#sha1(question)[:8]` |

## 3. 修复口径重打分历史结果(`tools/rescore_fixed.py`)

| 结果集 | stored → fixed(overall) | 说明 |
|---|---|---|
| 72B CoT-C0 全集 ExpVid (745) | 0.337 → **0.358** | fitb 0.012→0.127(分隔符修复) |
| 72B CoT-C0 全集 SciVB (218) | 0.349(不变) | 全 MC,不受 fitb 影响 |
| CoT 消融 72B ExpVid (80) | cot .337→.393 / direct .394→.444 | 差距 −5.1pp;已被统一矩阵取代 |
| C0-vs-C1 sweep(重解析) | 数字不变 | L2/3 −0.34 (p=.24) / L1 −4.19 (p≈0) / SciVB −10.55 (p=2e-5) |
| oracle-KG (n=10) | c0 .483 / auto .222 / oracle .200 | 实验因选样偏倚+非真 oracle 作废 |

## 4. 统一框架 72B 终审(2026-06-11 跑完,全量,修复后代码)

### 4.1 主表

| | ExpVid L2/L3 (n=745) | SciVideoBench (n=218) |
|---|---|---|
| **c0**(基线) | **0.3496** | **0.4174** |
| **cot** | 0.3595(+0.99pp,sign-p=0.95,零) | 0.3303(**−8.72pp,p=0.020**) |
| **c1**(注入笔记) | 0.3513(+0.18pp,p=0.24,零) | 0.3303(**−8.72pp,p=0.00031**) |
| **rep2 噪声底** | −0.15pp;12/745 题翻转(1.6%) | **0.00pp;0/218 题翻转** |

### 4.2 ExpVid 分任务(c0 / cot / c1)

| task | n | c0 | cot | c1 |
|---|---|---|---|---|
| mc(ordering+verification) | 302 | 0.477 | 0.447 | 0.467 |
| seqgen | 161 | 0.446 | **0.498** | 0.465 |
| steppred | 145 | 0.041 | **0.124** | 0.069 |
| fitb | 137 | 0.282 | 0.253 | 0.262 |

### 4.3 判读

- 测量学:greedy 全管线近乎确定(SciVB 重跑零翻转;ExpVid 1.6% 翻转来自帧抽取边界),
  ≥1pp 的效应可读;963 条 cot 输出 0 条缺 FINAL ANSWER 标记(旧 harness:7B seqgen 16/20 缺)。
- 旧结论对照:

| 旧结论(坏口径) | 修复后终判 |
|---|---|
| C1 注入伤 72B −4.2pp(跨基准拼接) | L2/3 **零**;SciVB −8.72 真;L1 −4.19 真(历史 sweep,布线已审计为净) |
| CoT 伤 72B −5.7pp(截断+解析污染) | ExpVid **零**(任务结构化);SciVB −8.72 真 |
| 完美 oracle-KG −33pp | 作废(选样回归+非真 oracle+解析 bug) |
| V9 KG/时序边有害/无用 | 绝对值作废;边消融≈0 仍立 |
| 72B 过度自信、弃答门有效 | 过度自信为真(emitted 均值 4.93);"弃答门"实为解析失败门,弃答路径从未被测 |

## 5. 7B 重跑(进行中)

7B × {ExpVid, SciVB} × {c0, cot, c1} + rep2 已于 2026-06-12 在同一契约下启动
(`scripts/run_unified_7b.sh`,GPU 0/1)。完成后本节补全:7B 主表 + **跨尺度 interaction**
(公平口径下"注入帮弱伤强"是否成立)——这是规模反转主张的最终检验。

## 6. 复现

```bash
# 72B(双实例,~6h)         # 7B(双实例,~4-5h)
bash scripts/run_unified_72b.sh
bash scripts/run_unified_7b.sh
# 历史结果修复口径重打分
python -m tools.rescore_fixed
# 聚合:results_unified/72b_*_{main,rep2}_chunk*.jsonl,配对 sign test 见审计文档 §6
```
