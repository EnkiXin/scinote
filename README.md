# scinote — Scientific-Video QA: 干预何时、为何、以何种形式有效

在 ExpVid L2/L3(745 题)与 SciVideoBench(218 题)上,用 Qwen2.5-VL-7B/72B 系统研究
"文本中间产物干预"(notes / KG / 精确原子 / agent / CoT)对视频问答的影响。
2026-06-11 的 harness 审计推翻了部分历史结论并建立了统一评测契约;**当前真相只看下面四份文档**:

| 文档 | 内容 |
|---|---|
| **[REPORT.md](./REPORT.md)** | 唯一结果汇总:修复后双尺度公平矩阵 + 新旧结论对照(先读这份) |
| [HARNESS_AUDIT_2026-06-11.md](./HARNESS_AUDIT_2026-06-11.md) | 审计:6 处 bug、复现验证、K1-K5 终判 |
| [V10_REASONING_GRAPH_PLAN.md](./V10_REASONING_GRAPH_PLAN.md) | 下一步:reasoning-graph v10 阶梯(感知控制器 + prose-free 裁决器)+ 试点 GATE + ICLR 2027 日历 |
| [deprecated/README.md](./deprecated/README.md) | 已退役内容清单(引用任何旧数字前必读) |

## 复现当前结果

```bash
# 统一公平框架(冻结契约:uid 题集 / 32 帧一次抽取 / greedy / 修复后解析打分)
bash scripts/run_unified_72b.sh   # 72B 矩阵(c0/cot/c1 + rep2 噪声底)
bash scripts/run_unified_7b.sh    # 7B 矩阵
python -m tools.aggregate_unified # 聚合 + 配对 sign test + 跨尺度 interaction
python -m tools.rescore_fixed     # 历史结果按修复口径重打分
```

## 目录地图

- `scripts/unified_harness.py` — 统一评测契约(一切新条件的接入点)
- `evaluate_c0_test_split.py` / `evaluate_unified.py` — 共享 BUILDERS/解析/打分(已修复)
- `protonote/` — 工具/笔记/agent 管线包
- `results_unified/` — 公平矩阵结果(冻结基线 + rep2 噪声底)
- `results_72b_cot_full/`、`results_protonote/`、`results_precision_gate/` — 仍被引用的原始证据
- `docs/` — 历史分析文档(错误分类学、案例研究、各版本报告;结论以审计为准)
- `deprecated/` — 被推翻的文档/脚本/结果
- `archive_results/` — 退役实验线的原始数据存档
