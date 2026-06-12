# deprecated/ — 审计后退役的文档、脚本与结果(2026-06-12)

本目录内容的结论已被 `../HARNESS_AUDIT_2026-06-11.md` 推翻或被统一框架
(`../scripts/unified_harness.py` + `../REPORT_FIXED_RESULTS_2026-06-12.md`)取代。
**引用任何此处数字前先读审计文档。**

| 项 | 退役原因 |
|---|---|
| PRECISION_GATE_RESULTS.md | headline 统计欠功效(n=75,McNemar p=0.219);Gate-C 实为解析失败门;布线干净的原始数据仍在 ../results_precision_gate/ |
| CAUSAL_EDGE_PLAN.md | LOG 中 oracle −33pp 等数字被推翻(选样回归+非真 oracle+解析 bug);路线由 ../V10_REASONING_GRAPH_PLAN.md 取代 |
| V9_RESEARCH_PLAN.md / V8_RESEARCH_PLAN_PATCH.md | V8/V9 注入线关闭 |
| EXPERIMENT_STATUS.md / EXECUTION.md | 2026-05-22 状态快照,已过期 |
| scripts/v9_*.py | 坏 harness(CoT prompt 配简答 scorer、MC 首字母解析、oracle 非 gold 接地);任何重做须走统一框架 |
| results_cot_ablation/ | 旧 CoT 消融原始输出(768-token 截断 + 旧解析);修正重打分见 ../results_rescore_fixed/;被统一矩阵取代 |
| results_protonote_v9/ | V9 probe/oracle 输出(坏 harness);实验作废 |
