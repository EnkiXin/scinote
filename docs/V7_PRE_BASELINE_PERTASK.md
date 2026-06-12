# V7 Pre-baseline per-task analysis (P0.4)

Cross-tab of 72B `pure_c0` vs 72B `v6_react` on identical sample sets,
followed by category-level diagnosis of HURT cases to predict which
V7 fixes will recover them. Pure analysis — no GPU used.

- **HURT** = C0 correct but v6_react wrong (V7 abstain / rewriter target)
- **SAVED** = v6_react correct but C0 wrong (tool calls helped here)

---

## 1. ExpVid L2/L3 — partial baseline coverage

Only `sequence_generation` (n=141) has a full 72B C0 baseline; the
remaining 604 items lack a v5 8-cond run on this 745-item subset.

| Task | n | C0 acc | v6_react acc | Δ (pp) | HURT | SAVED | Both ✓ | Both ✗ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sequence_generation | 141 | 6.4 | 7.8 | +1.4 | 0 | 0 | 7 | 134 |

**Takeaway**: `sequence_generation` is so hard for both models (intrinsic
ceiling near 8 %) that V7 will not move the needle here. We need C0
baselines on the other 5 ExpVid tasks before drawing strong V7
conclusions on ExpVid.

---

## 2. SciVideoBench — full alignment

| Task | n | C0 acc | v6_react acc | Δ (pp) | HURT | SAVED | Both ✓ | Both ✗ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mc | **218** | **35.78** | **32.11** | **−3.67** | **33** | **25** | **45** | **115** |

Net loss = 33 − 25 = 8 cases = 3.67 pp.

---

## 3. HURT diagnosis on SciVB (33 cases)

For each HURT case we extracted: notes content, action trace, retrieve
return-pass count, and duplicate-query signals. Categorization by which
V7 fix would trigger:

| Category | n | % | V7 fix that triggers |
|---|---:|---:|---|
| `no_kb_used` | 17 | 51.5 % | tool-selection guide (P1.2), confidence/abstain (P1.1/1.3) |
| `other` | 6 | 18.2 % | nothing automatic — content-level reasoning failure |
| `notes_unreliable` | 5 | 15.2 % | **auto-abstain (P1.1)** |
| `kb_all_empty` | 4 | 12.1 % | **query rewriter (P2.1)** + auto-abstain (P1.1) |
| `dup_queries` | 1 | 3.0 % | **dedup warning (P2.3)** |

Directly mechanically-recoverable HURT (categories the V7 fixes
trigger on without planner cooperation): **10 / 33 = 30.3 %**.

### Best-case acc projection (V7 vs V6 on SciVB n=218)

| Recovery rate | V7 acc | Δ vs v6_react | Δ vs 72B C0 |
|---|---:|---:|---:|
| v6_react base | 32.11 % | 0 | −3.67 |
| 50 % of recoverable | 34.40 % | +2.29 | −1.38 |
| 100 % of recoverable | 36.70 % | +4.59 | **+0.92** |

To break above 72B C0 (35.78), V7 needs to recover **~85 %** of the 10
directly-recoverable HURT cases AND keep the existing 25 SAVED cases.
The remaining 23 HURT cases require the tool-selection guide and
confidence-based answer to lift performance further.

---

## 4. Implications for V7 design

1. **Abstain mechanism (P1.1) is necessary but not sufficient.**
   ~30 % of HURT cases are mechanically catchable; the rest need
   better tool selection (planner believed in plausible-looking
   but wrong visual notes).
2. **Tool-selection guide (P1.2) is the high-leverage lever.**
   17 / 33 HURT cases used no KB at all yet the question required
   domain knowledge. A decision-tree prompt that pushes those to
   `retrieve` (or to `abstain` when confidence is low) is critical.
3. **Query rewriter (P2.1) addresses 4 cases** where every retrieve
   returned no passages — likely BM25/BGE missing because the planner
   used raw question text instead of protocol-style query.
4. **Dedup warning (P2.3) addresses only 1 case** — minor lever.
5. **ExpVid analysis is incomplete.** Only `sequence_generation` has
   a 72B C0 baseline on the 745-item set. Either run C0 on the
   remaining 5 tasks, OR rely on v6_react ExpVid 34.38 as an absolute
   number until a properly-aligned baseline lands.

## 5. Sample HURT case sample_ids (for manual inspection)

| Category | Sample IDs |
|---|---|
| `no_kb_used` (17) | `mc_58827_1`, `mc_67076_1`, `mc_67406_2`, … |
| `notes_unreliable` (5) | `mc_66530_4`, `mc_66420_3`, `mc_65956_1` |
| `kb_all_empty` (4) | `mc_65619_2`, `mc_53276_3`, `mc_59909_1` |
| `dup_queries` (1) | `mc_65619_2` |
| `other` (6) | `mc_52456_3`, `mc_55258_4`, `mc_57270_2` |

Reproduction: `python scripts/v7_hurt_diagnosis.py` (no GPU needed).
