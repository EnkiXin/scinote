"""V8: Two-stage entity grounding paradigm for scientific video understanding.

Replaces V6 ReAct loop with 4-stage pipeline:
  Stage 1: Abstract KG extraction with confidence
  Stage 2: Confidence-based routing
  Stage 3: Selective grounding (4 paths)
  Stage 4: KG-based reasoning

See PROTONOTE_V8_FINAL_PLAN_V2.md for full design.
"""
