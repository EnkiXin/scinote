"""Stage A of the oracle-KG experiment: use 72B to convert each sample's
GROUND-TRUTH oracle_note (+ frames) into a high-quality V9-schema KG.

The resulting "oracle KG" is rendered by the SAME Stage-4 renderer and
answered by the SAME 7B model as the auto-KG, so the only variable in the
C2-vs-C_oracle comparison is KG CONTENT QUALITY (form held constant).

Output: tools/oracle_kgs.json  {sample_id: {entities:[...], operations:[...]}}

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python -m scripts.v9_build_oracle_kg
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEL = ROOT / "tools" / "oracle_kg_sample_ids.json"
OUT = ROOT / "tools" / "oracle_kgs.json"

PROMPT = """You are building a high-quality ORACLE knowledge graph of a scientific \
experiment video. You are given (a) the video frames and (b) a GROUND-TRUTH \
step-by-step description. The description is authoritative for what happens; \
use the frames for visual/timing grounding.

Output STRICT JSON only (no prose, no markdown fences):
{{
  "entities": [
    {{"entity_id":"Entity_1","canonical_name":"<correct concise identity>",
      "type":"<Operator|Instrument|Container|Material|Display|Measurement>",
      "core_role":"<starting_material|tool|intermediate_product|final_product|control|experimental|null>",
      "states":[{{"state_id":"Entity_1_s1","time_interval":[0,5],
        "visual_features":"<state/appearance>",
        "lifecycle_status":"<active|consumed|transformed|merged|split>",
        "transmuted_to_entity_ids":[],"transmuted_from_entity_ids":[]}}]}}
  ],
  "operations": [
    {{"operation_id":"Op_1","action":"<verb phrase>",
      "action_category":"<mixing|heating|cooling|centrifuging|measuring|observing|transferring|preparing|incubating|weighing|titrating|other>",
      "timestamp":0,"duration":null,
      "input_states":["<state_id used/consumed>"],
      "output_states":["<state_id produced/resulting>"],
      "description":"<one line>"}}
  ]
}}

Rules:
- DEDUPLICATE: identical peers that undergo the same operations = ONE entity
  (canonical_name like "5 PCR tubes"). Never list many identical objects separately.
- CORRECT identities only; no hallucination. If unsure, use a correct generic
  name (e.g. "reagent solution") rather than guessing a specific product.
- Operations MUST wire the material/causal flow: an operation that produces a
  new state lists it in output_states, and the next operation consuming it lists
  that same state_id in input_states. This makes the causal chain explicit.
- Order operations by timestamp (use the step order; spread timestamps over the video).
- Output ONLY the JSON object.

GROUND-TRUTH step description:
{note}
"""


def parse_json(raw: str):
    s = raw.strip()
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # grab outermost braces
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        s = s[i:j + 1]
    return json.loads(s)


def main():
    from evaluate_c0_test_split import extract_frames
    from protonote.data.loaders import load_test_split, resolve_video_path
    from protonote.v6.llm_client import QwenVL72BClient

    sel = json.load(open(SEL))
    id_set = set(sel["ids"])
    items = {it["sample_id"]: it
             for it in load_test_split(benchmark="expvid", limit=None)
             if it.get("sample_id") in id_set}
    print(f"[oracle] {len(items)} target samples", flush=True)

    vlm = QwenVL72BClient(model_name="Qwen/Qwen2.5-VL-72B-Instruct", device="auto")
    print("[oracle] 72B ready", flush=True)

    out = {}
    t0 = time.time()
    for k, sid in enumerate(sel["ids"]):
        item = items.get(sid)
        if not item:
            print(f"  [skip] {sid} not found"); continue
        try:
            vp = resolve_video_path(item)
            frames = extract_frames(vp, max_frames=16)
            note = item.get("oracle_note") or ""
            raw = vlm.generate_video(PROMPT.format(note=note), frames,
                                     max_tokens=3000, temperature=0.0)
            kg = parse_json(raw)
            n_e = len(kg.get("entities", []))
            n_o = len(kg.get("operations", []))
            out[sid] = {"entities": kg.get("entities", []),
                        "operations": kg.get("operations", []),
                        "raw_72b": raw[:4000]}
            print(f"  [{k+1}/{len(sel['ids'])}] {sid[-38:]} -> "
                  f"entities={n_e} ops={n_o}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback
            out[sid] = {"error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()[-600:]}
            print(f"  [{k+1}] {sid[-38:]} ERROR: {e}", flush=True)
        json.dump(out, open(OUT, "w"), indent=2, default=str)
    print(f"[oracle] saved -> {OUT}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
