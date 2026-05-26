"""STOA (Subject-Tool-Object-Action) vocabulary for V8 Stage 1.

Stage 1 asks a VLM to extract a KnowledgeGraph from a scientific video.
The VLM is fragile to free-form output, so we constrain it with:

  - An enumerated entity-type vocabulary (matches `EntityType` in
    `kg/entity.py`).
  - An enumerated action vocabulary it MUST pick from when emitting
    Operations.
  - A schema description of the JSON envelope it must return.
  - A confidence calibration guide so the VLM produces useful
    initial_confidence values (not just 0.5 for everything).

The constants here are imported by `stages/stage1_extract.py` to build
the actual prompt.
"""

from __future__ import annotations

# ---- Entity vocabulary (mirrors EntityType in kg/entity.py) ----

ENTITY_TYPES = (
    "Operator",      # person performing experiment
    "Instrument",    # equipment (centrifuge, NMR, microscope, …)
    "Container",     # vessel (tube, flask, beaker, IV bag, …)
    "Material",      # substance (buffer, sample, reagent, powder, …)
    "Display",       # display / label / sticker with text
    "Measurement",   # numeric readout entity (often a Display sub-case)
)


# ---- Action vocabulary for Operations ----
#
# Closed set so the LLM doesn't invent verb variants. Inspired by
# JoVE / BioProtocol verb lists. We use a small but expressive core
# set; rare actions can be "use" + a descriptor in the description
# field.
ACTION_VOCAB = (
    "add",          # add reagent / material to a container
    "transfer",     # move material between containers
    "mix",          # stir / shake / pipette-mix
    "heat",         # heat a sample or container
    "cool",         # ice / refrigerate / chill
    "centrifuge",   # spin a sample
    "incubate",     # wait at a controlled temperature
    "filter",       # pass through filter / membrane
    "wash",         # rinse a sample
    "load",         # load sample into instrument
    "insert",       # insert vessel into a holder / instrument
    "remove",       # remove a vessel / cap
    "measure",      # read a value (often paired with a Display)
    "observe",      # passive observation (microscope view, etc.)
    "image",        # acquire an image
    "record",       # record audio / video / instrument trace
    "weigh",        # weigh a sample
    "dispense",     # pipette a liquid out
    "aspirate",     # pipette a liquid in
    "open",         # open a container / lid / drawer
    "close",        # close a container / lid / drawer
    "press",        # press a button / start an instrument
    "turn_on",      # power on an instrument
    "turn_off",     # power off
    "label",        # write/place a label
    "use",          # generic catch-all (only if none above fits)
)


# ---- Confidence calibration guide injected into the prompt ----
CONFIDENCE_GUIDE = """How to assign initial_confidence to an entity:
  0.90-1.00   Clearly visible, label is canonical (e.g. centrifuge,
              NMR spectrometer with text "NMR" on it).
  0.70-0.89   Confident from shape/material but no text confirms it.
  0.40-0.69   Plausible guess; could be confused with similar
              equipment (e.g. "looks like a flask" but unclear which
              flask type).
  0.20-0.39   Type known but identity unclear (e.g. "some glass
              container").
  0.00-0.19   Even the type is uncertain; ROIs may be partial."""


# ---- The actual extraction prompt template ----

STAGE1_SYSTEM_PROMPT = """You are a scientific-video understanding assistant. \
You extract a structured knowledge graph from a chemistry / biology / \
physics experiment video. Output JSON ONLY, no commentary."""


STAGE1_EXTRACTION_PROMPT = """Extract a knowledge graph from this scientific video.

You see a sequence of {n_frames} uniformly-sampled frames covering the entire video (~{duration_sec:.0f} seconds).
{question_block}
Identify:

  (A) ENTITIES: visible objects that play a role in the experiment.
      Pick the type from EXACTLY this set:
          {entity_types}

      For each entity provide:
        - id            : "Entity1", "Entity2", …  (unique)
        - type          : one of the types above
        - features      : 1 sentence describing what you SEE (shape,
                          color, size, distinguishing marks).
        - identity_guess: best guess at what the entity is (e.g.
                          "centrifuge", "NMR tube", "ethanol", "operator
                          in lab coat"). Use generic words if unsure.
        - initial_confidence: float 0.0-1.0. {conf_guide}
        - appearance_intervals: [[start_sec, end_sec], …] when the
                          entity is visible. Estimate from frame
                          positions (frame i ≈ {sec_per_frame:.1f}s).
        - bbox          : OPTIONAL [x1, y1, x2, y2] in pixel coords of
                          the BEST frame (the one where the entity is
                          most visible). Coordinates are in the original
                          frame resolution. If multiple instances of the
                          same entity-type appear, treat them as
                          separate entities (Entity1, Entity2, …).
        - ocr_candidate : true if the entity is a Display / label /
                          screen with text. false otherwise.

  (B) OPERATIONS: atomic actions happening in the video.
      Pick the action from EXACTLY this set:
          {action_vocab}

      For each operation provide:
        - id          : "Op1", "Op2", … (unique)
        - action      : one verb from the action vocabulary above.
        - subject     : Entity id of who/what performs the action
                        (usually an Operator).
        - object      : Entity id of what is acted upon.
        - timestamp   : integer seconds when the action starts.
        - duration    : OPTIONAL integer seconds (omit if instantaneous).
        - confidence  : float 0.0-1.0 (how sure you are that this action
                        actually occurs).
        - description : OPTIONAL natural-language detail (e.g. "5 ml of
                        ethanol added").

Output strict JSON with this exact envelope:

{{
  "entities": [ {{ … }}, … ],
  "operations": [ {{ … }}, … ]
}}

Rules:
  - JSON ONLY. No markdown fences, no preamble, no trailing prose.
  - Every operation's "subject" and "object" MUST be an existing
    entity id, OR the literal string "Entity_Operator" (a singleton
    operator entity that you may add to "entities").
  - Use the closed action vocabulary; if none fits, use "use".
  - If you cannot identify ANY entity reliably, return
    {{"entities": [], "operations": []}}.
"""


# ---- helper to assemble the prompt at call time ----

def build_extraction_prompt(n_frames: int,
                                       duration_sec: float,
                                       question: str | None = None,
                                       confidence_guide: str = CONFIDENCE_GUIDE,
                                       ) -> str:
    """Render STAGE1_EXTRACTION_PROMPT with concrete numbers."""
    if question:
        qblock = (
            f'\nThe downstream question is: "{question.strip()}". '
            f"Prioritize entities relevant to this question, but still "
            f"include any other visible scientific entities.\n"
        )
    else:
        qblock = ""
    sec_per_frame = duration_sec / max(n_frames, 1)
    return STAGE1_EXTRACTION_PROMPT.format(
        n_frames=n_frames,
        duration_sec=duration_sec,
        question_block=qblock,
        entity_types=", ".join(ENTITY_TYPES),
        action_vocab=", ".join(ACTION_VOCAB),
        conf_guide=confidence_guide,
        sec_per_frame=sec_per_frame,
    )
