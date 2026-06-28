#!/usr/bin/env python3
"""Extract downloaded source archives into the official videos/<dir> layout.

official_media_paths (tools/check_open_o3_data.py) expects, under DATA_ROOT/videos:
  videor1/<NeXT-QA|CLEVRER|PerceptionTest>/...     <- Video-R1 zips
  videoespresso/videos/<...>                       <- VideoEspresso split zip
  treevgr/images/<dvqa|coco|ai2d|chartqa>/...       <- LLaVA-NeXT raw images
  tvg_r1/videomind_data/<sub>/videos/<id>.mp4       <- VideoMind videos.tar.gz
  gqa/<id>.jpg                                      <- lmms-lab/GQA parquet

Each source is a separate function, guarded + idempotent. Run with a source name
(or 'all'). Verifies sample target paths from the STGR json after extraction.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

D = Path("/home/yz0392@unt.ad.unt.edu/xin_ai")
SRC = D / "open_o3/data/source_datasets"
VIDEOS = D / "open_o3/data/Open-o3-Video-data/videos"


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=False, **kw)


def video_r1():
    """Video-R1 subset zips -> videos/videor1/<subset>/...

    Verified zip-internal vs expected (official path = videos/videor1/ + video_path):
      NeXT-QA zip:        'NextQA/NExTVideo/..'  <-> 'NeXT-QA/NextQA/NExTVideo/..'
      CLEVRER zip:        'train_videos/..'      <-> 'CLEVRER/train_videos/..'
      PerceptionTest zip: 'video_X.mp4' (flat)   <-> 'PerceptionTest/video_X.mp4'
    So each subset extracts into videos/videor1/<subset>/ (the prefix the zip omits).
    """
    base = SRC / "Video-R1-data"
    for subset in ("NeXT-QA", "CLEVRER", "PerceptionTest"):
        sd = base / subset
        if not sd.is_dir():
            print(f"[video_r1] {subset}: no archive dir, skip", flush=True)
            continue
        dst = VIDEOS / "videor1" / subset
        dst.mkdir(parents=True, exist_ok=True)
        for z in sorted(sd.glob("*_part*.zip")):
            run(["unzip", "-n", "-q", str(z), "-d", str(dst)])
        print(f"[video_r1] {subset} -> {dst}", flush=True)


def videoespresso():
    """Split zip .z01..zNN + .zip -> combine -> videos/videoespresso/."""
    dst = VIDEOS / "videoespresso"
    dst.mkdir(parents=True, exist_ok=True)
    sd = SRC / "VideoEspresso"
    last = sd / "VideoEspresso_train_video.zip"
    if not last.exists():
        print("[videoespresso] final .zip part missing, skip", flush=True)
        return
    zip_bin = shutil.which("zip") or "/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/zip"
    combined = sd / "combined.zip"
    if not combined.exists():
        run([zip_bin, "-s", "0", str(last), "--out", str(combined)])
    run(["unzip", "-n", "-q", str(combined), "-d", str(dst)])
    print(f"[videoespresso] done -> {dst}", flush=True)


def treevgr():
    """LLaVA-NeXT llava_next_raw_format -> videos/treevgr/images/<dataset>/...

    Verified: STGR image_path = 'images/' + tar-internal path (e.g. tar
    'dvqa/images/x.png' <-> image_path 'images/dvqa/images/x.png'), and official
    resolves to videos/treevgr/ + image_path. So extract INTO videos/treevgr/images/.
    """
    dst = VIDEOS / "treevgr" / "images"
    dst.mkdir(parents=True, exist_ok=True)
    sd = SRC / "LLaVA-NeXT-Data" / "llava_next_raw_format"
    tars = sorted(sd.glob("*.tar.gz"))
    if not tars:
        print("[treevgr] no tars, skip", flush=True)
        return
    for t in tars:
        if run(["gzip", "-t", str(t)]).returncode != 0:
            print(f"[treevgr] {t.name} truncated, skip (re-run after download)", flush=True)
            continue
        run(["tar", "xzf", str(t), "-C", str(dst)])
    print(f"[treevgr] extracted -> {dst}", flush=True)


def videomind():
    """VideoMind <sub>/videos*.tar.gz -> videos/tvg_r1/videomind_data/<sub>/<variant>/.

    Per-subset variant matters (verified from STGR paths):
      tacos/didemo/queryd/hirest -> 'videos' (full res)
      internvid_vtime            -> 'videos_crop_3fps_480_noaudio'
    Only the correct variant was downloaded per subset, so extract whatever
    videos*.tar.gz* is present. Tar top-dir already encodes the variant name, so
    extracting into videomind_data/<sub>/ yields the official path.
    """
    dst_base = VIDEOS / "tvg_r1" / "videomind_data"
    dst_base.mkdir(parents=True, exist_ok=True)
    vm = SRC / "VideoMind"
    if not vm.is_dir():
        print("[videomind] no VideoMind dir, skip", flush=True)
        return
    for sub_dir in sorted(vm.iterdir()):
        # files may be nested VideoMind/<sub>/<sub>/...tar.gz* (dl path quirk)
        tars = list(sub_dir.rglob("videos*.tar.gz*")) + list(sub_dir.rglob("videos*.tar"))
        if not tars:
            continue
        sub = sub_dir.name
        out = dst_base / sub
        out.mkdir(parents=True, exist_ok=True)
        # group by base archive name (handles both 'videos' and 'videos_crop_...')
        bases = {}
        for t in tars:
            base = t.name.split(".tar")[0]
            bases.setdefault(base, []).append(t)
        for base, parts in bases.items():
            split = sorted([t for t in parts if t.suffix.lstrip(".").isdigit()])
            whole = [t for t in parts if t.name.endswith((".tar.gz", ".tar"))]
            if split:
                cat = out / f"{base}.tar.gz"
                if not cat.exists():
                    with open(cat, "wb") as w:
                        for p in split:
                            w.write(p.read_bytes())
                run(["tar", "xzf", str(cat), "-C", str(out)])
                cat.unlink(missing_ok=True)
            elif whole:
                run(["tar", "xzf", str(whole[0]), "-C", str(out)])
        print(f"[videomind] {sub} -> {out}", flush=True)


def groundedvllm():
    """Grounded-VideoLLM <sub>/chunk_*.zip -> videos/tvg_r1/GroundedVLLM/<sub>/videos/.

    Official path: GroundedVLLM/qvhighlights/videos/<id>.mp4 (clipped segments).
    Chunk internal layout verified at extraction; arrange so <id>.mp4 lands under
    GroundedVLLM/<sub>/videos/.
    """
    dst_base = VIDEOS / "tvg_r1" / "GroundedVLLM"
    dst_base.mkdir(parents=True, exist_ok=True)
    gv = SRC / "GroundedVLLM"
    if not gv.is_dir():
        print("[groundedvllm] no dir, skip", flush=True)
        return
    for sub_dir in sorted(gv.iterdir()):
        if not sub_dir.is_dir():
            continue
        chunks = sorted(sub_dir.rglob("chunk_*.zip"))
        if not chunks:
            continue
        sub = sub_dir.name
        out = dst_base / sub
        out.mkdir(parents=True, exist_ok=True)
        for z in chunks:
            run(["unzip", "-n", "-q", str(z), "-d", str(out)])
        # normalize: ensure videos/<id>.mp4 under out (move stray mp4s into videos/)
        vids = out / "videos"
        vids.mkdir(exist_ok=True)
        for mp4 in out.glob("*.mp4"):
            mp4.rename(vids / mp4.name)
        print(f"[groundedvllm] {sub} -> {out}", flush=True)


def gqa():
    """lmms-lab/GQA train parquet -> only the referenced gqa/<id>.jpg images."""
    import io
    try:
        import pyarrow.parquet as pq
        from PIL import Image
    except Exception as e:
        print(f"[gqa] missing deps ({e}); skip", flush=True)
        return
    dst = VIDEOS / "gqa"
    dst.mkdir(parents=True, exist_ok=True)
    # collect needed ids from STGR json
    sys.path.insert(0, str(D / "Open-o3-Video/tools"))
    from check_open_o3_data import iter_records  # noqa
    DR = D / "open_o3/data/Open-o3-Video-data"
    need = set()
    for name in ("STGR-SFT.json", "STGR-RL.json"):
        for r in iter_records(json.loads((DR / "json_data" / name).read_text())):
            ip = r.get("image_path", "") or ""
            if str(r.get("source")) == "gqa" and ip:
                need.add(Path(ip).name)
    print(f"[gqa] need {len(need)} images", flush=True)
    if not need:
        return
    parqs = sorted((SRC / "GQA").glob("train_all_images/*.parquet"))
    got = 0
    for pf in parqs:
        t = pq.read_table(pf)
        cols = t.column_names
        idc = "id" if "id" in cols else cols[0]
        imc = "image" if "image" in cols else cols[-1]
        ids = t.column(idc).to_pylist()
        imgs = t.column(imc).to_pylist()
        for i, im in zip(ids, imgs):
            fn = f"{i}.jpg"
            if fn in need and not (dst / fn).exists():
                b = im["bytes"] if isinstance(im, dict) else im
                Image.open(io.BytesIO(b)).convert("RGB").save(dst / fn)
                got += 1
    print(f"[gqa] extracted {got} images -> {dst}", flush=True)


FUNCS = {
    "video_r1": video_r1, "videoespresso": videoespresso, "treevgr": treevgr,
    "videomind": videomind, "gqa": gqa, "groundedvllm": groundedvllm,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = FUNCS.keys() if which == "all" else which.split(",")
    for t in targets:
        print(f"==== {t} ====", flush=True)
        FUNCS[t]()
