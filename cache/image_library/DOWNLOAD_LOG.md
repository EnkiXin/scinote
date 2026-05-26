# Image Library Download Log

Date: 2026-05-26

## Vector-LabPics V2

- Source: Zenodo record 4736111
- Direct URLs (use `?download=1` not the `/api/` path; Zenodo IP-rate-
  limits HEAD requests aggressively — keep them few):
    - `LabPicsMedical.zip`    280.8 MB
    - `LabPicsChemistry.zip`  3917.4 MB
- Status:
    - **LabPicsMedical**: ✅ downloaded and extracted (Image.jpg +
      Data.json only; masks under MaterialsAndParts/SemanticMaps were
      skipped at extract to save disk)
    - **LabPicsChemistry**: 🔄 in progress (background `b8fynkw1s`,
      ~1 MB/s, ETA ~60 min)
- Issues:
    - Zenodo API endpoint `/api/records/.../files/.../content` returned
      403 — switch to `/records/.../files/.../?download=1`.
    - Several short IP rate-limit blocks (~60 s); `requests.Session()`
      with `User-Agent: Mozilla/5.0` works once Zenodo allows.

## ChemEq25

- Source: figshare article 29110433 v3
  (ChemistryLabApparatus-25 by Hossain et al.)
- Direct file URL (NOT the `/articles/.../versions/N` async-zip URL,
  which returns 202 then 0 bytes): `https://ndownloader.figshare.com/files/56271119`
- File: `ChemistryLabApparatus-25.rar` 147 MB
- Status: ✅ downloaded + extracted (needs `unrar` — installed via
  conda-forge)
- Contents: 4,599 images across 25 classes (3220 train / 920 valid /
  459 test); YOLO format labels; CC BY 4.0.

## Physics-27

- Source attempted: figshare article 30519122
- Status: ❌ unavailable — article returned 0 files via API and the
  HTML article page contains no download links. Possibly retracted /
  privatized.
- **Per plan fallback: skip Physics. Use only Chemistry datasets.**

## Manifest Build Results (first pass, no LabPicsChemistry yet)

| Dataset            | images | rows | unmapped |
|--------------------|------:|-----:|---------:|
| chemeq25           | 4,599 | 6,330 |        0 |
| labpics_medical    | 1,215 | 3,005 |        0 |
| labpics_chemistry  |     0 |     0 |        — (still downloading) |
| **TOTAL so far**   | **5,814** | **9,335** |        0 |

After LabPicsChemistry finishes the totals should increase to roughly
~14-16K manifest rows.

## Top labels (current manifest)

```
1170  liquid                  (Material)
 775  round-bottom flask 1/2/3-neck  (Container)
 714  test tube               (Container)
 422  beaker                  (Container)
 421  pipette                 (Instrument)
 403  Erlenmeyer flask        (Container)
 391  IV bag                  (Container)
 338  calorimeter             (Instrument)
 315  separating funnel       (Container)
 302  burette stand           (Instrument)
```

Entity-type distribution: 4518 Container / 2760 Instrument / 1282 Material.

## Discipline coverage (best estimate)

| Discipline   | Coverage | Why |
|--------------|---------|-----|
| Chemistry    | ✅ Good | Vector-LabPics + ChemEq25 |
| Medical (IV/blood) | ✅ Some | LabPicsMedical |
| Biology      | ⚠️ Partial | Some overlap from generic vessels |
| Physics      | ❌ None | Physics-27 unavailable |
| Engineering  | ❌ None | |
| Earth sci.   | ❌ None | |

Plan-§ for "expected ExpVid coverage 40-50 %" assumed Physics-27 worked.
With Physics dropped, realistic ExpVid coverage estimate: **30-45 %**
(chemistry-heavy tasks should still benefit; physics-heavy tasks
won't).

## Disk usage

```
cache/image_library/raw/
  chemeq25/                     288 MB  (.rar + extracted)
  vector_labpics/               ~652 MB so far (Medical extracted) → +3.9 GB when Chemistry done
```
