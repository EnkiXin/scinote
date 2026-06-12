# SciVB breakdown — V8 7B vs 7B C0 vs 72B C0

SciVB only has one task_type (`mc`); the meaningful axes are
**discipline** (8 fields), **question_type** (3-4 styles), and
**subject** (finer-grained topic). All accuracies are 0/1 MC.

Coverage: V8 218 / 7B C0 218 / 72B C0 218
Source metadata records: 676; unmapped V8 items: 0

## By discipline

### Discipline

| Group | n | V8 | 7B C0 | Δ vs 7B C0 | 72B C0 | Δ vs 72B C0 |
|---|---:|---:|---:|---:|---:|---:|
| Engineering | 53 | 26.42% | 30.19% | -3.77 | 45.28% | -18.87 |
| Chemistry | 44 | 15.91% | 6.82% | +9.09 | 31.82% | -15.91 |
| Biology | 44 | 36.36% | 29.55% | +6.82 | 27.27% | +9.09 |
| Medicine | 36 | 30.56% | 27.78% | +2.78 | 36.11% | -5.56 |
| Biochemistry | 19 | 21.05% | 15.79% | +5.26 | 31.58% | -10.53 |
| Bioengineering | 16 | 18.75% | 18.75% | +0.00 | 43.75% | -25.00 |
| Physics | 6 | 16.67% | 16.67% | +0.00 | 33.33% | -16.67 |
| **TOTAL** | **218** | **25.69%** | **22.48%** | **+3.21** | **35.78%** | **-10.09** |

## By question type

### Question type

| Group | n | V8 | 7B C0 | Δ vs 7B C0 | 72B C0 | Δ vs 72B C0 |
|---|---:|---:|---:|---:|---:|---:|
| Hypothetical Reasoning | 126 | 28.57% | 26.98% | +1.59 | 37.30% | -8.73 |
| Quantitative Reasoning | 64 | 18.75% | 14.06% | +4.69 | 23.44% | -4.69 |
| Conceptual Reasoning | 28 | 28.57% | 21.43% | +7.14 | 57.14% | -28.57 |
| **TOTAL** | **218** | **25.69%** | **22.48%** | **+3.21** | **35.78%** | **-10.09** |

## By subject (top 15 by n)

### Subject (top 15)

| Group | n | V8 | 7B C0 | Δ vs 7B C0 | 72B C0 | Δ vs 72B C0 |
|---|---:|---:|---:|---:|---:|---:|
| Neuroscience | 27 | 48.15% | 40.74% | +7.41 | 25.93% | +22.22 |
| Materials Science | 26 | 23.08% | 23.08% | +0.00 | 53.85% | -30.77 |
| Materials Chemistry | 18 | 16.67% | 0.00% | +16.67 | 33.33% | -16.67 |
| Biochemistry | 16 | 18.75% | 18.75% | +0.00 | 31.25% | -12.50 |
| Bioengineering | 16 | 18.75% | 18.75% | +0.00 | 43.75% | -25.00 |
| Oncology | 15 | 33.33% | 33.33% | +0.00 | 33.33% | +0.00 |
| Nanomaterials | 11 | 9.09% | 0.00% | +9.09 | 36.36% | -27.27 |
| Microfluidics | 10 | 30.00% | 40.00% | -10.00 | 30.00% | +0.00 |
| Cell Biology | 7 | 14.29% | 14.29% | +0.00 | 14.29% | +0.00 |
| Immunology | 5 | 0.00% | 0.00% | +0.00 | 60.00% | -60.00 |
| Semiconductor | 5 | 60.00% | 80.00% | -20.00 | 100.00% | -40.00 |
| Photovoltaics | 4 | 50.00% | 50.00% | +0.00 | 50.00% | +0.00 |
| Dentistry | 3 | 66.67% | 0.00% | +66.67 | 0.00% | +66.67 |
| Structural Biology | 3 | 33.33% | 0.00% | +33.33 | 66.67% | -33.33 |
| Molecular Biology | 3 | 66.67% | 33.33% | +33.33 | 33.33% | +33.33 |
| **TOTAL** | **169** | **28.40%** | **23.67%** | **+4.73** | **38.46%** | **-10.06** |
