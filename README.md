# EEG spectral entropy modulation — replication analysis

Replication of the statistical analyses reported in:

> Molina V, Lubeiro A, de Luis Garcia R, Gomez-Pilar J, et al. (2020).
> *Deficits of entropy modulation of the EEG: A biomarker for altered function
> in schizophrenia and bipolar disorder?* **J Psychiatry Neurosci**.
> DOI: 10.1503/jpn.190032

The script reproduces the paper's results from the study dataset
(`articulo_SE_nuevo_081118.xlsx`): factor analysis of spectral entropy (SE)
modulation, group comparisons (ANCOVA + post-hoc), treatment-effect analyses,
structural connectivity (DTI) analyses, and clinical/cognitive correlates.

## Requirements

- Python 3.10+ (developed and tested on Python 3.14)
- The packages listed in `requirements.txt`:
  `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `openpyxl`

## Installation

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Input files

| File | Role |
|---|---|
| `articulo_SE_nuevo_081118.xlsx` | Study data (sheet `Datos`, 156 participants × 242 variables). Must sit next to the script. |
| `Diccionario_interpretativo_entropia_EEG.md` | Interpretation of each variable (reference only; not read by the script). |
| `Deficit SE _JPN_2020.pdf` | Published paper whose results are replicated (reference only). |

Key variables used (see the dictionary for details):

- `sujeto3_31102018` — diagnostic group: 1 = chronic schizophrenia (n = 49),
  3 = first-episode schizophrenia (n = 30), 6 = bipolar disorder (n = 29),
  4 = control (n = 48).
- `*_b` / `*_r` — spectral entropy per electrode in the baseline (−300–0 ms)
  and response (150–450 ms) windows; `Mod_Se_*` / `Mod_Sec_*` = response −
  baseline (28 sensors; verified exact in the script).
- `FAC1_1`–`FAC4_1` — stored SPSS factor scores of SE modulation (used for
  inference, as in the paper).
- `CLC_dti`, `PL_dt`, `SW_dt`, `D_dt` — DTI graph parameters (n = 80).
- Clinical/cognitive: `panss_*`, `bacs_*`, `wais_citT1`, `msceit_citT1`,
  `HAM_cT1`, `equiv_cpz`, `Duracion`, medication flags (`Bzd`, `litio`,
  `antidepre`, `BIP_CON_SIN_ap`).

## Run

```bash
source .venv/bin/activate
python entropy_eeg_analysis.py
```

No arguments are needed; the script reads `articulo_SE_nuevo_081118.xlsx` from
its own folder and writes everything to `results/`.

## What it does

1. **Integrity check** — confirms `Mod_*` = response − baseline for every
   subject and sensor.
2. **Factor analysis** — PCA on the correlation matrix (no rotation) of the 28
   SE-modulation columns; standardized first-factor scores. Repeated for
   pre-stimulus SE (29 sensors). Compares the recomputed factor with the stored
   `FAC1_1` (r ≈ 0.998) and uses the stored scores for inference.
3. **Table 1** — demographic, clinical and cognitive descriptives per group with
   one-way ANOVA and a sex χ² test.
4. **Main ANCOVA** — SE-modulation factor ~ group + age (η² effect size), plus
   age and group × age tests.
5. **Post-hoc** — Welch t-tests of each patient group vs controls (Bonferroni ×3).
6. **Pre-stimulus SE** — group effect on the baseline factor (null-result check).
7. **Treatment effects** — Pearson r with CPZ dose and illness duration;
   Mann-Whitney for bipolar ± antipsychotic, lithium, benzodiazepines and
   antidepressants.
8. **Structural connectivity** — ANCOVA of small-world index (`SW_dt`) and
   connectivity strength (`D_dt`) ~ group + age, post-hocs, and Pearson r with
   the SE-modulation factor (all / patients / schizophrenia).
9. **Clinical & cognitive correlates** — OLS regressions of PANSS, BACS, MSCEIT
   and Hamilton scores on the SE-modulation factor, separately in schizophrenia
   and bipolar groups.
10. **Figures** — group factor scores with 95% CIs, regional modulation CIs, and
    treatment panels.

## Outputs (`results/`)

| File | Content |
|---|---|
| `table1_descriptives.csv` | Table 1 descriptives + ANOVA per variable |
| `posthoc_modulation_vs_control.csv` | Post-hoc t-tests vs controls |
| `dti_posthoc.csv` | DTI pairwise comparisons vs controls |
| `dti_vs_modulation_corr.csv` | DTI ↔ SE-modulation correlations |
| `clinical_cognitive_regressions.csv` | Clinical/cognitive regression results |
| `factor_scores.csv` | Per-subject factor scores (recomputed + stored) |
| `fig1_factor_scores_by_group.png` | Factor scores by group (Fig. 1 style) |
| `fig3_modulation_CI_by_region.png` | Modulation 95% CI by region (Fig. 3 style) |
| `fig4_treatment_effects.png` | Treatment panels (Fig. 4 style) |

The console prints every statistic next to the published value for quick
comparison.

## Fidelity notes

- Group sizes, sex distribution, age, Table 1 descriptives, the main group
  effect on SE modulation (F ≈ 5.5, p ≈ 0.001, η² ≈ 0.10), the null
  pre-stimulus effect, the treatment analyses and the clinical/cognitive
  regressions reproduce closely from this file.
- The paper's factor analysis was run on 235 subjects (127 controls from the
  full EEG database); only the 156 study participants are in this file, so
  recomputed PCA eigenvalues/variance differ slightly (55.4% vs 56.3%).
- Median frequency (Fig. 2 / Table S2) cannot be recomputed: the file contains
  only SE values, not spectra.
- The DTI age effect reported in the paper (F = 0.768 / 0.005, n.s.) does not
  reproduce from this file (age is strongly related to `SW_dt`/`D_dt` here);
  the group effect on connectivity strength reproduces in direction and
  significance. The paper's DTI model likely differed (e.g., covariates or
  sample).
