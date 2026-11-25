#!/usr/bin/env python3
"""
Replication script for:
  Molina V, Lubeiro A, de Luis Garcia R, Gomez-Pilar J, et al. (2020)
  "Deficits of entropy modulation of the EEG: A biomarker for altered function
   in schizophrenia and bipolar disorder?" J Psychiatry Neurosci.

Input : articulo_SE_nuevo_081118.xlsx  (sheet "Datos", 156 participants x 242 vars)
        Variable meanings: Diccionario_interpretativo_entropia_EEG.md
Output: console report + CSV tables + PNG figures in ./results/

Pipeline (mirrors the paper's Methods):
  1. Spectral entropy (SE) per sensor, baseline (-300..0 ms) and response
     (150..450 ms) windows; modulation = response - baseline (already in data).
  2. Factor analysis (PCA on the correlation matrix, no rotation) on the 28
     modulation columns -> first factor score per subject (SPSS-style
     regression factor scores, standardized). Same for pre-stimulus SE.
  3. Univariate GLM (ANCOVA) of factor scores ~ group + age; group x age
     interaction tested separately. Eta-squared effect sizes.
  4. Post-hoc Welch t-tests vs controls (Bonferroni x3).
  5. Treatment effects: Pearson r (CPZ dose, illness duration), Mann-Whitney
     (bipolar +/- antipsychotic, lithium, benzodiazepines, antidepressants).
  6. Structural connectivity (DTI subsample, n=80): ANCOVA of small-world index
     (SW_dt) and connectivity strength (D_dt) ~ group + age; Pearson r between
     these and the SE-modulation factor.
  7. Clinical/cognitive correlates: OLS regressions of PANSS / BACS scores on
     the SE-modulation factor, separately in schizophrenia and bipolar groups.

Notes on fidelity to the published numbers:
  - Group sizes, sex distribution, age, Table 1 descriptives, the main group
    effect on SE modulation, the null pre-stimulus effect, the treatment
    analyses and the clinical/cognitive regressions reproduce closely from
    this file (see comments with the published values next to each test).
  - The published factor analysis was run on 235 subjects (127 controls from
    the full EEG database); only 156 are in this file, so PCA eigenvalues/
    variance here (55.4% for modulation) differ slightly from the paper
    (56.3%, eigenvalue 17.77). The stored factor scores FAC1_1..FAC4_1 in the
    file are the original SPSS scores and are used for the inferential stats
    when present (they are an exact linear combination of the 28 modulations).
  - Median frequency (Fig. 2 / Table S2) cannot be recomputed: the file has no
    spectral data, only SE values.
  - The paper's DTI age effect (F = 0.768 / 0.005, n.s.) does NOT reproduce
    from this file (age is strongly related to SW_dt/D_dt here). The group
    effects on connectivity strength reproduce in direction and significance.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_FILE = "articulo_SE_nuevo_081118.xlsx"
DATA_SHEET = "Datos"
OUT_DIR = "results"

GROUP_COL = "sujeto3_31102018"
GROUP_LABELS = {
    1: "Chronic schizophrenia",
    3: "First-episode schizophrenia",
    6: "Bipolar disorder",
    4: "Control",
}
GROUP_ORDER = [1, 3, 6, 4]  # paper's table order
PATIENT_GROUPS = [1, 3, 6]

# Electrode regions used in Fig. 3 of the paper
FRONTAL = ["F3", "Fz", "F4"]
FRONTOCENTRAL = ["FC5", "FC6", "FCz"]
PARIETAL = ["P3", "Pz", "P4", "P7", "P8"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_numeric(series: pd.Series) -> pd.Series:
    """Numeric coercion tolerant of Spanish decimal commas and stray spaces."""
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_data(path: str = DATA_FILE, sheet: str = DATA_SHEET) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    return df


def modulation_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Mod_")]


def baseline_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_b")]


def response_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_r")]


def check_modulation_identity(df: pd.DataFrame) -> float:
    """Verify Mod_* == response - baseline for every subject (max abs error)."""
    mod_cols = modulation_columns(df)
    max_err = 0.0
    for mc in mod_cols:
        stem = mc.replace("Mod_Se_", "").replace("Mod_Sec_", "").lower()
        base = [c for c in baseline_columns(df) if c[:-2].lower() == stem]
        resp = [c for c in response_columns(df) if c[:-2].lower() == stem]
        if not base or not resp:
            continue
        diff = to_numeric(df[mc]) - (
            to_numeric(df[resp[0]]) - to_numeric(df[base[0]])
        )
        max_err = max(max_err, float(diff.abs().max()))
    return max_err


def pca_factor_scores(X: pd.DataFrame) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """
    Factor analysis as in the paper: PCA on the correlation matrix, no
    rotation; returns standardized first-factor scores (SPSS 'regression'
    method equivalent up to sign), eigenvalues and first-factor loadings.
    """
    Xv = X.values.astype(float)
    C = np.corrcoef(Xv, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    scores = (Xv - Xv.mean(axis=0)) @ eigvecs[:, 0]
    scores = scores / scores.std(ddof=1)
    loadings = eigvecs[:, 0] * np.sqrt(eigvals[0])
    return pd.Series(scores, index=X.index), eigvals, loadings


def design_matrix(d: pd.DataFrame, group_col: str, covariates: list[str] | None = None):
    """Dummy-coded group design (first group = reference) + covariates."""
    groups = sorted(d[group_col].dropna().unique())
    cols, names = [], []
    for gr in groups[1:]:
        cols.append((d[group_col] == gr).values.astype(float))
        names.append(f"group[{GROUP_LABELS.get(int(gr), gr)}]")
    for cov in covariates or []:
        cols.append(d[cov].values.astype(float))
        names.append(cov)
    return np.column_stack(cols), names


def f_test_block(y, X, names, block_names):
    """Partial F-test for a block of predictors in an OLS model."""
    m = sm.OLS(y, X).fit()
    idx = [names.index(b) for b in block_names]
    R = np.zeros((len(idx), X.shape[1]))
    for i, j in enumerate(idx):
        R[i, j] = 1.0
    ft = m.f_test(R)
    F, p = float(ft.fvalue), float(ft.pvalue)
    df1, df2 = len(idx), int(m.df_resid)
    ss_res = float(m.ssr)
    ss_eff = F * df1 * (ss_res / df2)
    eta2 = ss_eff / (ss_eff + ss_res)
    return m, F, p, df1, df2, eta2


def ancova(df, dv, group_col=GROUP_COL, covariates=("Edad",)):
    d = df[[dv, group_col] + list(covariates)].apply(
        lambda s: to_numeric(s) if s.name != group_col else s
    ).dropna()
    y = d[dv].values.astype(float)
    X, names = design_matrix(d, group_col, list(covariates))
    block = [n for n in names if n.startswith("group")]
    m, F, p, df1, df2, eta2 = f_test_block(y, X, names, block)
    return dict(model=m, F=F, p=p, df1=df1, df2=df2, eta2=eta2, n=len(d), data=d)


def posthoc_vs_control(df, dv, group_col=GROUP_COL, control=4):
    """Welch t-tests of each patient group vs controls, Bonferroni x3."""
    out = []
    ctrl = to_numeric(df.loc[df[group_col] == control, dv]).dropna()
    for gr in PATIENT_GROUPS:
        x = to_numeric(df.loc[df[group_col] == gr, dv]).dropna()
        t, p = stats.ttest_ind(x, ctrl, equal_var=False)
        out.append(
            dict(
                group=GROUP_LABELS[gr],
                n=len(x),
                mean=x.mean(),
                sd=x.std(ddof=1),
                t=t,
                p_raw=p,
                p_bonf=min(1.0, 3 * p),
            )
        )
    return pd.DataFrame(out)


def group_descriptives(df, dv, group_col=GROUP_COL):
    rows = []
    for gr in GROUP_ORDER:
        x = to_numeric(df.loc[df[group_col] == gr, dv]).dropna()
        rows.append(
            dict(group=GROUP_LABELS[gr], n=len(x), mean=x.mean(), sd=x.std(ddof=1))
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def build_table1(df):
    """Demographic / clinical / cognitive descriptives per group (paper Table 1)."""
    specs = [
        ("Age, yr", "Edad", "num"),
        ("Illness duration, mo", "Duracion", "num"),
        ("Antipsychotic dose, CPZ eq mg/d", "equiv_cpz", "num"),
        ("Education proxy: BACS verbal memory", "bacs_a6T1", "num"),
        ("BACS working memory", "bacs_b1T1", "num"),
        ("BACS motor speed", "bacs_c1T1", "num"),
        ("BACS verbal fluency (combined d/e)", "bacs_deT1", "num"),
        ("BACS performance speed", "bacs_f1T1", "num"),
        ("BACS problem solving", "bacs_g1T1", "num"),
        ("MSCEIT total", "msceit_citT1", "num"),
        ("Total IQ (WAIS)", "wais_citT1", "num"),
        ("Hamilton depression total", "HAM_cT1", "num"),
        ("PANSS positive", "panss_pT1", "num"),
        ("PANSS negative", "panss_nT1", "num"),
        ("PANSS total", "panss_tT1", "num"),
    ]
    frames = []
    for label, col, _ in specs:
        d = group_descriptives(df, col)
        d.insert(0, "variable", label)
        # one-way ANOVA across the 4 groups (available cases)
        samples = [
            to_numeric(df.loc[df[GROUP_COL] == gr, col]).dropna().values
            for gr in GROUP_ORDER
        ]
        samples = [s for s in samples if len(s) > 1]
        if len(samples) > 1:
            F, p = stats.f_oneway(*samples)
            d["ANOVA_F"], d["ANOVA_p"] = F, p
        frames.append(d)
    table1 = pd.concat(frames, ignore_index=True)

    # sex distribution (chi-square)
    ct = pd.crosstab(df[GROUP_COL], to_numeric(df["Sexo"]))
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    sex_row = pd.DataFrame(
        [
            dict(
                variable="Sex (1/2)",
                group=GROUP_LABELS[gr],
                n=int(ct.loc[gr].sum()),
                mean=np.nan,
                sd=np.nan,
                ANOVA_F=chi2,
                ANOVA_p=p,
            )
            for gr in GROUP_ORDER
            if gr in ct.index
        ]
    )
    table1 = pd.concat([table1, sex_row], ignore_index=True)
    return table1


def treatment_analyses(df, factor="FAC1_1"):
    """Paper: 'Treatment effects' section."""
    res = {}
    fac = to_numeric(df[factor])
    pat = df[df[GROUP_COL].isin(PATIENT_GROUPS)]

    # correlations with antipsychotic dose and illness duration (all patients)
    for var in ["equiv_cpz", "Duracion"]:
        s = pd.DataFrame({var: to_numeric(pat[var]), "f": fac[pat.index]}).dropna()
        r, p = stats.pearsonr(s[var], s["f"])
        res[f"corr_{var}"] = dict(n=len(s), r=r, p=p)

    # bipolar patients with vs without antipsychotics (Mann-Whitney)
    bip = df[df[GROUP_COL] == 6]
    ap = to_numeric(bip["BIP_CON_SIN_ap"])
    yes = fac[bip.index][ap == 1].dropna()
    no = fac[bip.index][ap == 0].dropna()
    u, p = stats.mannwhitneyu(yes, no)
    res["bipolar_antipsychotic"] = dict(
        n_yes=len(yes), mean_yes=yes.mean(), sd_yes=yes.std(ddof=1),
        n_no=len(no), mean_no=no.mean(), sd_no=no.std(ddof=1), U=u, p=p,
    )

    # lithium (bipolar), benzodiazepines and antidepressants (all patients)
    for var, sub in [("litio", bip), ("Bzd", pat), ("antidepre", pat)]:
        col = sub[var].astype(str).str.strip()
        yes = fac[sub.index][col.isin(["1", "1,00", "1.0", "1,0"])].dropna()
        no = fac[sub.index][col.isin(["0", ",00", "0.0", "0,0"])].dropna()
        if len(yes) and len(no):
            u, p = stats.mannwhitneyu(yes, no)
            res[f"{var}_yes_vs_no"] = dict(
                n_yes=len(yes), mean_yes=yes.mean(),
                n_no=len(no), mean_no=no.mean(), U=u, p=p,
            )
    return res


def clinical_regressions(df, factor="FAC1_1"):
    """OLS: clinical/cognitive score ~ SE-modulation factor, per patient group."""
    rows = []
    outcomes = [
        ("PANSS positive", "panss_pT1"),
        ("PANSS negative", "panss_nT1"),
        ("PANSS total", "panss_tT1"),
        ("BACS verbal memory", "bacs_a6T1"),
        ("BACS working memory", "bacs_b1T1"),
        ("BACS motor speed", "bacs_c1T1"),
        ("BACS verbal fluency", "bacs_deT1"),
        ("BACS performance speed", "bacs_f1T1"),
        ("BACS problem solving", "bacs_g1T1"),
        ("MSCEIT total", "msceit_citT1"),
        ("Hamilton depression", "HAM_cT1"),
    ]
    scopes = {
        "Schizophrenia (chronic + first episode)": [1, 3],
        "Bipolar disorder": [6],
    }
    for scope, groups in scopes.items():
        sub = df[df[GROUP_COL].isin(groups)]
        for label, col in outcomes:
            d = pd.DataFrame(
                {"y": to_numeric(sub[col]), "x": to_numeric(sub[factor])}
            ).dropna()
            if len(d) < 5:
                continue
            X = sm.add_constant(d["x"].values.astype(float))
            m = sm.OLS(d["y"].values.astype(float), X).fit()
            rows.append(
                dict(
                    scope=scope, outcome=label, n=len(d),
                    R2=m.rsquared, beta=m.params[1], F=m.fvalue, p=m.pvalues[1],
                )
            )
    return pd.DataFrame(rows)


def structural_analyses(df, factor="FAC1_1"):
    """DTI graph parameters: group ANCOVA + correlations with SE modulation."""
    out = {"ancova": [], "corr": []}
    for dv, label in [("SW_dt", "Small-world index"), ("D_dt", "Connectivity strength")]:
        a = ancova(df, dv, covariates=("Edad",))
        out["ancova"].append(
            dict(measure=label, F=a["F"], p=a["p"], df1=a["df1"], df2=a["df2"],
                 eta2=a["eta2"], n=a["n"])
        )
        # pairwise vs control
        out.setdefault("posthoc", []).append(
            posthoc_vs_control(df, dv).assign(measure=label)
        )

    s = df[["SW_dt", "D_dt", factor, "Sujeto", GROUP_COL]].apply(
        lambda c: to_numeric(c)
    ).dropna()
    scopes = {
        "all": s,
        "patients": s[s["Sujeto"] == 1],
        "schizophrenia": s[s[GROUP_COL].isin([1, 3])],
    }
    for scope, d in scopes.items():
        for dv in ["SW_dt", "D_dt"]:
            r, p = stats.pearsonr(d[dv], d[factor])
            out["corr"].append(dict(scope=scope, measure=dv, n=len(d), r=r, p=p))
    out["corr"] = pd.DataFrame(out["corr"])
    out["posthoc"] = pd.concat(out["posthoc"], ignore_index=True)
    return out


def make_figures(df, factor_scores, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = [GROUP_LABELS[g] for g in GROUP_ORDER]

    # Fig 1: individual factor scores + 95% CI per group
    fig, ax = plt.subplots(figsize=(7, 5))
    data, positions = [], []
    for i, gr in enumerate(GROUP_ORDER):
        x = factor_scores[df[GROUP_COL] == gr].dropna()
        data.append(x.values)
        positions.append(i + 1)
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        ax.errorbar(i + 1, m, yerr=1.96 * se, fmt="o", color="black", capsize=6, zorder=3)
    ax.boxplot(data, positions=positions, widths=0.5, showfliers=False)
    for i, vals in enumerate(data):
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.25
        ax.scatter(np.full(len(vals), positions[i]) + jitter, vals,
                   s=12, alpha=0.5, color="steelblue", zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Spectral entropy modulation factor score")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_factor_scores_by_group.png"), dpi=150)
    plt.close(fig)

    # Fig 3-like: 95% CI of raw modulation at F / FC / P electrodes
    mod_cols = modulation_columns(df)
    regions = {"Frontal (F3,Fz,F4)": FRONTAL,
               "Frontocentral (FC5,FC6,FCz)": FRONTOCENTRAL,
               "Parietal (P3,Pz,P4,P7,P8)": PARIETAL}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (rname, elecs) in zip(axes, regions.items()):
        cols = [c for c in mod_cols if c.replace("Mod_Se_", "").replace("Mod_Sec_", "").lower()
                in [e.lower() for e in elecs]]
        for i, gr in enumerate(GROUP_ORDER):
            sub = df[df[GROUP_COL] == gr]
            m = sub[cols].apply(to_numeric).mean(axis=1).dropna()
            se = m.std(ddof=1) / np.sqrt(len(m))
            ax.errorbar(i + 1, m.mean(), yerr=1.96 * se, fmt="o", capsize=5)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["Chronic", "First ep.", "Control", "Bipolar"],
                           rotation=15, ha="right", fontsize=8)
        ax.set_title(rname, fontsize=9)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
    axes[0].set_ylabel("SE modulation (response - baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_modulation_CI_by_region.png"), dpi=150)
    plt.close(fig)

    # Fig 4-like: treatment panels
    fac = factor_scores
    pat = df[df[GROUP_COL].isin(PATIENT_GROUPS)]
    bip = df[df[GROUP_COL] == 6]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ap = to_numeric(bip["BIP_CON_SIN_ap"])
    d0, d1 = fac[bip.index][ap == 0].dropna(), fac[bip.index][ap == 1].dropna()
    axes[0].boxplot([d0.values, d1.values], tick_labels=["No", "Yes"])
    axes[0].set_title("Bipolar: antipsychotic")
    axes[0].set_ylabel("SE modulation factor score")

    x = to_numeric(pat["equiv_cpz"])
    y = fac[pat.index]
    ok = x.notna() & y.notna()
    axes[1].scatter(x[ok], y[ok], s=12, alpha=0.6)
    if ok.sum() > 2:
        b1, b0 = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        axes[1].plot(xs, b1 * xs + b0, color="black", lw=1)
    axes[1].set_xlabel("CPZ equivalents (mg/d)")
    axes[1].set_title("Dose vs modulation")

    x = to_numeric(pat["Duracion"])
    ok = x.notna() & y.notna()
    axes[2].scatter(x[ok], y[ok], s=12, alpha=0.6)
    if ok.sum() > 2:
        b1, b0 = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        axes[2].plot(xs, b1 * xs + b0, color="black", lw=1)
    axes[2].set_xlabel("Illness duration (months)")
    axes[2].set_title("Duration vs modulation")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_treatment_effects.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    df = load_data()
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("EEG spectral entropy modulation — replication of Molina et al. 2020")
    print("=" * 78)
    print(f"Participants: {len(df)}")
    print(df[GROUP_COL].map(GROUP_LABELS).value_counts().to_string())

    # 1. data integrity: modulation == response - baseline
    err = check_modulation_identity(df)
    print(f"\n[check] max |Mod - (resp - base)| over all sensors/subjects: {err:.2e}")

    # 2. factor analysis on the 28 modulation columns
    mod_cols = modulation_columns(df)
    X = df[mod_cols].apply(to_numeric)
    scores, eigvals, loadings = pca_factor_scores(X.dropna())
    print(f"\n[factor] modulation SE, {len(mod_cols)} sensors, n={len(scores)}")
    print(f"  PC1 eigenvalue = {eigvals[0]:.2f}  "
          f"({100 * eigvals[0] / len(mod_cols):.1f}% of total variance)")
    print(f"  PC1 loadings range: {loadings.min():.3f} – {loadings.max():.3f}")
    print("  (paper, n=235: eigenvalue 17.77, 56.3% common variance, "
          "loadings 0.680–0.812)")

    # agreement with the stored SPSS factor scores
    fac1 = to_numeric(df["FAC1_1"])
    common = scores.index.intersection(fac1.dropna().index)
    r = np.corrcoef(scores[common], fac1[common])[0, 1]
    print(f"  corr(recomputed PC1, stored FAC1_1) = {r:.4f}")

    # pre-stimulus SE factor
    base_cols = baseline_columns(df)
    Xb = df[base_cols].apply(to_numeric)
    scores_b, eig_b, load_b = pca_factor_scores(Xb.dropna())
    print(f"\n[factor] pre-stimulus SE, {len(base_cols)} sensors, n={len(scores_b)}")
    print(f"  PC1 eigenvalue = {eig_b[0]:.2f}  "
          f"({100 * eig_b[0] / len(base_cols):.1f}% of total variance)")
    print(f"  PC1 loadings range: {load_b.min():.3f} – {load_b.max():.3f}")
    print("  (paper, n=235: eigenvalue 20.67, 71.28% variance, loadings 0.64–0.911)")

    # use the stored SPSS factor scores for inference (original paper values)
    df = df.copy()
    df["SE_mod_factor"] = to_numeric(df["FAC1_1"])
    df["SE_base_factor"] = np.nan
    df.loc[scores_b.index, "SE_base_factor"] = scores_b.values

    # 3. Table 1
    table1 = build_table1(df)
    table1.to_csv(os.path.join(out_dir, "table1_descriptives.csv"), index=False)
    print("\n[table1] saved -> results/table1_descriptives.csv")

    # 4. main ANCOVA on modulation factor
    a = ancova(df, "SE_mod_factor")
    print(f"\n[ANCOVA] SE modulation factor ~ group + age")
    print(f"  group: F({a['df1']},{a['df2']}) = {a['F']:.3f}, p = {a['p']:.4f}, "
          f"eta2 = {a['eta2']:.3f}, n = {a['n']}")
    print("  (paper: F3 = 5.752, p < 0.001, eta2 = 0.106)")

    # age effect and group x age interaction
    d = a["data"]
    X, names = design_matrix(d, GROUP_COL, ["Edad"])
    _, F, p, df1, df2, _ = f_test_block(
        d["SE_mod_factor"].values.astype(float), X, names, ["Edad"]
    )
    print(f"  age: F({df1},{df2}) = {F:.3f}, p = {p:.3f}   (paper: F1 = 0.459, p = 0.50)")
    cols, nm = [], []
    groups = sorted(d[GROUP_COL].unique())
    base_X, base_names = design_matrix(d, GROUP_COL, ["Edad"])
    Xint = np.column_stack([
        base_X,
        *[(d[GROUP_COL] == gr).values.astype(float) * d["Edad"].values.astype(float)
          for gr in groups[1:]],
    ])
    nm = base_names + [f"{n}:age" for n in base_names if n.startswith("group")]
    _, F, p, df1, df2, _ = f_test_block(
        d["SE_mod_factor"].values.astype(float), Xint, nm,
        [n for n in nm if ":age" in n],
    )
    print(f"  group x age: F({df1},{df2}) = {F:.3f}, p = {p:.3f}   "
          f"(paper: F1 = 1.053, p = 0.31)")

    # post-hoc
    ph = posthoc_vs_control(df, "SE_mod_factor")
    ph.to_csv(os.path.join(out_dir, "posthoc_modulation_vs_control.csv"), index=False)
    print("\n[post-hoc] SE modulation factor vs controls (Welch t, Bonferroni x3):")
    print(ph.to_string(index=False))
    print("  (paper: chronic **, first-episode *, bipolar *** vs controls; "
          "no differences among patient groups)")

    # 5. pre-stimulus factor group effect
    ab = ancova(df.dropna(subset=["SE_base_factor"]), "SE_base_factor")
    print(f"\n[ANCOVA] pre-stimulus SE factor ~ group + age")
    print(f"  group: F({ab['df1']},{ab['df2']}) = {ab['F']:.3f}, p = {ab['p']:.3f}  "
          f"(paper: F3 = 0.46, p = 0.70)")

    # 6. treatment effects
    print("\n[treatment]")
    tr = treatment_analyses(df)
    for k, v in tr.items():
        print(f"  {k}: " + ", ".join(
            f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}"
            for kk, vv in v.items()))
    print("  (paper: dose r = -0.06 p = 0.54; duration r = 0.15 p = 0.24; "
          "bipolar AP U = 78, p = 0.65)")

    # 7. structural connectivity
    print("\n[structural connectivity, DTI subsample]")
    st = structural_analyses(df)
    st_df = pd.DataFrame(st["ancova"])
    print(st_df.to_string(index=False))
    print("  (paper: SW F3 = 2.419 p = 0.07; strength F3 = 6.760 p < 0.001; "
          "age n.s. — age effect does NOT reproduce from this file, see docstring)")
    st["posthoc"].to_csv(os.path.join(out_dir, "dti_posthoc.csv"), index=False)
    st["corr"].to_csv(os.path.join(out_dir, "dti_vs_modulation_corr.csv"), index=False)
    print("\n  correlations with SE-modulation factor:")
    print(st["corr"].to_string(index=False))
    print("  (paper: all r = -0.181/-0.183 n.s.; patients -0.182/-0.112 n.s.; "
          "schizophrenia -0.074/-0.059 n.s.)")

    # 8. clinical / cognitive correlates
    print("\n[clinical & cognitive correlates]")
    cr = clinical_regressions(df)
    cr.to_csv(os.path.join(out_dir, "clinical_cognitive_regressions.csv"), index=False)
    print(cr.to_string(index=False))
    print("  (paper, schizophrenia: PANSS negative F = 5.353, beta = 0.270, p = 0.02; "
          "PANSS positive n.s.; verbal memory p = 0.05; working memory p = 0.08)")

    # 9. figures
    make_figures(df, df["SE_mod_factor"], out_dir)
    print(f"\n[figures] saved in {out_dir}/")

    # 10. factor-score table
    fs = pd.DataFrame({
        "Codigo": df["Codigo"],
        "group_code": df[GROUP_COL],
        "group": df[GROUP_COL].map(GROUP_LABELS),
        "SE_mod_factor_recomputed": np.nan,
        "FAC1_1_stored": df["FAC1_1"],
        "SE_base_factor_recomputed": df["SE_base_factor"],
    })
    fs.loc[scores.index, "SE_mod_factor_recomputed"] = scores.values
    fs.to_csv(os.path.join(out_dir, "factor_scores.csv"), index=False)
    print("[scores] saved -> results/factor_scores.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
