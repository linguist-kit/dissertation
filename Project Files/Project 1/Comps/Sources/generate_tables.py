import pandas as pd

BASE = "/Users/glossophilia/Library/CloudStorage/GoogleDrive-rumilofaniel@gmail.com/.shortcut-targets-by-id/1rb22-U6MRLBxShIa-cAEV9DA9ZOAGCFS/Diss/Project Files/Project 1/Comps/"

MODELS  = ["Claude", "Gemini", "OpenAI"]
METRICS = ["Precision", "Recall", "F1", "Kappa", "Alpha"]

# Display labels for table 1 permutation rows
ROW_LABELS_T1 = {
    "full": "full CRI set",
    "abl1": "intuition",
    "abl2": r"set\_to\_1\_if",
    "abl3": r"set\_to\_0\_if",
    "abl4": r"edge\_cases",
    "abl5": r"examples\_positive",
    "abl6": r"examples\_negative",
    "all1": "intuition",
    "all2": r"set\_to\_1\_if",
    "all3": r"set\_to\_0\_if",
    "all4": r"edge\_cases",
    "all5": r"examples\_positive",
    "all6": r"examples\_negative",
}

# Groups: (rotated label or None, [row keys])
GROUPS_T1 = [
    (None,         ["full"]),
    ("Ablations",  ["abl1", "abl2", "abl3", "abl4", "abl5", "abl6"]),
    ("Allations",  ["all1", "all2", "all3", "all4", "all5", "all6"]),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_table(path, drop_rows=None):
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df = df.dropna(how="all")
    if drop_rows:
        df = df[~df.index.isin(drop_rows)]
    df = df.astype(float).round(4)
    return df


def fmt(val, bold=False):
    if pd.isna(val):
        return "--"
    s = f"{val:.4f}"
    return r"\textbf{" + s + "}" if bold else s


def multirow(n, content):
    return r"\multirow{" + str(n) + r"}{*}{" + content + "}"


def rotatebox(text):
    return r"\rotatebox[origin=c]{90}{" + text + "}"


# ── Table 1: stacked by model, grouped permutations ──────────────────────────

def t1_to_latex(df, caption, label):
    # Precompute row counts (same for every model)
    n_per_group = [(g, [k for k in keys if k in df.index])
                   for g, keys in GROUPS_T1]
    n_model_rows = sum(len(keys) for _, keys in n_per_group)

    # col spec: model | group | permutation | Precision Recall F1 Kappa Alpha
    n_data_cols   = len(METRICS)           # 5
    n_label_cols  = 3                      # model, group, perm
    n_total_cols  = n_label_cols + n_data_cols  # 8

    lines = []
    lines.append(r"\begin{table*}[p]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lllrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & & Permutation & " + " & ".join(METRICS) + r" \\")
    lines.append(r"\midrule")

    for m_idx, model in enumerate(MODELS):
        if m_idx > 0:
            lines.append(r"\midrule")

        best_f1_key = df[(model, "F1")].idxmax()
        first_row_of_model = True

        for g_idx, (group_name, valid_keys) in enumerate(n_per_group):
            if not valid_keys:
                continue

            # ── separator between groups within a model ───────────────────
            if g_idx > 0:
                # cmidrule skips the model column (col 1) so multirow stays intact
                lines.append(r"\cmidrule(l){2-" + str(n_total_cols) + "}")

            for r_idx, key in enumerate(valid_keys):
                row = df.loc[key]

                # Model cell: only on the very first row of this model block
                if first_row_of_model:
                    model_cell = multirow(n_model_rows, model)
                    first_row_of_model = False
                else:
                    model_cell = ""

                # Group cell: first row of each non-None group, rotated
                if group_name is not None and r_idx == 0:
                    group_cell = multirow(len(valid_keys), rotatebox(group_name))
                else:
                    group_cell = ""

                perm_label = ROW_LABELS_T1.get(key, key)

                cells = [model_cell, group_cell, perm_label]
                for metric in METRICS:
                    bold = (metric == "F1" and key == best_f1_key)
                    cells.append(fmt(row[(model, metric)], bold))

                lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Table 2: stacked by model, all features ───────────────────────────────────

def t2_to_latex(df, caption, label):
    n_features   = len(df)
    n_total_cols = 2 + len(METRICS)   # model, feature, 5 metrics = 7

    lines = []
    lines.append(r"\begin{table*}[p]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Feature & " + " & ".join(METRICS) + r" \\")
    lines.append(r"\midrule")

    for m_idx, model in enumerate(MODELS):
        if m_idx > 0:
            lines.append(r"\midrule")

        first_row = True
        for idx, row in df.iterrows():
            model_cell = multirow(n_features, model) if first_row else ""
            first_row  = False

            cells = [model_cell, str(idx)]
            for metric in METRICS:
                cells.append(fmt(row[(model, metric)]))

            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Run ───────────────────────────────────────────────────────────────────────

t1 = read_table(BASE + "Sources/table1_by_permutation.csv")
latex1 = t1_to_latex(t1, caption="Results by permutation.", label="tab:by_permutation")

t2 = read_table(BASE + "Sources/table2_by_feature.csv", drop_rows=["AktSemel"])
latex2 = t2_to_latex(t2, caption="Results by feature.", label="tab:by_feature")

out_path = BASE + "Results/tables.tex"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("% ── Table 1 ──────────────────────────────────────────────────────\n")
    f.write(latex1)
    f.write("\n\n")
    f.write("% ── Table 2 ──────────────────────────────────────────────────────\n")
    f.write(latex2)
    f.write("\n")

print(f"Written to {out_path}")
print("\n=== TABLE 1 ===\n")
print(latex1)
print("\n=== TABLE 2 ===\n")
print(latex2)
