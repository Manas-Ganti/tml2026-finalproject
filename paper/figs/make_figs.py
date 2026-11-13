"""All paper figures and tables, generated only from results/*/metrics.json (task C5). No hand-entered numbers.

    python paper/figs/make_figs.py --bundle mvr-v1 --model EleutherAI/pythia-160m EleutherAI/pythia-410m

Writes paper/figs/{fig1_dose,fig2_survival,fig4_qq}.pdf and paper/tables/{table1_bundle,scorecard}.tex,
plus paper/tables/{scorecard.md,primary_cells.csv} for quick reading. Definitions: docs/preregistration.md.
Colors: validated categorical palette (dataviz skill); every series also has its own marker (two slots are
below 3:1 contrast, so color never carries identity alone).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TYPES = ["random_seq", "fluent_trap", "watermark", "fictional_fact"]
TYPE_LABEL = {"random_seq": "Random seq.", "fluent_trap": "Fluent trap", "watermark": "Watermark",
              "fictional_fact": "Fictional fact"}
TYPE_TICK = {"random_seq": "Random\nseq.", "fluent_trap": "Fluent\ntrap", "watermark": "Water-\nmark",
             "fictional_fact": "Fict.\nfact"}
PIPE_TICK = {"none": "none", "substring-dedup": "dedup", "nfkc": "NFKC", "paraphrase": "para-\nphrase"}
PIPES = ["none", "substring-dedup", "nfkc", "paraphrase"]
PIPE_LABEL = {"none": "none", "substring-dedup": "substring dedup", "nfkc": "NFKC", "paraphrase": "paraphrase 30%"}
PIPE_COLOR = {"none": "#2a78d6", "substring-dedup": "#eb6834", "nfkc": "#1baf7a", "paraphrase": "#eda100"}
PIPE_MARKER = {"none": "o", "substring-dedup": "s", "nfkc": "^", "paraphrase": "D"}
DOSES = [1, 4, 16, 64]
ACCESS_LABEL = {"gray": "Gray box (log-probs)", "black": "Black box (text only)"}
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
ALPHA = 0.01   # pre-registered
HEAT_DOSE = 16


def load_runs(results: Path, bundle: str, model: str) -> tuple[dict[str, dict], dict | None, dict | None]:
    runs, clean, null = {}, None, None
    for p in sorted(results.glob("*/metrics.json")):
        m = json.loads(p.read_text())
        if m.get("bundle_id") != bundle or m.get("model") != model or m.get("limit"):
            continue
        if m.get("is_clean_model"):
            clean = m
            nf = p.parent / "null_pvalues.json"
            null = json.loads(nf.read_text()) if nf.exists() else None
        elif m.get("pipeline") in PIPES:
            if m["pipeline"] in runs:
                raise SystemExit(f"two runs for pipeline {m['pipeline']}: {runs[m['pipeline']]['run_id']}, {m['run_id']}")
            runs[m["pipeline"]] = m
    return runs, clean, null


def primary(m: dict, access: str, ctype: str, dose: int) -> dict | None:
    for c in m["cells"]:
        if c["primary"] and c["access"] == access and c["type"] == ctype and c["dose"] == dose:
            return c
    return None


def min_dose(m: dict, access: str, ctype: str) -> int | None:
    """Pre-registered: smallest dose with p < alpha."""
    for d in DOSES:
        c = primary(m, access, ctype, d)
        if c and c["p_value"] < ALPHA:
            return d
    return None


def stable_min_dose(m: dict, access: str, ctype: str) -> int | None:
    """Exploratory (added after seeing results, labelled as such in the paper): smallest dose d such that every
    dose >= d is detected. Discards isolated low-dose hits that higher doses don't reproduce."""
    best = None
    for d in reversed(DOSES):
        c = primary(m, access, ctype, d)
        if not (c and c["p_value"] < ALPHA):
            break
        best = d
    return best


def _style(plt):
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.linewidth": 0.6, "font.family": "sans-serif", "pdf.fonttype": 42})


def model_label(model: str) -> str:
    return model.split("/")[-1].replace("pythia-", "Pythia-").replace("m", "M")


def fig1(rows, out: Path, plt):
    """rows: list of (model, runs). One row of panels per model, columns = access level."""
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(5.0, 1.75 * n + 0.35), sharey=True, sharex=True, squeeze=False)
    ylev = DOSES + [128]
    for r, (model, runs) in enumerate(rows):
        for c, access in enumerate(("gray", "black")):
            ax = axes[r][c]
            for k, pipe in enumerate(p for p in PIPES if p in runs):
                xs, ys, iso = [], [], []
                for i, t in enumerate(TYPES):
                    d = min_dose(runs[pipe], access, t)
                    xs.append(i + (k - 1.5) * 0.14)
                    ys.append(np.log2(d if d else 128))
                    iso.append(d is not None and stable_min_dose(runs[pipe], access, t) != d)
                xs, ys, iso = np.array(xs), np.array(ys), np.array(iso)
                ax.scatter(xs[~iso], ys[~iso], s=18, marker=PIPE_MARKER[pipe], color=PIPE_COLOR[pipe],
                           edgecolors="white", linewidths=0.6, zorder=3, label=PIPE_LABEL[pipe])
                if iso.any():   # isolated hit: some higher dose is not detected -> hollow marker
                    ax.scatter(xs[iso], ys[iso], s=16, marker=PIPE_MARKER[pipe], facecolors="white",
                               edgecolors=PIPE_COLOR[pipe], linewidths=1.0, zorder=3)
            ax.set_yticks([np.log2(v) for v in ylev], ["1", "4", "16", "64", ">64"])
            ax.set_xticks(range(len(TYPES)), [TYPE_TICK[t] for t in TYPES], fontsize=6)
            ax.set_title(f"{model_label(model)} · {ACCESS_LABEL[access]}", fontsize=7, color=INK)
            ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
            ax.set_ylim(-0.6, 7.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[r][0].set_ylabel(f"min. copies, p < {ALPHA}")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig1_dose.pdf"); plt.close(fig)


def fig2(rows, out: Path, plt):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(5.0, 1.55 * n + 0.3), sharey=True, squeeze=False)
    for r, (model, runs) in enumerate(rows):
        pipes = [p for p in PIPES if p in runs]
        for c, access in enumerate(("gray", "black")):
            ax = axes[r][c]
            M = np.full((len(TYPES), len(pipes)), np.nan)
            for i, t in enumerate(TYPES):
                for j, p in enumerate(pipes):
                    cell = primary(runs[p], access, t, HEAT_DOSE)
                    M[i, j] = cell["tpr_at_1fpr"] if cell else np.nan
            ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for i in range(len(TYPES)):
                for j in range(len(pipes)):
                    v = M[i, j]
                    ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if (not np.isnan(v) and v > 0.55) else INK)
            ax.set_xticks(range(len(pipes)), [PIPE_LABEL[p].replace(" 30%", "") for p in pipes], fontsize=6)
            ax.set_yticks(range(len(TYPES)), [TYPE_LABEL[t] for t in TYPES], fontsize=6)
            ax.set_title(f"{model_label(model)} · {ACCESS_LABEL[access]}", fontsize=7, color=INK)
            ax.set_xticks(np.arange(-0.5, len(pipes)), minor=True); ax.set_yticks(np.arange(-0.5, len(TYPES)), minor=True)
            ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig2_survival.pdf"); plt.close(fig)


def fig4(nulls, out: Path, plt):
    """nulls: list of (model, null_pvalues dict)."""
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    styles = ["-", (0, (3, 1.5))]
    for (model, null), ls in zip(nulls, styles):
        for det, color, acc in (("loss", "#2a78d6", "gray"), ("completion", "#eb6834", "black")):
            ps = np.sort([p for k, v in null.items() if k.endswith("|" + det) for p in v])
            if len(ps) == 0:
                continue
            q = (np.arange(1, len(ps) + 1) - 0.5) / len(ps)
            ax.plot(q, ps, color=color, linewidth=1.5, linestyle=ls, label=f"{model_label(model)} {acc}")
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle=":", label="uniform")
    ax.set_xlabel("expected p (uniform)"); ax.set_ylabel("observed p (clean model)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig4_qq.pdf"); plt.close(fig)


def table1(rows, cleans, out: Path):
    """rows: list of (model, runs); cleans: {model: clean metrics}. Min bundle size on pipeline none."""
    lines = [r"\begin{tabular}{ll" + "rrrr" * 2 + "}", r"\toprule",
             r" & & \multicolumn{4}{c}{Gray box} & \multicolumn{4}{c}{Black box} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             "Model & Copies & " + " & ".join(str(d) for d in DOSES) + " & " + " & ".join(str(d) for d in DOSES) + r" \\"]
    for model, runs in rows:
        lines.append(r"\midrule")
        none = runs.get("none")
        for i, t in enumerate(TYPES):
            row = [model_label(model) if i == 0 else "", TYPE_LABEL[t]]
            for access in ("gray", "black"):
                for d in DOSES:
                    c = primary(none, access, t, d) if none else None
                    row.append("--" if not c or c.get("min_bundle") is None else str(c["min_bundle"]))
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    for model, _ in rows:
        clean = cleans.get(model)
        if clean:
            f = clean["fpr_check"]
            lines.append(rf"\multicolumn{{10}}{{l}}{{Clean {model_label(model)}: "
                         rf"{sum(1 for c in clean['cells'] if c['primary'] and c['p_value'] < ALPHA)}/{f['n_primary_cells']} "
                         rf"primary cells at $p<{ALPHA}$; re-split nulls: "
                         rf"{f['resplit_frac_p_lt_0.01']*100:.1f}\% at $p<0.01$, KS $p={f['resplit_ks_uniform_p_gray']:.2f}$}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1_bundle.tex").write_text("\n".join(lines) + "\n")


def scorecard(rows, cleans, out: Path):
    head = ["Model", "Canary type", "Min. copies (gray, none)", "Survives dedup?", "Survives NFKC?",
            "Survives paraphrase?", "Works text-only?", "False accusations (clean model)"]
    body = []
    for model, runs in rows:
        def md(p, a, t):
            if p not in runs:
                return None, "n/a"
            d = stable_min_dose(runs[p], a, t)
            return d, (str(d) if d else ">64")
        clean = cleans.get(model)
        for i, t in enumerate(TYPES):
            _, base = md("none", "gray", t)
            surv = []
            for p in ("substring-dedup", "nfkc", "paraphrase"):
                d, sv = md(p, "gray", t)
                surv.append("n/a" if sv == "n/a" else ("yes (" + sv + ")" if d else "no"))
            d_b, s_b = md("none", "black", t)
            text_only = "n/a" if s_b == "n/a" else ("yes (" + s_b + ")" if d_b else "no")
            fa = "n/a"
            if clean:
                cs = [c for c in clean["cells"] if c["primary"] and c["type"] == t]
                fa = f"{sum(c['p_value'] < ALPHA for c in cs)}/{len(cs)}"
            body.append((i == 0, [model_label(model) if i == 0 else "", TYPE_LABEL[t], base] + surv + [text_only, fa]))
    md_lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for _, r in body]
    (out / "scorecard.md").write_text("\n".join(md_lines) + "\n\n'yes (N)': detectable at p<0.01 from N copies onward (every dose >= N; gray box unless the column says "
        "text-only) after that pipeline. Stable rule = exploratory; the pre-registered rule is Fig 1.\n")
    tex = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r"Model & Type & Min.\ copies & Dedup & NFKC & Paraphrase & Text-only & False acc. \\"]
    for first, r in body:
        if first:
            tex.append(r"\midrule")
        tex.append(" & ".join(r).replace("%", r"\%") + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "scorecard.tex").write_text("\n".join(tex) + "\n")


def _fmt_p(p: float) -> str:
    if p < 1e-3:
        mant, e = f"{p:.1e}".split("e")
        return rf"${mant}{{\times}}10^{{{int(e)}}}$"
    return f"{p:.3f}"


def table_all_doses(runs, out: Path, name: str = "app_all_doses.tex"):
    """Appendix: every primary cell (TPR@1%FPR, AUC, p) per pipeline x access x type x dose."""
    lines = [r"\begin{tabular}{llr" + "ccc" * 2 + "}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Gray box (loss)} & \multicolumn{3}{c}{Black box (completion)} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Pipeline & Type & Copies & TPR & AUC & $p$ & TPR & AUC & $p$ \\", r"\midrule"]
    for pi, pipe in enumerate(p for p in PIPES if p in runs):
        if pi:
            lines.append(r"\midrule")
        for t in TYPES:
            for d in DOSES:
                row = [PIPE_LABEL[pipe] if (t == TYPES[0] and d == DOSES[0]) else "",
                       TYPE_LABEL[t] if d == DOSES[0] else "", str(d)]
                for access in ("gray", "black"):
                    c = primary(runs[pipe], access, t, d)
                    if not c:
                        row += ["--"] * 3; continue
                    pv = _fmt_p(c["p_value"])
                    row += [f"{c['tpr_at_1fpr']:.2f}", f"{c['auc']:.2f}",
                            (r"\textbf{" + pv + "}") if c["p_value"] < ALPHA else pv]
                lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / name).write_text("\n".join(lines) + "\n")


def table_survival(results: Path, bundle: str, out: Path):
    f = results / f"data-{bundle}" / "survival.json"
    if not f.exists():
        return
    s = json.loads(f.read_text())
    pipes = [p for p in ("substring-dedup", "nfkc", "paraphrase") if p in s]
    lines = [r"\begin{tabular}{l" + "r" * len(pipes) + "}", r"\toprule",
             "Type & " + " & ".join(PIPE_LABEL[p].replace("%", r"\%") for p in pipes) + r" \\", r"\midrule"]
    for t in TYPES:
        row = [TYPE_LABEL[t]]
        for p in pipes:
            v = (s[p].get("survival_by_type") or {}).get(t)
            row.append("--" if not v else f"{100 * v['frac']:.1f}\\% ({v['alive']:,}/{v['inserted']:,})")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    lines.append("Tokens & " + " & ".join(f"{s[p]['token_count'] / 1e6:.1f}M" for p in pipes) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "app_survival.tex").write_text("\n".join(lines) + "\n")


def table_survival_dose(results: Path, bundle: str, out: Path):
    f = results / f"data-{bundle}" / "survival_by_dose.json"
    if not f.exists():
        return
    s = json.loads(f.read_text())
    pipes = [p for p in ("substring-dedup", "nfkc", "paraphrase") if p in s]
    lines = [r"\begin{tabular}{lr" + "r" * len(pipes) + "}", r"\toprule",
             "Type & Copies & " + " & ".join(PIPE_LABEL[p].replace("%", r"\%") for p in pipes) + r" \\", r"\midrule"]
    for t in TYPES:
        for d in DOSES:
            row = [TYPE_LABEL[t] if d == DOSES[0] else "", str(d)]
            for p in pipes:
                v = s[p].get(f"{t}|{d}")
                row.append("--" if not v else f"{v['alive']:,}/{v['inserted']:,}")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "app_survival_dose.tex").write_text("\n".join(lines) + "\n")


def table_lr(results: Path, out: Path):
    rows = []
    for f in sorted(results.glob("lrcheck*/train_metrics.json")):
        m = json.loads(f.read_text())
        cfg = m.get("config", {})
        rows.append((cfg.get("precision", "?"), cfg.get("lr"), m["heldout_loss_before"], m["heldout_loss_after"]))
    if not rows:
        return
    rows.sort(key=lambda r: (r[0] != "tf32", r[1]))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Precision & LR & Held-out loss before & after & $\Delta$ \\", r"\midrule"]
    for prec, lr, b, a in rows:
        lines.append(rf"{prec} & {lr:.0e} & {b:.3f} & {a:.3f} & {a - b:+.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "app_lr.tex").write_text("\n".join(lines) + "\n")


def cells_csv(metrics: list[dict], out: Path):
    with open(out / "primary_cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "pipeline", "access", "type", "dose", "auc", "tpr_at_1fpr", "p_value", "min_bundle"])
        for m in metrics:
            for c in m["cells"]:
                if c["primary"]:
                    w.writerow([m["run_id"], m["pipeline"], c["access"], c["type"], c["dose"], f"{c['auc']:.4f}",
                                f"{c['tpr_at_1fpr']:.4f}", f"{c['p_value']:.3e}", c.get("min_bundle")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True, nargs="+", help="one or more HF ids; one figure row per model")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper"))
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)

    results = Path(args.results)
    rows, cleans, nulls, metrics = [], {}, [], []
    for model in args.model:
        runs, clean, null = load_runs(results, args.bundle, model)
        print(f"figs: {model}: pipelines {sorted(runs)} | clean model: {bool(clean)}")
        if runs:
            rows.append((model, runs)); metrics += list(runs.values())
        if clean:
            cleans[model] = clean; metrics.append(clean)
        if null:
            nulls.append((model, null))
    figs, tabs = Path(args.out) / "figs", Path(args.out) / "tables"
    figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    if rows:
        fig1(rows, figs, plt); fig2(rows, figs, plt); table1(rows, cleans, tabs); scorecard(rows, cleans, tabs)
        for model, runs in rows:
            table_all_doses(runs, tabs, f"app_all_doses_{model.split('/')[-1].replace('pythia-', 'p')}.tex")
    if nulls:
        fig4(nulls, figs, plt)
    table_survival(results, args.bundle, tabs)
    table_survival_dose(results, args.bundle, tabs)
    table_lr(results, tabs)
    cells_csv(metrics, tabs)
    print(f"figs: wrote {figs} and {tabs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
