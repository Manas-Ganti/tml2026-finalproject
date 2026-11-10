"""All paper figures and tables, generated only from results/*/metrics.json (task C5). No hand-entered numbers.

    python paper/figs/make_figs.py --bundle mvr-v1 --model EleutherAI/pythia-160m

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
    for d in DOSES:
        c = primary(m, access, ctype, d)
        if c and c["p_value"] < ALPHA:
            return d
    return None


def _style(plt):
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.linewidth": 0.6, "font.family": "sans-serif", "pdf.fonttype": 42})


def fig1(runs, out: Path, plt):
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    ylev = DOSES + [128]
    for ax, access in zip(axes, ("gray", "black")):
        for k, pipe in enumerate(p for p in PIPES if p in runs):
            xs, ys = [], []
            for i, t in enumerate(TYPES):
                d = min_dose(runs[pipe], access, t)
                xs.append(i + (k - 1.5) * 0.14)
                ys.append(np.log2(d if d else 128))
            ax.scatter(xs, ys, s=16, marker=PIPE_MARKER[pipe], color=PIPE_COLOR[pipe], edgecolors="white",
                       linewidths=0.6, zorder=3, label=PIPE_LABEL[pipe])
        ax.set_yticks([np.log2(v) for v in ylev], ["1", "4", "16", "64", ">64"])
        ax.set_xticks(range(len(TYPES)), [TYPE_TICK[t] for t in TYPES], fontsize=6)
        ax.set_title(ACCESS_LABEL[access], fontsize=7, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.set_ylim(-0.6, 7.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(f"min. copies for p < {ALPHA}\n(>64 = not detected)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig1_dose.pdf"); plt.close(fig)


def fig2(runs, out: Path, plt):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    pipes = [p for p in PIPES if p in runs]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    for ax, access in zip(axes, ("gray", "black")):
        M = np.full((len(TYPES), len(pipes)), np.nan)
        for i, t in enumerate(TYPES):
            for j, p in enumerate(pipes):
                c = primary(runs[p], access, t, HEAT_DOSE)
                M[i, j] = c["tpr_at_1fpr"] if c else np.nan
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(TYPES)):
            for j in range(len(pipes)):
                v = M[i, j]
                ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if (not np.isnan(v) and v > 0.55) else INK)
        ax.set_xticks(range(len(pipes)), [PIPE_TICK[p] for p in pipes], fontsize=6)
        ax.set_yticks(range(len(TYPES)), [TYPE_LABEL[t] for t in TYPES], fontsize=6)
        ax.set_title(ACCESS_LABEL[access], fontsize=7, color=INK)
        ax.set_xticks(np.arange(-0.5, len(pipes)), minor=True); ax.set_yticks(np.arange(-0.5, len(TYPES)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"TPR at 1% FPR, {HEAT_DOSE} copies", fontsize=7, color=INK2, y=0.98)
    fig.tight_layout()
    fig.savefig(out / "fig2_survival.pdf"); plt.close(fig)


def fig4(null: dict, out: Path, plt):
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    for det, color, marker, label in (("loss", "#2a78d6", "o", "gray: loss"), ("completion", "#eb6834", "s", "black: completion")):
        ps = np.sort([p for k, v in null.items() if k.endswith("|" + det) for p in v])
        if len(ps) == 0:
            continue
        q = (np.arange(1, len(ps) + 1) - 0.5) / len(ps)
        ax.plot(q, ps, color=color, linewidth=1.5, label=label)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle="--", label="uniform")
    ax.set_xlabel("expected p (uniform)"); ax.set_ylabel("observed p (clean model)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig4_qq.pdf"); plt.close(fig)


def table1(runs, clean, out: Path):
    lines = [r"\begin{tabular}{l" + "rrrr" * 2 + "}", r"\toprule",
             r" & \multicolumn{4}{c}{Gray box} & \multicolumn{4}{c}{Black box} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             "Copies & " + " & ".join(str(d) for d in DOSES) + " & " + " & ".join(str(d) for d in DOSES) + r" \\",
             r"\midrule"]
    none = runs.get("none")
    for t in TYPES:
        row = [TYPE_LABEL[t]]
        for access in ("gray", "black"):
            for d in DOSES:
                c = primary(none, access, t, d) if none else None
                row.append("--" if not c or c.get("min_bundle") is None else str(c["min_bundle"]))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    if clean:
        f = clean["fpr_check"]
        lines.append(rf"\multicolumn{{9}}{{l}}{{Clean base model: {f['frac_primary_cells_p_lt_alpha']*100:.1f}\% of "
                     rf"{f['n_primary_cells']} primary cells at $p<{ALPHA}$; re-split nulls: "
                     rf"{f['resplit_frac_p_lt_0.01']*100:.1f}\% at $p<0.01$}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1_bundle.tex").write_text("\n".join(lines) + "\n")


def scorecard(runs, clean, out: Path):
    def md(p, a, t):
        if p not in runs:
            return None, "n/a"
        d = min_dose(runs[p], a, t)
        return d, (str(d) if d else ">64")

    rows = []
    for t in TYPES:
        _, base = md("none", "gray", t)
        surv = []
        for p in ("substring-dedup", "nfkc", "paraphrase"):
            d, s = md(p, "gray", t)
            surv.append("n/a" if s == "n/a" else ("yes (" + s + ")" if d else "no"))
        d_b, s_b = md("none", "black", t)
        text_only = "n/a" if s_b == "n/a" else ("yes (" + s_b + ")" if d_b else "no")
        fa = "n/a"
        if clean:
            cs = [c for c in clean["cells"] if c["primary"] and c["type"] == t]
            fa = f"{sum(c['p_value'] < ALPHA for c in cs)}/{len(cs)}"
        rows.append([TYPE_LABEL[t], base] + surv + [text_only, fa])
    head = ["Canary type", "Min. copies (gray, none)", "Survives dedup?", "Survives NFKC?", "Survives paraphrase?",
            "Works text-only?", "False accusations (clean model)"]
    md_lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    (out / "scorecard.md").write_text("\n".join(md_lines) + "\n\n'yes (N)': detectable at p<0.01 from N copies (gray box) after that pipeline.\n")
    tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Type & Min.\ copies & Dedup & NFKC & Paraphrase & Text-only & False acc. \\", r"\midrule"]
    tex += [" & ".join(r).replace("%", r"\%") + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "scorecard.tex").write_text("\n".join(tex) + "\n")


def cells_csv(runs, clean, out: Path):
    with open(out / "primary_cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "pipeline", "access", "type", "dose", "auc", "tpr_at_1fpr", "p_value", "min_bundle"])
        for m in list(runs.values()) + ([clean] if clean else []):
            for c in m["cells"]:
                if c["primary"]:
                    w.writerow([m["run_id"], m["pipeline"], c["access"], c["type"], c["dose"], f"{c['auc']:.4f}",
                                f"{c['tpr_at_1fpr']:.4f}", f"{c['p_value']:.3e}", c.get("min_bundle")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper"))
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)

    runs, clean, null = load_runs(Path(args.results), args.bundle, args.model)
    print(f"figs: pipelines {sorted(runs)} | clean model: {bool(clean)}")
    figs, tabs = Path(args.out) / "figs", Path(args.out) / "tables"
    figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    if runs:
        fig1(runs, figs, plt); fig2(runs, figs, plt); table1(runs, clean, tabs); scorecard(runs, clean, tabs)
    if null:
        fig4(null, figs, plt)
    cells_csv(runs, clean, tabs)
    print(f"figs: wrote {figs} and {tabs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
