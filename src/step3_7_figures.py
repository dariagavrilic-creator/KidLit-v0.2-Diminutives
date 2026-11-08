#!/usr/bin/env python3
"""KidLit by Numbers — шаг 3.6: рисунки для статьи."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444",
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True})
C = {"RUS-O": "#0072B2", "RUS-T": "#E69F00"}      # Wong (2011), как в предыдущих статьях KidLit
H = {"RUS-O": "///", "RUS-T": "xxx"}
A = "kidlit_dim/analysis/"
CLS_EN = {"животные": "animals", "природа и погода": "nature & weather", "семья и люди": "family & people",
          "грамматическое слово": "function words", "игрушки и игры": "toys & play", "транспорт и техника": "vehicles & machines",
          "действия и движение": "actions & motion", "эмоции, оценка, отношения": "emotion & evaluation",
          "цвет, размер, форма": "colour, size, shape", "время и место": "time & place", "дом и быт": "home & household",
          "части тела": "body parts", "речь и этикет": "speech & etiquette", "сказочное и волшебное": "fairy-tale & magic",
          "еда и напитки": "food & drink", "имя собственное": "proper names", "звукоподражания и междометия": "onomatopoeia & interjections",
          "праздники": "holidays", "книги и учёба": "books & learning", "прочее": "other"}


def fig2():
    m = pd.read_csv(A + "book_metrics.csv")
    r = pd.read_csv(A + "group_tests.csv").set_index("metric")
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.0))
    rng = np.random.default_rng(1)
    for ax, (k, lab) in zip(axs, [("dim_per_1000w", "Diminutives per 1,000 words"), ("dim_noun_per_100n", "Diminutive nouns per 100 nouns")]):
        data = [m[m.group == g][k].values for g in ["RUS-O", "RUS-T"]]
        bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True, medianprops=dict(color="#222", lw=1.5),
                        whiskerprops=dict(color="#777"), capprops=dict(color="#777"))
        for patch, g in zip(bp["boxes"], ["RUS-O", "RUS-T"]):
            patch.set_facecolor("white"); patch.set_edgecolor(C[g]); patch.set_linewidth(1.5); patch.set_hatch(H[g])
        for i, (g, d) in enumerate(zip(["RUS-O", "RUS-T"], data), start=1):
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(d)), d, s=12, color=C[g], alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(i + 0.3, np.median(d), f"{np.median(d):.1f}", va="center", fontsize=8, color="#222")
        rr = r.loc[k]
        ax.set_title(f"{lab}\nCliff's δ = {rr.cliffs_delta:.2f} [{rr.delta_ci_lo:.2f}, {rr.delta_ci_hi:.2f}], q = {rr.q_bh:.3f}",
                     fontsize=8.5, loc="left")
        ax.set_xticks([1, 2]); ax.set_xticklabels(["RUS-O (n = 60)", "RUS-T (n = 60)"])
        ax.set_xlim(0.5, 2.75)
    fig.tight_layout(); fig.savefig(A + "figures/fig2_dim_by_book.png", dpi=300); plt.close(fig)


def fig3():
    k = pd.read_excel("kidlit_dim/keyness/filled/kidlit_vs_ref_top100.xlsx", "Разметка")
    vc = k.sem_class.map(CLS_EN).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.barh(vc.index, vc.values, color=C["RUS-O"], height=0.65); ax.grid(axis="y", visible=False)
    for y, v in enumerate(vc.values):
        ax.text(v + 0.3, y, str(v), va="center", fontsize=8, color="#222")
    ax.set_xlabel("Lemmas among the top 100 KidLit keywords\n(vs. Lyashevskaya–Sharoff frequency dictionary)")  # two lines: one line was cut off at the right edge
    fig.tight_layout(); fig.savefig(A + "figures/fig3_keyword_classes.png", dpi=300); plt.close(fig)


def fig5():
    p = pd.read_csv(A + "pleonasm.csv")
    p = p[p.adjectives.str.startswith("размер") & p.group.isin(["RUS-O", "RUS-T"])]
    cats = [("dim_per_100", "with the diminutive\n(маленький домик)"), ("base_per_100", "with the same base\n(маленький дом)"),
            ("other_per_100", "with other\nnouns")]
    fig, ax = plt.subplots(figsize=(6.3, 2.7)); x = np.arange(3); w = 0.36
    for j, g in enumerate(["RUS-O", "RUS-T"]):
        row = p[p.group == g].iloc[0]; vals = [row[c] for c, _ in cats]
        ax.bar(x + (j - 0.5) * w, vals, w - 0.04, color="white", edgecolor=C[g], hatch=H[g], linewidth=1.5, label=g)
        for xx, v in zip(x + (j - 0.5) * w, vals):
            ax.text(xx, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.5, color="#222")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in cats]); ax.set_ylabel("per 100 nouns"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Size modifiers (маленький, крошечный, небольшой, …)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(A + "figures/fig5_size_pleonasm.png", dpi=300); plt.close(fig)




def fig4():
    s = pd.read_csv(A + "semantic_classes_by_group.csv", index_col=0)
    en = dict(CLS_EN, **{"одежда": "clothing"})
    s = s[(s["n_RUS-O"] + s["n_RUS-T"]) >= 20].copy()
    s.index = s.index.map(en)
    s = s.sort_values("per1000w_RUS-O")
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.9), sharey=True)
    y = np.arange(len(s))
    for ax, (cols, xl) in zip(axs, [(("share_RUS-O", "share_RUS-T"), "share of diminutive tokens, %"),
                                    (("per1000w_RUS-O", "per1000w_RUS-T"), "diminutives per 1,000 words")]):
        mult = 100 if "share" in cols[0] else 1
        ax.hlines(y, (s[list(cols)] * mult).min(axis=1), (s[list(cols)] * mult).max(axis=1), color="#cfcfcf", lw=1.5, zorder=1)
        ax.scatter(s[cols[0]] * mult, y, s=34, color=C["RUS-O"], marker="o", label="RUS-O", zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter(s[cols[1]] * mult, y, s=38, color=C["RUS-T"], marker="D", label="RUS-T", zorder=3, edgecolor="white", linewidth=0.8)
        ax.set_xlabel(xl); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels(s.index)
    axs[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(A + "figures/fig4_semantic_classes.png", dpi=300); plt.close(fig)




def fig1():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.3, 3.7)); ax.set_xlim(-0.8, 101.2); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, title, body, edge="#555", tfs=7.0, bfs=5.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", lw=1.1, ec=edge, fc="white"))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=tfs, weight="bold", color="#222")
        if body:
            ax.text(x + w / 2, y + h - 5.8, body, ha="center", va="top", fontsize=bfs, color="#333", linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0, shrinkA=1, shrinkB=1))

    w, gap, y0, h = 17.2, 3.4, 38, 20
    top = [("Corpus", "120 books, age 0+\nRUS-O 60 / RUS-T 60\n176k words", "#555"),
           ("Preprocessing", "razdel tokens\nspaCy UD parse\npymorphy3 lemmas\nlemma repair", "#555"),
           ("Candidates", "diminutive\nsuffix list\nnouns, adjectives,\nadverbs", "#555"),
           ("Base recovery", "pymorphy3 +\nDerivBase.Ru;\ngender, animacy,\nstem checks", "#555"),
           ("Labels", "precedent lexicon\n(guidelines v0.4)\nDIM / LEX / HYP\nYOUNG / NO", C["RUS-O"])]
    for i, (t, b, e) in enumerate(top):
        x = 0.5 + i * (w + gap)
        box(x, y0, w, h, t, b, edge=e)
        if i < 4:
            arrow(x + w + 0.4, y0 + h / 2, x + w + gap - 0.4, y0 + h / 2)
    box(0.5, 1, 31, 29, "Gold standard & evaluation",
        "guidelines + pilot\n8 books full reading, 400 candidates\ndouble annotation, adjudication\n(precedents P-01–P-26)\n"
        "dev tuning → single test run\nP / R / F1 with 95% Wilson CI", edge=C["RUS-T"], bfs=5.9)
    box(36, 1, 63.5, 29, "Analyses (unit: book)", "", tfs=7.0)
    items = [("Frequency & choice", "per 1,000 words, per 100 nouns,\ndiminutive choice; MW, Cliff's δ, BH, OLS"),
             ("Keyness & semantics", "LL / log ratio vs. frequency\ndictionary; 21 semantic classes"),
             ("Collocations", "amod logDice, size pleonasm,\naddress forms"),
             ("Sentiment & NER", "Label Studio layers: DIM sentences,\ncharacter mentions")]
    for i, (t, b) in enumerate(items):
        x = 38 + (i % 2) * 31; y = 23.5 - (i // 2) * 11.5
        ax.text(x + 14.5, y, t, ha="center", va="top", fontsize=6.4, weight="bold", color="#222")
        ax.text(x + 14.5, y - 3.2, b, ha="center", va="top", fontsize=5.6, color="#333", linespacing=1.3)
    lx = 0.5 + 4 * (w + gap) + w / 2
    arrow(lx, y0 - 0.6, 75, 30.6)
    arrow(lx - 4, y0 - 0.6, 24, 30.6)
    fig.savefig(A + "figures/fig1_pipeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
