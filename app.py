"""MergeMeter: the ghosts of closed PRs predict which open PRs will live."""

from __future__ import annotations

import html
import os

import gradio as gr

from mergemeter import github
from mergemeter.core import score_draft, score_repo
from mergemeter.features import draft_features
from mergemeter.model import NotEnoughHistory, make_tabpfn

EXAMPLES = ["encode/httpx", "huggingface/transformers", "psf/requests", "fastapi/sqlmodel", "pallets/flask"]

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#title h1 { font-size: 2.2rem; margin-bottom: 0; }
#title p { opacity: .8; }
.mm-summary { display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 8px; }
.mm-stat { flex: 1 1 140px; border: 1px solid #3b2a4d; border-radius: 10px; padding: 10px 14px; background: #160f1f; }
.mm-stat b { display: block; font-size: 1.5rem; color: #ff9a3c; }
.mm-stat span { font-size: .8rem; color: #b9a8cc; }
.mm-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
.mm-table th { text-align: left; color: #b9a8cc; font-weight: 600; border-bottom: 1px solid #3b2a4d; padding: 6px; }
.mm-table td { border-bottom: 1px solid #241a30; padding: 6px; vertical-align: middle; }
.mm-table a { color: #ffb36b; text-decoration: none; }
.mm-meter { position: relative; height: 10px; width: 120px; background: #2a1f36; border-radius: 6px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 8px; }
.mm-meter i { position: absolute; inset: 0 auto 0 0; border-radius: 6px; }
.mm-bars .row { display: grid; grid-template-columns: 220px 1fr 64px; gap: 10px; align-items: center; margin: 5px 0; font-size: .88rem; }
.mm-bars .bar { height: 14px; border-radius: 4px; background: linear-gradient(90deg, #7b3fe4, #ff9a3c); }
.mm-bars .dir { font-size: .75rem; color: #b9a8cc; }
.mm-gauge { text-align: center; padding: 18px; border: 1px solid #3b2a4d; border-radius: 12px; background: #160f1f; }
.mm-gauge b { font-size: 3rem; display: block; }
.mm-scroll { max-height: 460px; overflow-y: auto; }
"""


def _color(p: int) -> str:
    return "#5fd38d" if p >= 70 else "#ffb347" if p >= 40 else "#ff5f6d"


def _spirit(p: int) -> str:
    return "🌼 alive" if p >= 70 else "🕯️ limbo" if p >= 40 else "👻 ghost"


def summary_html(res: dict) -> str:
    rate = res["n_merged"] / res["n_closed"] * 100
    auc = res["holdout_auc"]
    auc_txt = f"{auc:.2f}" if auc == auc else "n/a"
    lite = "" if res["mode"] == "graphql" else (
        "<p>⚠️ Lite mode: no GITHUB_TOKEN, so PR size, files and early comments are unknown for this repo.</p>"
    )
    stats = [
        (res["n_closed"], "closed PRs summoned"),
        (f"{rate:.0f}%", "of them merged"),
        (res["n_open"], "open PRs scored"),
        (auc_txt, "ROC-AUC on newest 20%"),
        (f"{res['seconds']['total']:.0f}s", "fetch + TabPFN on CPU"),
    ]
    cards = "".join(f"<div class='mm-stat'><b>{v}</b><span>{k}</span></div>" for v, k in stats)
    return f"<div class='mm-summary'>{cards}</div>{lite}"


def table_html(res: dict) -> str:
    table = res["open_table"]
    if table.empty:
        return "<p>No open PRs right now. Every PR here has already crossed over.</p>"
    rows = []
    for r in table.itertuples():
        p = int(r[4])
        rows.append(
            f"<tr><td><a href='{html.escape(r.Link)}' target='_blank'>{r.PR}</a></td>"
            f"<td>{html.escape(r.Title)}</td><td>{html.escape(r.Author)}</td>"
            f"<td><span class='mm-meter'><i style='width:{p}%;background:{_color(p)}'></i></span><b>{p}%</b></td>"
            f"<td>{_spirit(p)}</td></tr>"
        )
    head = "<tr><th>PR</th><th>Title</th><th>Author</th><th>Merge chance</th><th>Fate</th></tr>"
    return f"<div class='mm-scroll'><table class='mm-table'>{head}{''.join(rows)}</table></div>"


def factors_html(res: dict) -> str:
    f = res["factors"]
    f = f[f["auc_drop"] > 0].head(8)
    if f.empty:
        return "<p>Not enough recent history to rank factors.</p>"
    top = f["auc_drop"].max()
    rows = "".join(
        f"<div class='row'><span>{html.escape(r.factor)} <span class='dir'>{r.direction}</span></span>"
        f"<div class='bar' style='width:{max(r.auc_drop / top * 100, 3):.0f}%'></div>"
        f"<span>−{r.auc_drop:.3f}</span></div>"
        for r in f.itertuples()
    )
    return (
        "<div class='mm-bars'>" + rows + "</div>"
        "<p style='font-size:.8rem;color:#b9a8cc'>Bar = how much TabPFN's ROC-AUC on the newest 20% of PRs drops "
        "when that factor is shuffled. \"helps\" / \"hurts\" = whether more of it goes with merging.</p>"
    )


def run(repo_text: str):
    try:
        res = score_repo(repo_text)
    except (github.GitHubError, NotEnoughHistory) as e:
        raise gr.Error(str(e))
    medians = res["medians"]
    return (
        summary_html(res), table_html(res), factors_html(res), res["repo"],
        gr.update(value=int(medians.get("additions") or 40)),
        gr.update(value=int(medians.get("deletions") or 10)),
        gr.update(value=int(medians.get("changed_files") or 2)),
    )


def what_if(repo, additions, deletions, files, tests, docs, maintainer, merged_before, rejected_before, title, body, labels):
    if not repo:
        raise gr.Error("Summon a repo first; the draft is scored against its history.")
    res = score_repo(repo)
    feats = draft_features(
        additions=additions or 0, deletions=deletions or 0, changed_files=files or 0,
        touches_tests=tests, docs_only=docs, is_maintainer=maintainer,
        prior_merged=merged_before or 0, prior_unmerged=rejected_before or 0,
        title=title or "", body=body or "", labels=labels or [],
    )
    p = round(score_draft(res, feats) * 100)
    return (
        f"<div class='mm-gauge'><b style='color:{_color(p)}'>{p}%</b>"
        f"{_spirit(p)}: chance this draft merges in <code>{html.escape(repo)}</code>, judged by its last "
        f"{res['n_closed']} closed PRs.</div>"
    )


with gr.Blocks(title="MergeMeter: will my PR get merged?") as demo:
    gr.Markdown(
        "# 👻 MergeMeter\n"
        "**The ghosts of PRs past predict the fate of PRs present.** Point it at a public repo: it summons the last "
        "300 closed pull requests, and a local open-weight [TabPFN](https://github.com/PriorLabs/TabPFN) model "
        "learns *in context* which ones lived (merged) and which became ghosts (closed unmerged), then scores every "
        "open PR. No training loop and no hosted API: the model runs on this machine's CPU.",
        elem_id="title",
    )
    with gr.Row():
        repo_in = gr.Textbox(label="Public GitHub repo", placeholder="owner/repo or URL", value="encode/httpx", scale=4)
        go = gr.Button("🕯️ Summon the ghosts", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=repo_in, label="Try one")
    repo_state = gr.State("")
    summary = gr.HTML()
    with gr.Tabs():
        with gr.Tab("Open PRs"):
            table = gr.HTML()
        with gr.Tab("What decides a PR's fate"):
            factors = gr.HTML()
        with gr.Tab("Draft PR what-if"):
            gr.Markdown("Describe a PR you haven't opened yet and see how this repo's history would judge it.")
            with gr.Row():
                add_in = gr.Number(label="Lines added", value=40, precision=0)
                del_in = gr.Number(label="Lines deleted", value=10, precision=0)
                files_in = gr.Number(label="Files changed", value=2, precision=0)
            with gr.Row():
                tests_in = gr.Checkbox(label="Touches tests")
                docs_in = gr.Checkbox(label="Docs only")
                maint_in = gr.Checkbox(label="I'm a maintainer")
                merged_in = gr.Number(label="My earlier merged PRs here", value=0, precision=0)
                rejected_in = gr.Number(label="My earlier unmerged PRs here", value=0, precision=0)
            title_in = gr.Textbox(label="Title", value="fix: handle empty response body")
            body_in = gr.Textbox(label="Description", lines=3, value="Fixes #123. Adds a regression test.")
            labels_in = gr.CheckboxGroup(["bug", "documentation", "enhancement", "hacktoberfest", "dependencies"], label="Labels")
            score_btn = gr.Button("🔮 Read its fortune", variant="primary")
            gauge = gr.HTML()

    gr.Markdown(
        "<small>Built for the DEV Hacktoberfest 2026 Open-Source AI Challenge (Week 4: Ghost in the Machine). "
        "**Built with TabPFN** (v2 open weights, © Prior Labs, Prior Labs License). Code: "
        "[github.com/RostyslavDzhohola/mergemeter](https://github.com/RostyslavDzhohola/mergemeter). "
        "Built by an AI coding agent for Rostyslav Dzhohola.</small>"
    )

    outs = [summary, table, factors, repo_state, add_in, del_in, files_in]
    go.click(run, repo_in, outs)
    repo_in.submit(run, repo_in, outs)
    score_btn.click(
        what_if,
        [repo_state, add_in, del_in, files_in, tests_in, docs_in, maint_in, merged_in, rejected_in, title_in, body_in, labels_in],
        gauge,
    )


def warm_up() -> None:
    """Load the checkpoint before the first visitor arrives."""
    import numpy as np

    make_tabpfn().fit(np.random.rand(20, 3), np.arange(20) % 2).predict_proba(np.random.rand(2, 3))


if __name__ == "__main__":
    warm_up()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Base(primary_hue="orange", neutral_hue="slate").set(
            body_background_fill="#0d0912", body_background_fill_dark="#0d0912",
            block_background_fill="#120c19", block_background_fill_dark="#120c19",
        ),
        css=CSS,
        # The palette is built for the dark theme; force it regardless of the OS setting.
        js="() => document.body.classList.add('dark')",
        show_error=True,
    )
