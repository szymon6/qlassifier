r"""``scripts/paper_tables.py``: every number in the paper comes from the saved results (TDD §7.12, PRD §9).

Real data only: the saved phase-9 results in ``benchmarks/`` and the phase-7 experiment files. The phase-7 oracles
below are the numbers printed by the phase-7 ``summarize.py`` scripts (e07, e09, e11, e12 ``summary.md`` / README).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from types import ModuleType

import pytest

from tests.conftest import ROOT

pytestmark = pytest.mark.docs

MAIN = ROOT / "docs" / "paper" / "main.tex"


def test_tex_escapes_text_and_symbols(paper_tables: ModuleType) -> None:
    tex = paper_tables.tex
    assert tex("a_b & 50%") == r"a\_b \& 50\%"
    assert tex("Δ >= 2 pp → 1,420×") == r"$\Delta$ $\geq$ 2 pp $\rightarrow$ 1,420$\times$"
    assert tex("Qlassifier - ZZ < 0, -1.3") == r"Qlassifier $-$ ZZ $<$ 0, $-$1.3"
    assert tex("`qlassifier benchmark breast_cancer`") == r"\texttt{qlassifier benchmark breast\_cancer}"
    assert tex("FQK-bw full, ceil(8/2)") == "FQK-bw full, ceil(8/2)"
    with pytest.raises(ValueError, match="no LaTeX mapping"):
        tex("ψ")
    with pytest.raises(ValueError, match="unbalanced"):
        tex("`open")


def test_number_formats(paper_tables: ModuleType) -> None:
    pt = paper_tables
    assert pt.fixed(-0.0004) == "0.000"
    assert pt.signed(37.64) == "+37.6"
    assert pt.signed(-1.31) == "$-$1.3"
    assert pt.signed(-0.04) == "0.0"
    assert pt.signed(-1.31, style="md") == "−1.3"
    assert pt.pp(0.37642) == "+37.6"
    assert pt.ci_pp(-0.0309, 0.0047) == "[$-$3.1, +0.5]"
    assert pt.ci_abs(0.9284, 0.9681) == "[0.928, 0.968]"
    assert pt.thousands(2423299) == "2,423,299"
    assert pt.wins(0.6, 10) == "6/10"
    with pytest.raises(pt.InputError):
        pt.ci_pp(None, 0.1)


def test_phase7_history_reproduces_the_phase7_summaries(paper_tables: ModuleType) -> None:
    pt = paper_tables
    hist, loss = pt.history(pt.EXPERIMENTS)

    def fmt(p: object, nd: int) -> str:
        return f"{pt.pp(p.mean, nd, 'md')} {pt.ci_pp(p.lo, p.hi, nd, 'md')}"  # type: ignore[attr-defined]

    assert [(look.exp, look.paired.b, fmt(look.paired, 2 if look.paired.level > 0.95 else 1)) for look in hist] == [
        ("e07", "mlp", "+1.4 [+0.3, +2.4]"),
        ("e11", "mlp", "+0.7 [−0.6, +2.1]"),
        ("e12", "mlp", "+0.25 [−1.13, +1.62]"),
        ("e12", "grad_boost", "+0.81 [−0.61, +2.23]"),
    ]
    assert [look.paired.level for look in hist] == [0.95, 0.95, 0.975, 0.975]
    assert [look.seeds for look in hist] == [(200, 209), (200, 209), (300, 309), (300, 309)]
    assert loss["quakes"].b == "random_forest"
    assert fmt(loss["quakes"], 1) == "−6.6 [−8.5, −4.8]"
    assert fmt(loss["quakes.wide"], 1) == "−2.1 [−5.2, +1.1]"
    assert fmt(loss["quakes.rbf"], 1) == "+5.6 [+2.3, +9.0]"


def test_claims_hold_and_a_broken_claim_is_caught(paper_tables: ModuleType) -> None:
    results = paper_tables.load_results(paper_tables.BENCH)
    assert paper_tables.claims(results) == []
    broken = copy.deepcopy(results)
    tie = broken["breast_cancer"]["highlights"]["vs_best_classical"]
    tie["ci_low"], tie["ci_high"] = 0.01, 0.03
    zz = broken["heart"]["highlights"]["vs_standard_qml"]
    zz["win_rate"] = 0.9
    assert paper_tables.claims(broken) == [
        "breast_cancer: tied with the best classical model",
        "heart: Qlassifier beats ZZ QSVC on every seed, CI > 0",
    ]


def test_every_value_in_the_paper_is_generated(paper_tables: ModuleType) -> None:
    files = paper_tables.build()
    values = files[paper_tables.PAPER / "generated" / "values.tex"].decode()
    defined = set(re.findall(r"\\qlvalue\{([a-z0-9.]+)\}", values))
    used = set(re.findall(r"\\val\{([a-z0-9.]+)\}", MAIN.read_text(encoding="utf-8")))
    assert used, "main.tex prints no generated value"
    assert used - defined == set()


def test_review_values_are_generated_not_typed(paper_tables: ModuleType) -> None:
    """Peer-review findings 1-3, 6, 7: worst seeds, the final-Gram circuit ratio, grid edges, target statuses, wins."""
    pt = paper_tables
    files = pt.build()
    values = dict(
        re.findall(r"\\qlvalue\{([a-z0-9.]+)\}\{(.*)\}", files[pt.PAPER / "generated" / "values.tex"].decode())
    )
    assert values["noise.max"] == "0.19"  # largest mean over seeds (Part A, devices)
    assert (values["noise.worst"], values["noise.worst.dataset"]) == ("0.94", "Ionosphere")  # worst single seed
    assert values["bc.noiseb.qlassifier.worst"] == "2.5"
    assert (values["bc.circuitratio"], values["bc.circuitratio.final"]) == ("1,420", "86")
    assert values["edge.list"].startswith(
        "Random forest 1.0 (Breast Cancer, Ionosphere); FQK-bw full 0.9 (Breast Cancer)"
    )
    assert "Qlassifier 0.6 (Heart)" in values["edge.list"]
    assert (values["edge.qlassifier.max"], values["edge.qlassifier.where"]) == ("0.6", "Heart")
    assert (values["targets.met"], values["targets.missed"], values["targets.reportonly"]) == ("9", "0", "1")
    assert (values["targets.reportonly.ids"], values["targets.missed.ids"]) == ("BT-6", "none")
    assert values["bt8.status"] == "met"
    targets = files[pt.PAPER / "generated" / "targets.tex"].decode()
    assert "pending" not in targets and targets.count("& met \\\\") == 9
    readme = files[pt.README].decode()
    assert "vs MLP: tie (0/10 seeds)" in readme
    # stage-2 research results (e07 hardware record, e02 sweep, e01 datasets)
    assert (values["hw.backend"], values["hw.jobid"], values["hw.qpu"]) == (r"ibm\_fez", "dbesrqnpc3tc739mb92g", "82")
    assert (values["hw.acc"], values["hw.pred"], values["hw.pred.sd"], values["hw.delta"]) == ("0.900", "0.880", "0.010", "+2.0")
    assert (values["hw.circuits"], values["hw.shots"], values["hw.correct"], values["hw.ntest"]) == ("300", "1,024", "36", "40")
    assert (values["sweep.auc.q"], values["sweep.auc.fqk"], values["sweep.auc.zz"]) == ("0.900", "0.738", "0.625")
    assert (values["sweep.q.drop1"], values["sweep.fqk.drop0"]) == ("+0.12", "19.2")
    assert (values["s2.spambase.q"], values["s2.spambase.vszz"], values["s2.banknote.vszz"]) == ("0.911", "+1.2", "+4.4")
    assert values["s2.spambase.vsmlp.ci"] == "[$-$1.2, $-$0.2]"
    assert (values["hw.all.seeds"], values["hw.all.later"]) == ("5", "4")
    assert values["hw.all.pass"] in {"3", "4", "5"} and 300 < float(values["hw.all.qpu"]) < 600
    seeds_table = files[pt.PAPER / "generated" / "hardware-seeds.tex"].decode()
    assert seeds_table.count("\\\\") == 6 and "dbesrqnpc3tc739mb92g" in seeds_table
    hardware = files[pt.PAPER / "generated" / "hardware.tex"].decode()
    assert hardware.count("\\\\") == 5 and "Readout error" in hardware
    assert "Spambase & 4,601 & 0.911" in files[pt.PAPER / "generated" / "stage2-datasets.tex"].decode()
    assert (pt.PAPER / "figures" / "breast_cancer-noise-sweep.png") in files
    assert "mean accuracy loss is at most 0.19 pp on any dataset" in readme
    assert "Worst single seed: 0.94 pp (Ionosphere)" in readme
    assert (
        "(1,420× more, including its on-device bandwidth search; its final Gram matrices alone need 147,061, 86× more)"
        in readme
    )
    assert "published" not in readme.lower()  # no third-party quantum numbers in the README (user decision)
    with pytest.raises(pt.InputError, match="unknown status"):
        pt.target_values(pt.Values(), [{"id": "BT-1", "status": "maybe"}])


def test_committed_outputs_match_the_saved_results(paper_tables: ModuleType) -> None:
    assert paper_tables.main(["--check"]) == 0


def test_check_mode_reports_drift_and_write_repairs_it(paper_tables: ModuleType, tmp_path: Path) -> None:
    pt = paper_tables
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{pt.README_BEGIN}\nold\n{pt.README_END}\n\nRest.\n", encoding="utf-8")
    paper = tmp_path / "paper"
    files = pt.build(readme=readme, paper=paper)
    assert len(pt.drift(files, paper)) == len(files)
    pt.write(files, paper)
    assert pt.drift(files, paper) == []
    (paper / "generated" / "leaderboard.tex").write_text("% edited by hand\n")
    (paper / "generated" / "old.tex").write_text("")
    problems = pt.drift(files, paper)
    assert any(p.startswith("differs") and p.endswith("leaderboard.tex") for p in problems)
    assert any(p.startswith("stale") and p.endswith("old.tex") for p in problems)
    pt.write(files, paper)
    assert pt.drift(files, paper) == []
    assert not (paper / "generated" / "old.tex").exists()
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# Title")
    assert text.endswith("Rest.\n")
    assert "| Breast Cancer |" in text
    assert "old" not in text


def test_missing_results_name_the_command(
    paper_tables: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(paper_tables.InputError, match="qlassifier benchmark --noise"):
        paper_tables.build(bench=tmp_path)
    monkeypatch.setattr(paper_tables, "BENCH", tmp_path)
    assert paper_tables.main(["--check"]) == 2
    assert "missing" in capsys.readouterr().err


def test_readme_without_markers_is_an_input_error(paper_tables: ModuleType) -> None:
    with pytest.raises(paper_tables.InputError, match="results markers"):
        paper_tables.splice_readme("# README without markers\n", "block")


# Numbers that may appear in the paper's prose as typed text: definitions of the method and the protocol, names and
# references. Everything measured comes from \val{...}; a new typed number fails this test until it is generated.
DEFINITIONS = {
    "0",  # seeds are numbered from 0
    "0.5",  # chance level of balanced accuracy
    "1",  # within 1 pp (the BT-3 rule)
    "2",  # section 2 of the experiments README
    "3",  # optimisation level 3
    "8",  # the angle-scale cap at 8 features
    "7", "9",  # project phases: 7 (research loop), 9 (the shipped run)
    "0005", "0006",  # ADR numbers
    "7.13",  # TDD section on reproducibility
    "2026",  # date
    "2.0", "3.12",  # Apache-2.0, Python 3.12
    "e05", "e07", "e11", "e12", "e14",  # experiment ids
}  # fmt: skip


def prose_numbers(tex: str) -> set[str]:
    """Numbers typed in the body of main.tex outside math, commands' arguments and comments."""
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    body = re.sub(r"(?<!\\)%.*", "", body)
    body = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
    body = re.sub(r"\$[^$]*\$", " ", body)
    body = re.sub(r"\\(val|cite|ref|label|input|includegraphics|path|code|begin|end)(\[[^\]]*\])?\{[^}]*\}", " ", body)
    body = re.sub(r"\\href\{[^}]*\}", " ", body)
    body = re.sub(r"\\[a-zA-Z]+", " ", body)
    body = re.sub(r"BT-\d+", " ", body)
    return set(re.findall(r"(?<![A-Za-z])e?\d+(?:[.,]\d+)*", body))


def test_paper_prose_has_no_typed_results() -> None:
    assert prose_numbers(MAIN.read_text(encoding="utf-8")) - DEFINITIONS == set()


def test_prose_number_scan_finds_typed_numbers() -> None:
    sample = r"\begin{document} We got 0.948 here, $x = 3$ there, \val{bc.ba} % 0.99 in a comment"
    sample += "\n" + r"\end{document}"
    assert prose_numbers(sample) == {"0.948"}


def test_wins_chart_draws_every_dataset(tmp_path: Path) -> None:
    """``scripts/wins_chart.py``: the headline chart of README and paper, from the saved results."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("wins_chart", ROOT / "scripts" / "wins_chart.py")
    assert spec is not None and spec.loader is not None
    wins = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wins)
    data = wins.rows()
    assert [r["label"] for r in data] == [
        "Breast Cancer", "Protein angles", "Heart", "Sonar", "Ionosphere", "Banknote\u2020", "Haberman\u2020", "Spambase\u2020"
    ]  # fmt: skip
    margins = [wins.margin(r) for r in data]
    assert margins == sorted(margins, reverse=True)  # largest gain over the standard quantum baseline first
    assert all(r["qlassifier"][0] > r["zz_qsvc"][0] for r in data)  # the first claim of the README
    assert len(wins.draw(data).axes[0].patches) == 3 * len(data)  # paper: ZZ, Qlassifier, best classical
    assert len(wins.draw(data, classical=False).axes[0].patches) == 2 * len(data)  # README hero: quantum only
    assert wins.main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "wins.pdf").stat().st_size > 1000 and (tmp_path / "wins.png").stat().st_size > 1000
