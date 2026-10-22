"""CLI tests — help text for every subcommand + one tiny end-to-end
run (scan -> probes -> recipe -> validate -> report) on the built-in
``tiny`` model. Real computations throughout; the battery leg runs a
single MMLU item through three arms."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
CALIB_DOC = ("The quick brown fox jumps over the lazy dog. Pack my box "
             "with five dozen liquor jugs. How vexingly quick daft "
             "zebras jump! Bright vixens jump; dozy fowl quack. "
             "A wizard's job is to vex chumps quickly in fog. ")


def run_cli(*argv, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "bitblame.cli", *argv],
        capture_output=True, text=True, cwd=cwd or ROOT,
        env=env, timeout=1200)


@pytest.mark.parametrize("argv", [
    ["--help"],
    ["scan", "--help"],
    ["probes", "--help"],
    ["recipe", "--help"],
    ["validate", "--help"],
    ["report", "--help"],
])
def test_help_works(argv):
    r = run_cli(*argv)
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout.lower()


def test_cli_tiny_end_to_end(tmp_path):
    calib = tmp_path / "calib.txt"
    calib.write_text(CALIB_DOC + "\n" + CALIB_DOC.replace("fox", "cat") + "\n")
    scan = tmp_path / "scan.npz"
    r = run_cli("scan", "--model", "tiny", "--calib", str(calib),
                "--out", str(scan), "--n-seqs", "2", "--seq-len", "16",
                "--batch-size", "2", "--n-draws", "1", "--n-null-draws", "1")
    assert r.returncode == 0, r.stderr
    assert scan.exists()

    probes = tmp_path / "probes.json"
    r = run_cli("probes", "--scan", str(scan), "--model", "tiny",
                "--calib", str(calib), "--out", str(probes),
                "--top-k", "2", "--bits", "3", "4", "--bootstrap-b", "50")
    assert r.returncode == 0, r.stderr
    analysis = json.loads(probes.read_text())
    assert len(analysis["unit_rows"]) == 6
    assert set(analysis["stage2"]) <= {u["id"] for u in analysis["unit_rows"]}
    assert len(analysis["stage2"]) == 2

    recipe = tmp_path / "recipe.json"
    r = run_cli("recipe", "--probes", str(probes), "--out", str(recipe))
    assert r.returncode == 0, r.stderr
    rec = json.loads(recipe.read_text())
    assert set(rec["alloc_types"]) == {u["id"] for u in analysis["unit_rows"]}
    assert (tmp_path / "recipe.sh").exists()
    assert "llama-quantize" in (tmp_path / "recipe.sh").read_text()

    validation = tmp_path / "validation.json"
    r = run_cli("validate", "--recipe", str(recipe), "--model", "tiny",
                "--data-dir", os.path.join(ROOT, "data"),
                "--out", str(validation),
                "--categories", "mmlu", "--max-items", "1",
                "--max-new-tokens", "4", "--bootstrap-b", "200")
    assert r.returncode == 0, r.stderr
    val = json.loads(validation.read_text())
    assert val["n_items"] == 1
    assert val["primary"]["verdict"] in (
        "SIGNIFICANT_GAIN", "NO_SIGNIFICANT_DIFFERENCE",
        "SIGNIFICANT_REGRESSION")

    report = tmp_path / "report.html"
    r = run_cli("report", "--runs", str(tmp_path), "--out", str(report))
    assert r.returncode == 0, r.stderr
    assert report.exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.png").exists()
    assert "BitBlame" in report.read_text()


def test_cli_recipe_infeasible_budget_fails(tmp_path):
    calib = tmp_path / "calib.txt"
    calib.write_text(CALIB_DOC + "\n" + CALIB_DOC.replace("fox", "cat") + "\n")
    scan = tmp_path / "scan.npz"
    assert run_cli("scan", "--model", "tiny", "--calib", str(calib),
                   "--out", str(scan), "--n-seqs", "2", "--seq-len", "16",
                   "--batch-size", "2", "--n-draws", "1",
                   "--n-null-draws", "1").returncode == 0
    probes = tmp_path / "probes.json"
    assert run_cli("probes", "--scan", str(scan), "--model", "tiny",
                   "--calib", str(calib), "--out", str(probes),
                   "--top-k", "2", "--bits", "3", "4",
                   "--bootstrap-b", "50").returncode == 0
    r = run_cli("recipe", "--probes", str(probes),
                "--out", str(tmp_path / "recipe.json"), "--budget", "0.0001")
    assert r.returncode != 0
    assert "infeasible" in (r.stderr + r.stdout).lower()
