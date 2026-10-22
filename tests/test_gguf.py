"""GGUF header tests — parse the three real Phase-B GGUFs."""
import os

import pytest

from bitblame import gguf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "data", "models")
FILES = ("SmolLM2-135M-f16.gguf", "SmolLM2-135M-q4km.gguf",
         "SmolLM2-135M-recipe.gguf")


@pytest.mark.parametrize("fname", FILES)
def test_header_parses_and_tensor_count(fname):
    path = os.path.join(MODELS, fname)
    h = gguf.read_gguf_header(path)
    assert h["version"] == 3
    assert h["file_bytes"] == os.path.getsize(path)
    assert len(h["tensors"]) == 272
    assert "token_embd.weight" in h["tensors"]
    assert h["tensors"]["blk.0.attn_q.weight"]["type"] in (
        "F16", "Q4_K", "Q5_0", "Q5_1", "Q8_0", "Q4_0")
    assert h["metadata"].get("general.architecture") == "llama"


def test_f16_types_and_sizes():
    h = gguf.read_gguf_header(os.path.join(MODELS, FILES[0]))
    assert h["tensors"]["token_embd.weight"]["type"] == "F16"
    assert h["tensors"]["blk.0.attn_q.weight"]["type"] == "F16"
    assert h["file_bytes"] == 270885376


def test_q4km_tied_embedding_lands_q8_0():
    # tied token_embd: Q6_K request demoted by fallback (576 % 256 != 0)
    h = gguf.read_gguf_header(os.path.join(MODELS, FILES[1]))
    assert h["tensors"]["token_embd.weight"]["type"] == "Q8_0"


def test_recipe_differs_only_on_overridden_blocks():
    base = gguf.read_gguf_header(os.path.join(MODELS, FILES[1]))
    recipe = gguf.read_gguf_header(os.path.join(MODELS, FILES[2]))
    assert set(base["tensors"]) == set(recipe["tensors"])
    assert recipe["file_bytes"] < base["file_bytes"]


def test_bytes_per_param():
    assert gguf.bytes_per_param("Q4_K") == pytest.approx(144 / 256)
    assert gguf.bytes_per_param("Q8_0") == pytest.approx(34 / 32)
    assert gguf.bytes_per_param("F16") == 2.0
    with pytest.raises(ValueError):
        gguf.bytes_per_param("NOPE")


def test_bad_magic_rejected(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"XXXX" + b"\x00" * 64)
    with pytest.raises(gguf.GGUFError):
        gguf.read_gguf_header(str(p))
