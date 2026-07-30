#!/usr/bin/env python3
"""Generate a MIDI file from a text description, using amaai-lab/text2midi.

WHY A SEPARATE SCRIPT AND A SEPARATE VENV

text2midi needs its own transformers/torch/miditok stack, and ComfyUI's venv is
load-bearing for every other engine in soundmon — SFX, music, songs, narration.
Perturbing it to add a text-to-MIDI model would risk all of that for one optional
feature, so this runs in an isolated venv and is invoked as a subprocess.

The model was chosen over MuseCoco (the obvious candidate for text->symbolic
music) for a practical reason rather than a musical one: MuseCoco pins
fairseq==0.10.2 and transformers==4.26.0, a 2021 stack that will not build against
a modern torch. text2midi ships its own architecture file, so it depends on no
framework version at all.

SETUP

    python3 -m venv ~/.cache/soundmon/t2m
    ~/.cache/soundmon/t2m/bin/pip install torch --index-url \\
        https://download.pytorch.org/whl/cpu
    ~/.cache/soundmon/t2m/bin/pip install transformers sentencepiece \\
        huggingface_hub miditok symusic
    # then fetch pytorch_model.bin, vocab_remi.pkl and transformer_model.py from
    # https://huggingface.co/amaai-lab/text2midi into ~/.cache/soundmon/text2midi

USAGE

    ~/.cache/soundmon/t2m/bin/python tools/text2midi-gen.py \\
        "A brooding dungeon theme in D minor, slow tempo, church organ and strings" \\
        --out dungeon.mid

The output is an ordinary .mid, so it feeds straight into `soundmon --from-midi`
and gets played on an OPL3 or a 2A03 like any other MIDI.
"""
import argparse
import os
import pickle
import sys

HOME = os.path.expanduser("~/.cache/soundmon/text2midi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--out", default="output.mid")
    ap.add_argument("--max-len", type=int, default=1200,
                    help="tokens to generate; ~1200 is a short piece")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--home", default=HOME)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    sys.path.insert(0, a.home)
    try:
        import torch
        import torch.nn as nn
        from transformers import T5Tokenizer
        from transformer_model import Transformer
    except ImportError as e:
        sys.exit(f"text2midi deps missing ({e}). See the setup block in this file.")

    model_path = os.path.join(a.home, "pytorch_model.bin")
    vocab_path = os.path.join(a.home, "vocab_remi.pkl")
    for p in (model_path, vocab_path):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — fetch it from huggingface.co/amaai-lab/text2midi")

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(vocab_path, "rb") as fh:
        r_tokenizer = pickle.load(fh)
    vocab_size = len(r_tokenizer)

    # COMPAT SHIM. vocab_remi.pkl is a pickled miditok tokenizer from an older
    # release, and unpickling restores the object WITHOUT running __init__ — so
    # every config field miditok has added since is simply absent, and decode
    # dies on the first one it touches.
    #
    # Enumerating the missing fields by hand is whack-a-mole (it took three
    # rounds before I stopped). Instead, build a DEFAULT config with the installed
    # miditok and copy across anything the pickle lacks: that covers every field
    # added since, in one step, and keeps working when miditok adds more.
    cfg = getattr(r_tokenizer, "config", None)
    if cfg is not None:
        try:
            from miditok import TokenizerConfig
            fresh = TokenizerConfig()
            added = []
            for k, v in vars(fresh).items():
                if not hasattr(cfg, k):
                    setattr(cfg, k, v)
                    added.append(k)
            if added:
                print(f"  compat: backfilled {len(added)} config fields "
                      f"({', '.join(added[:6])}{'...' if len(added) > 6 else ''})",
                      file=sys.stderr)
        except Exception as e:
            print(f"  compat shim failed: {e}", file=sys.stderr)

    # Architecture parameters are fixed by the checkpoint, not free choices;
    # they come from the model card's quickstart.
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tok = T5Tokenizer.from_pretrained("google/flan-t5-base")

    print(f"  device={device}  vocab={vocab_size}  prompt={a.prompt[:70]!r}",
          file=sys.stderr)
    inputs = tok(a.prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True,
                                         padding_value=0).to(device)
    attn = nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True,
                                     padding_value=0).to(device)
    with torch.no_grad():
        out = model.generate(input_ids, attn, max_len=a.max_len,
                             temperature=a.temperature)
    midi = r_tokenizer.decode(out[0].tolist())
    midi.dump_midi(a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
