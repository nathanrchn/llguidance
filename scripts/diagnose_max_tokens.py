#!/usr/bin/env python3
"""
Self-contained diagnostic for the maxTokens body cap on JSON strings.

Run this on the cluster (inside an SGLang worker venv) to definitively
show whether the body lexeme cap is being enforced in your environment
with your tokenizer:

    python diagnose_max_tokens.py LiquidAI/liquidthink_tokenizer 64

The script:
  1. Loads the given HF tokenizer.
  2. Builds a JSON schema with `{"analysis": {maxTokens: N}}`.
  3. Constructs an llguidance LLInterpreter directly (bypasses SGLang).
  4. Walks an enormous backslash-only target through the interpreter,
     greedily consuming whichever target token is allowed at each step.
  5. Reports:
       * llguidance file path (proves which build is loaded)
       * total target tokens
       * how many LLM tokens the *body* lexeme consumed
       * the produced byte stream (truncated for readability)
       * whether the produced bytes parse as valid JSON
       * for the parsed analysis: byte length, last 32 chars

If the cap is being enforced you should see roughly N body tokens and
the produced bytes parse cleanly. If you see body tokens >> N, or the
JSON fails to parse, the cluster is running stale llguidance OR the
schema being sent to SGLang doesn't carry the maxTokens key (most
common issue).
"""

import inspect
import json
import sys

import llguidance
import llguidance.hf
from transformers import AutoTokenizer


def main(tokenizer_name: str, max_tokens: int) -> None:
    print(f"=== llguidance build ===")
    print(f"  module file:    {inspect.getsourcefile(llguidance)}")
    src_dir = "/".join((inspect.getsourcefile(llguidance) or "").split("/")[:-2])

    # Verify the defer fix is present in the installed binary by reading
    # the .so for the symbol name. Crude but works.
    so_path = None
    try:
        from llguidance import _lib  # type: ignore
        so_path = _lib.__file__
    except Exception:
        pass
    if so_path:
        print(f"  native lib:     {so_path}")
        try:
            with open(so_path, "rb") as f:
                blob = f.read()
            print(
                f"  defer fix:      "
                f"{'is_accepting_for_lexeme' in blob.decode('latin1', 'replace')}"
            )
        except Exception as e:
            print(f"  defer fix:      could not check ({e})")
    print()

    print(f"=== tokenizer ===")
    print(f"  name:           {tokenizer_name}")
    hf_tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    ll_tok = llguidance.LLTokenizer(llguidance.hf.from_tokenizer(hf_tok))
    print(f"  vocab size:     {ll_tok.vocab_size}")
    print()

    schema = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "maxTokens": max_tokens},
            "score": {"type": "integer"},
        },
        "required": ["analysis", "score"],
        "additionalProperties": False,
    }
    print(f"=== schema ===")
    print(json.dumps(schema, indent=2))
    print()

    grammar = json.dumps(
        {
            "grammars": [{"json_schema": schema}],
        }
    )
    interp = llguidance.LLInterpreter(ll_tok, grammar, log_level=0)
    interp.start_without_prompt()

    # Build a target with a long run of escaped backslashes — the failure
    # pattern from the cluster logs.
    body = "\\\\" * 1000
    target = json.dumps({"analysis": body, "score": 7})
    target_tokens = hf_tok.encode(target, add_special_tokens=False)
    print(f"=== target ===")
    print(f"  bytes:          {len(target)}")
    print(f"  tokens:         {len(target_tokens)}")
    print()

    produced = bytearray()
    body_token_count = 0
    in_body = False
    body_started_at = 0
    t_idx = 0
    steps = 0

    while not interp.is_accepting():
        steps += 1
        if steps > 5000:
            raise RuntimeError(
                f"parser did not accept in 5000 steps; produced {len(produced)} bytes"
            )

        mask_bytes, _ = interp.compute_mask()
        if mask_bytes is None:
            break  # parser stopped

        # Pick the next target token if allowed; otherwise pick the
        # shortest non-whitespace allowed token.
        def is_allowed(tok_id: int) -> bool:
            byte_idx, bit_idx = divmod(tok_id, 8)
            return byte_idx < len(mask_bytes) and (
                (mask_bytes[byte_idx] >> bit_idx) & 1
            ) == 1

        chosen = None
        if t_idx < len(target_tokens) and is_allowed(target_tokens[t_idx]):
            chosen = target_tokens[t_idx]
            t_idx += 1
        else:
            best = None
            best_len = 1 << 30
            for tok_id in range(ll_tok.vocab_size):
                if not is_allowed(tok_id):
                    continue
                tok_bytes = ll_tok.decode_bytes([tok_id])
                if not tok_bytes or all(b in b" \t\n\r" for b in tok_bytes):
                    continue
                if len(tok_bytes) < best_len:
                    best = tok_id
                    best_len = len(tok_bytes)
                    if best_len == 1:
                        break
            if best is None:
                raise RuntimeError(
                    f"no usable token at step {steps}; produced {len(produced)} bytes"
                )
            chosen = best
            if t_idx < len(target_tokens):
                t_idx += 1

        tok_bytes = ll_tok.decode_bytes([chosen])
        produced.extend(tok_bytes)

        # Track body lexeme tokens by scanning the produced bytes for the
        # `"analysis":"` opening and the next unescaped `"`.
        if not in_body:
            opening = b'"analysis":"'
            pos = produced.find(opening)
            if pos >= 0:
                in_body = True
                body_started_at = pos + len(opening)
        if in_body:
            j = body_started_at
            escaped = False
            still_in_body = True
            while j < len(produced):
                c = produced[j]
                if escaped:
                    escaped = False
                elif c == ord("\\"):
                    escaped = True
                elif c == ord('"'):
                    still_in_body = False
                    break
                j += 1
            if still_in_body:
                body_token_count += 1
            else:
                in_body = False

        ok = interp.consume_token(chosen)
        if not ok:
            raise RuntimeError(f"consume_token failed at step {steps}")

    print(f"=== result ===")
    print(f"  total steps:    {steps}")
    print(f"  body LLM toks:  {body_token_count}  (cap = {max_tokens})")
    print(f"  produced bytes: {len(produced)}")
    if len(produced) > 800:
        print(f"  produced [head]: {produced[:400]!r}")
        print(f"  produced [tail]: {produced[-400:]!r}")
    else:
        print(f"  produced:       {bytes(produced)!r}")
    print()

    try:
        parsed = json.loads(bytes(produced))
        analysis = parsed["analysis"]
        print(f"  json.loads:     OK")
        print(f"  analysis len:   {len(analysis)} chars")
        print(f"  analysis tail:  {analysis[-32:]!r}")
    except Exception as e:
        print(f"  json.loads:     FAILED — {e}")
        print()
        print("  *** This is the bug. The cluster's llguidance is not enforcing")
        print("  *** maxTokens at the requested boundary. ***")
        sys.exit(1)

    if body_token_count > max_tokens + 5:
        print()
        print(f"  *** body consumed {body_token_count} LLM tokens but the cap was")
        print(f"  *** {max_tokens} — this is a bug. The cluster is either running")
        print(f"  *** stale llguidance or the schema didn't carry maxTokens. ***")
        sys.exit(1)

    print()
    print("OK — cap is being enforced correctly in this environment.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: python diagnose_max_tokens.py <hf-tokenizer-name> [max_tokens=64]",
            file=sys.stderr,
        )
        sys.exit(2)
    tokenizer_name = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    main(tokenizer_name, max_tokens)
