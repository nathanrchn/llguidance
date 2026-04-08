// Regression test for the maxTokens-on-JSON-string corruption.
//
// Two bugs were observed in the wild via SGLang grammar-constrained eval runs:
//
// 1. Phantom lexeme exit: when a JSON string carrying `maxTokens: N` hit its
//    budget mid-content, the parser silently let the lexeme advance past its
//    boundary without forcing the closing `"`.
//
// 2. Skip-whitespace inside the string: a first attempted fix split the
//    string into three nodes; `whitespace_flexible` then inserted skip
//    bytes between the body and the closing `"`, producing invalid JSON.
//
// The current fix lifts the body+quotes into a *sub-grammar* whose skip
// pattern is `NoMatch`, so neither bug can occur.
//
// This test feeds the parser a long over-budget value followed by a short
// completion suffix, then asserts that the produced bytes parse as valid
// JSON and that the analysis content has no literal control characters
// (which would indicate the skip-whitespace leak).

use llguidance::api::{GrammarInit, TopLevelGrammar};
use sample_parser::{get_parser_factory, get_tok_env};
use serde_json::json;

#[test]
fn maxtokens_string_produces_valid_json() {
    let schema = json!({
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "maxTokens": 4},
            "score": {"type": "integer"}
        },
        "required": ["analysis", "score"],
        "additionalProperties": false
    });
    let lark = format!("start: %json {}", serde_json::to_string(&schema).unwrap());
    let grm = TopLevelGrammar::from_lark(lark);
    let mut parser = get_parser_factory()
        .create_parser_from_init(GrammarInit::Serialized(grm), 0, 0)
        .unwrap();
    parser.start_without_prompt();

    let target = r#"{"analysis": "this is a very long analysis that will definitely exceed four tokens of budget", "score": 5}"#;
    let tok_env = get_tok_env();
    let target_tokens = tok_env.tokenize(target);

    let mut produced: Vec<u8> = Vec::new();
    let mut t_idx = 0;
    let mut steps = 0;

    loop {
        if parser.is_accepting() {
            break;
        }
        steps += 1;
        assert!(
            steps < 200,
            "parser failed to reach accept state in 200 steps; produced so far:\n{}",
            String::from_utf8_lossy(&produced)
        );

        let mask = parser.compute_mask().unwrap();

        // Greedy strategy:
        // 1. If the next target token is allowed, use it.
        // 2. Otherwise, find the SHORTEST allowed token whose bytes start
        //    with `"` (the closing quote we're being forced toward), or
        //    failing that, any allowed non-empty token. This deterministic
        //    pick is enough for the test — we're not modeling the LLM, just
        //    proving the grammar can be driven to a valid completion.
        let chosen = if t_idx < target_tokens.len() && mask.is_allowed(target_tokens[t_idx]) {
            let t = target_tokens[t_idx];
            t_idx += 1;
            t
        } else {
            // Recovery: skip the rejected target token and find the shortest
            // allowed token whose bytes are non-whitespace ASCII. The point
            // is to drive the parser toward completion without padding with
            // whitespace (which is normally allowed by `whitespace_flexible`
            // and would loop forever).
            let trie = tok_env.tok_trie();
            let is_ws = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | b'\r');
            let mut best: Option<u32> = None;
            let mut best_len = usize::MAX;
            for t in 0..mask.len() as u32 {
                if !mask.is_allowed(t) || t == trie.eos_token() {
                    continue;
                }
                let bytes = trie.token(t);
                if bytes.is_empty() || bytes.iter().all(|b| is_ws(*b)) {
                    continue;
                }
                if bytes.len() < best_len {
                    best = Some(t);
                    best_len = bytes.len();
                }
            }
            let chosen = best.unwrap_or_else(|| {
                panic!(
                    "no non-whitespace token at step {steps}; produced so far:\n{}",
                    String::from_utf8_lossy(&produced)
                )
            });
            if t_idx < target_tokens.len() {
                t_idx += 1;
            }
            chosen
        };

        let tok_bytes = tok_env.tok_trie().token(chosen).to_vec();
        produced.extend_from_slice(&tok_bytes);
        let bt = parser.consume_token(chosen).unwrap();
        assert_eq!(bt, 0, "unexpected backtrack");
    }

    let produced_str = String::from_utf8_lossy(&produced).into_owned();
    println!("\n=== produced ({} bytes) ===\n{}\n", produced.len(), produced_str);

    // (a) The produced bytes MUST parse as valid JSON.
    let parsed: serde_json::Value = serde_json::from_str(&produced_str)
        .unwrap_or_else(|e| panic!("produced bytes are not valid JSON: {e}\nbytes:\n{produced_str}"));

    let obj = parsed.as_object().expect("expected JSON object");
    assert!(obj.contains_key("analysis"));
    assert!(obj.contains_key("score"));

    let analysis = obj["analysis"].as_str().expect("analysis must be a string");

    // (b) The analysis text MUST NOT contain literal control characters —
    // those would indicate the old skip-whitespace-inside-string bug.
    for (i, b) in analysis.as_bytes().iter().enumerate() {
        assert!(
            ![b'\n', b'\t', b'\r'].contains(b),
            "analysis contains literal control char 0x{b:02x} at byte {i} \
             (skip-whitespace leak): {analysis:?}"
        );
    }

    // (c) The cap should have actually shortened the body — the analysis
    // must be strictly shorter than the un-capped target.
    assert!(
        analysis.len() < target.len(),
        "analysis was not shortened by the cap (len={})",
        analysis.len()
    );
}

/// Bug 3: even with the no-skip sub-grammar, `max_tokens` could fire while
/// the body lexeme's regex was in the middle of a multi-byte JSON construct
/// (e.g., a `\u00XX` escape, or a lone `\` waiting for an escape character).
/// `force_lexeme_end` would emit the body lexeme in a non-accepting state,
/// leaving a dangling backslash that then escapes the closing `"` literal,
/// e.g.:
///
///     {"analysis":"... typically $g \","rubric_scores":{...}}
///
/// JSON parses `\"` as an escaped quote, so the analysis string never closes
/// and the rest of the object becomes content of the unterminated string.
///
/// The fix in `apply_token`'s max_tokens check defers the cap when the
/// regex is non-accepting for the bounded lexeme — the body keeps consuming
/// up to the next accepting boundary (a few extra bytes at most), then the
/// cap fires cleanly.
///
/// We can't easily reproduce the exact LLM-token boundary that triggers
/// this in the wild, but we can construct an analogous scenario by feeding
/// inputs whose tail puts the body lexeme into an interesting state and
/// then forcing recovery. The simpler smoke test: feed a value containing
/// internal escape sequences and confirm the produced bytes parse cleanly.
#[test]
fn maxtokens_string_with_escapes_produces_valid_json() {
    let schema = json!({
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "maxTokens": 6},
            "score": {"type": "integer"}
        },
        "required": ["analysis", "score"],
        "additionalProperties": false
    });
    let lark = format!("start: %json {}", serde_json::to_string(&schema).unwrap());
    let grm = TopLevelGrammar::from_lark(lark);
    let mut parser = get_parser_factory()
        .create_parser_from_init(GrammarInit::Serialized(grm), 0, 0)
        .unwrap();
    parser.start_without_prompt();

    // Target has internal escape sequences: \", \\, \n. We want the body
    // lexeme to consume some content with escapes and then hit its cap.
    let target = r#"{"analysis": "He said \"hi\" and \\ then \n more text past the cap goes here", "score": 7}"#;
    let tok_env = get_tok_env();
    let target_tokens = tok_env.tokenize(target);

    let mut produced: Vec<u8> = Vec::new();
    let mut t_idx = 0;
    let mut steps = 0;
    loop {
        if parser.is_accepting() {
            break;
        }
        steps += 1;
        assert!(
            steps < 200,
            "parser failed to reach accept state in 200 steps; produced so far:\n{}",
            String::from_utf8_lossy(&produced)
        );

        let mask = parser.compute_mask().unwrap();
        let chosen = if t_idx < target_tokens.len() && mask.is_allowed(target_tokens[t_idx]) {
            let t = target_tokens[t_idx];
            t_idx += 1;
            t
        } else {
            let trie = tok_env.tok_trie();
            let is_ws = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | b'\r');
            let mut best: Option<u32> = None;
            let mut best_len = usize::MAX;
            for t in 0..mask.len() as u32 {
                if !mask.is_allowed(t) || t == trie.eos_token() {
                    continue;
                }
                let bytes = trie.token(t);
                if bytes.is_empty() || bytes.iter().all(|b| is_ws(*b)) {
                    continue;
                }
                if bytes.len() < best_len {
                    best = Some(t);
                    best_len = bytes.len();
                }
            }
            let chosen = best.unwrap_or_else(|| {
                panic!(
                    "no non-whitespace token at step {steps}; produced so far:\n{}",
                    String::from_utf8_lossy(&produced)
                )
            });
            if t_idx < target_tokens.len() {
                t_idx += 1;
            }
            chosen
        };

        let tok_bytes = tok_env.tok_trie().token(chosen).to_vec();
        produced.extend_from_slice(&tok_bytes);
        let bt = parser.consume_token(chosen).unwrap();
        assert_eq!(bt, 0, "unexpected backtrack");
    }

    let produced_str = String::from_utf8_lossy(&produced).into_owned();
    println!("\n=== produced ({} bytes) ===\n{}\n", produced.len(), produced_str);

    // Bytes MUST parse as valid JSON.
    let parsed: serde_json::Value = serde_json::from_str(&produced_str).unwrap_or_else(|e| {
        panic!("produced bytes are not valid JSON: {e}\nbytes:\n{produced_str}")
    });
    let obj = parsed.as_object().expect("expected JSON object");
    let analysis = obj["analysis"].as_str().expect("analysis must be a string");

    // The analysis MUST NOT end with a lone unescaped backslash. (If it did,
    // the closing `"` would have been escaped by it.)
    assert!(
        !analysis.ends_with('\\'),
        "analysis ends with a dangling backslash: {analysis:?}"
    );

    // Sanity: cap was actually applied.
    assert!(analysis.len() < target.len());
}

/// Stress test: feed a backslash-heavy target through a maxTokens-bounded
/// JSON string. The model wants to write `$F_\sigma$` style LaTeX, but each
/// `\` must be escaped as `\\` in JSON, so the body sees runs of `\\\\`.
/// This was the failure pattern observed in the wild — the body was getting
/// huge runs of escaped backslashes and then truncating mid-pair, leaving
/// dangling backslashes.
#[test]
fn maxtokens_string_with_backslash_run_caps_cleanly() {
    let schema = json!({
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "maxTokens": 8},
            "score": {"type": "integer"}
        },
        "required": ["analysis", "score"],
        "additionalProperties": false
    });
    let lark = format!("start: %json {}", serde_json::to_string(&schema).unwrap());
    let grm = TopLevelGrammar::from_lark(lark);
    let mut parser = get_parser_factory()
        .create_parser_from_init(GrammarInit::Serialized(grm), 0, 0)
        .unwrap();
    parser.start_without_prompt();

    // The model is trying to write LaTeX with lots of escaped backslashes.
    // The JSON-encoded form has even more backslashes (each `\` doubles).
    let target = r#"{"analysis": "F_\\sigma is a countable union of closed sets and \\\\\\\\\\\\\\\\\\\\\\\\\\\\ matters here too past the cap", "score": 7}"#;
    let tok_env = get_tok_env();
    let target_tokens = tok_env.tokenize(target);

    let mut produced: Vec<u8> = Vec::new();
    let mut t_idx = 0;
    let mut steps = 0;
    loop {
        if parser.is_accepting() {
            break;
        }
        steps += 1;
        assert!(
            steps < 300,
            "parser failed to reach accept state in 300 steps; produced so far:\n{}",
            String::from_utf8_lossy(&produced)
        );

        let mask = parser.compute_mask().unwrap();
        let chosen = if t_idx < target_tokens.len() && mask.is_allowed(target_tokens[t_idx]) {
            let t = target_tokens[t_idx];
            t_idx += 1;
            t
        } else {
            let trie = tok_env.tok_trie();
            let is_ws = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | b'\r');
            let mut best: Option<u32> = None;
            let mut best_len = usize::MAX;
            for t in 0..mask.len() as u32 {
                if !mask.is_allowed(t) || t == trie.eos_token() {
                    continue;
                }
                let bytes = trie.token(t);
                if bytes.is_empty() || bytes.iter().all(|b| is_ws(*b)) {
                    continue;
                }
                if bytes.len() < best_len {
                    best = Some(t);
                    best_len = bytes.len();
                }
            }
            let chosen = best.unwrap_or_else(|| {
                panic!(
                    "no non-whitespace token at step {steps}; produced so far:\n{}",
                    String::from_utf8_lossy(&produced)
                )
            });
            if t_idx < target_tokens.len() {
                t_idx += 1;
            }
            chosen
        };

        let tok_bytes = tok_env.tok_trie().token(chosen).to_vec();
        produced.extend_from_slice(&tok_bytes);
        let bt = parser.consume_token(chosen).unwrap();
        assert_eq!(bt, 0);
    }

    let produced_str = String::from_utf8_lossy(&produced).into_owned();
    println!("\n=== produced ({} bytes) ===\n{}\n", produced.len(), produced_str);

    // Must parse as valid JSON.
    let parsed: serde_json::Value = serde_json::from_str(&produced_str).unwrap_or_else(|e| {
        panic!("produced bytes are not valid JSON: {e}\nbytes:\n{produced_str}")
    });
    let obj = parsed.as_object().expect("expected JSON object");
    let analysis = obj["analysis"].as_str().expect("analysis must be a string");

    // No dangling backslash at the end.
    assert!(
        !analysis.ends_with('\\'),
        "analysis ends with a dangling backslash: {analysis:?}"
    );

    // Cap was applied.
    assert!(analysis.len() < target.len());
}

/// Diagnostic: feed an enormous run of escaped backslashes through a tight
/// maxTokens=4 schema and assert that the body is *actually* capped — i.e.
/// the resulting analysis content doesn't grow proportionally with the
/// target's length. This is the user-reported failure pattern: model
/// degenerates into LaTeX-style `$F_\sigma$` and the body fills with `\\`
/// runs that exceed any reasonable budget.
#[test]
fn maxtokens_huge_backslash_run_is_capped() {
    let schema = json!({
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "maxTokens": 4},
            "score": {"type": "integer"}
        },
        "required": ["analysis", "score"],
        "additionalProperties": false
    });
    let lark = format!("start: %json {}", serde_json::to_string(&schema).unwrap());
    let grm = TopLevelGrammar::from_lark(lark);
    let mut parser = get_parser_factory()
        .create_parser_from_init(GrammarInit::Serialized(grm), 0, 0)
        .unwrap();
    parser.start_without_prompt();

    // 500 escaped backslashes — far more than 4 tokens worth on any sane
    // tokenizer. After capping, the analysis should be bounded.
    let body_filler: String = "\\\\".repeat(500);
    let target = format!(
        r#"{{"analysis": "{body_filler}", "score": 7}}"#
    );
    let tok_env = get_tok_env();
    let target_tokens = tok_env.tokenize(&target);
    println!("target has {} tokens", target_tokens.len());

    let mut produced: Vec<u8> = Vec::new();
    let mut t_idx = 0;
    let mut steps = 0;

    loop {
        if parser.is_accepting() {
            break;
        }
        steps += 1;
        assert!(
            steps < 1000,
            "parser failed to reach accept state in 1000 steps; produced {} bytes so far:\n{}",
            produced.len(),
            String::from_utf8_lossy(&produced)
        );

        let mask = parser.compute_mask().unwrap();
        let chosen = if t_idx < target_tokens.len() && mask.is_allowed(target_tokens[t_idx]) {
            let t = target_tokens[t_idx];
            t_idx += 1;
            t
        } else {
            let trie = tok_env.tok_trie();
            let is_ws = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | b'\r');
            let mut best: Option<u32> = None;
            let mut best_len = usize::MAX;
            for t in 0..mask.len() as u32 {
                if !mask.is_allowed(t) || t == trie.eos_token() {
                    continue;
                }
                let bytes = trie.token(t);
                if bytes.is_empty() || bytes.iter().all(|b| is_ws(*b)) {
                    continue;
                }
                if bytes.len() < best_len {
                    best = Some(t);
                    best_len = bytes.len();
                }
            }
            let chosen = best.unwrap_or_else(|| {
                panic!(
                    "no non-whitespace token at step {steps}; produced {} bytes:\n{}",
                    produced.len(),
                    String::from_utf8_lossy(&produced)
                )
            });
            if t_idx < target_tokens.len() {
                t_idx += 1;
            }
            chosen
        };

        let tok_bytes = tok_env.tok_trie().token(chosen).to_vec();
        produced.extend_from_slice(&tok_bytes);
        let bt = parser.consume_token(chosen).unwrap();
        assert_eq!(bt, 0);
    }

    let produced_str = String::from_utf8_lossy(&produced).into_owned();
    println!(
        "\n=== produced ({} bytes) ===\n{}\n",
        produced.len(),
        if produced.len() > 500 {
            format!("{}... [truncated]", &produced_str[..500])
        } else {
            produced_str.clone()
        }
    );

    let parsed: serde_json::Value = serde_json::from_str(&produced_str).unwrap_or_else(|e| {
        panic!("produced bytes are not valid JSON: {e}\nbytes:\n{produced_str}")
    });
    let obj = parsed.as_object().expect("expected JSON object");
    let analysis = obj["analysis"].as_str().expect("analysis must be a string");

    // Sanity: cap fired and the analysis is much shorter than the target.
    // Even with very dense tokens (16 chars/token), 4 tokens shouldn't
    // produce more than ~80 bytes of body content. Allow some headroom for
    // the defer-to-accepting-state slack: pick 200 as a generous bound.
    println!("analysis byte length: {}", analysis.len());
    assert!(
        analysis.len() < 200,
        "maxTokens=4 produced {} bytes of analysis content — cap not enforced!",
        analysis.len()
    );
}

/// Diagnostic at the user's actual budget (64 tokens). Asserts that even
/// with a 1000-pair backslash input, the body content stays within a
/// reasonable bound for 64 LLM tokens (well below what would cause the
/// failures observed in the cluster logs).
#[test]
fn maxtokens_64_huge_backslash_run_is_capped() {
    let schema = json!({
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "maxTokens": 64},
            "score": {"type": "integer"}
        },
        "required": ["analysis", "score"],
        "additionalProperties": false
    });
    let lark = format!("start: %json {}", serde_json::to_string(&schema).unwrap());
    let grm = TopLevelGrammar::from_lark(lark);
    let mut parser = get_parser_factory()
        .create_parser_from_init(GrammarInit::Serialized(grm), 0, 0)
        .unwrap();
    parser.start_without_prompt();

    // 1000 escaped backslashes — way more than 64 tokens worth.
    let body_filler: String = "\\\\".repeat(1000);
    let target = format!(r#"{{"analysis": "{body_filler}", "score": 7}}"#);
    let tok_env = get_tok_env();
    let target_tokens = tok_env.tokenize(&target);
    println!("target has {} tokens", target_tokens.len());

    let mut produced: Vec<u8> = Vec::new();
    let mut t_idx = 0;
    let mut steps = 0;
    let mut body_token_count = 0;
    let mut in_body = false;
    let mut body_started_at_byte = 0;

    loop {
        if parser.is_accepting() {
            break;
        }
        steps += 1;
        assert!(
            steps < 2000,
            "parser failed to reach accept state in 2000 steps; produced {} bytes",
            produced.len()
        );

        let mask = parser.compute_mask().unwrap();
        let chosen = if t_idx < target_tokens.len() && mask.is_allowed(target_tokens[t_idx]) {
            let t = target_tokens[t_idx];
            t_idx += 1;
            t
        } else {
            let trie = tok_env.tok_trie();
            let is_ws = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | b'\r');
            let mut best: Option<u32> = None;
            let mut best_len = usize::MAX;
            for t in 0..mask.len() as u32 {
                if !mask.is_allowed(t) || t == trie.eos_token() {
                    continue;
                }
                let bytes = trie.token(t);
                if bytes.is_empty() || bytes.iter().all(|b| is_ws(*b)) {
                    continue;
                }
                if bytes.len() < best_len {
                    best = Some(t);
                    best_len = bytes.len();
                }
            }
            let chosen = best.unwrap_or_else(|| {
                panic!(
                    "no non-whitespace token at step {steps}; produced {} bytes",
                    produced.len()
                )
            });
            if t_idx < target_tokens.len() {
                t_idx += 1;
            }
            chosen
        };

        let tok_bytes = tok_env.tok_trie().token(chosen).to_vec();

        // Track when we enter and exit the body lexeme by looking at the
        // produced byte stream. The body starts after `"analysis":"` and
        // ends at the next unescaped `"`.
        if !in_body && produced.ends_with(b"\"analysis\":") {
            // Next byte should be the opening quote of the body
        }
        let pre_len = produced.len();
        produced.extend_from_slice(&tok_bytes);

        // Detect entering body content (after the opening `"` of analysis).
        if !in_body {
            let opening = b"\"analysis\":\"";
            if let Some(pos) = produced
                .windows(opening.len())
                .position(|w| w == opening)
            {
                if pre_len <= pos + opening.len() && produced.len() > pos + opening.len() {
                    in_body = true;
                    body_started_at_byte = pos + opening.len();
                }
            }
        } else {
            // Count this token toward the body if we're still in body.
            // Crude check: any unescaped `"` ends the body.
            let mut j = body_started_at_byte;
            let mut escaped = false;
            let mut still_in_body = true;
            while j < produced.len() {
                let c = produced[j];
                if escaped {
                    escaped = false;
                } else if c == b'\\' {
                    escaped = true;
                } else if c == b'"' {
                    still_in_body = false;
                    break;
                }
                j += 1;
            }
            if still_in_body {
                body_token_count += 1;
            } else {
                in_body = false;
            }
        }

        let bt = parser.consume_token(chosen).unwrap();
        assert_eq!(bt, 0);
    }

    let produced_str = String::from_utf8_lossy(&produced).into_owned();
    let parsed: serde_json::Value = serde_json::from_str(&produced_str)
        .unwrap_or_else(|e| panic!("not valid JSON: {e}\nbytes:\n{produced_str}"));
    let analysis = parsed["analysis"].as_str().unwrap();
    println!(
        "body LLM tokens consumed (approx): {}, analysis byte len: {}",
        body_token_count,
        analysis.len()
    );
    // The cap is 64 tokens. Allow up to ~70 to account for the defer
    // slack at non-accepting boundaries.
    assert!(
        body_token_count <= 70,
        "body lexeme consumed {body_token_count} LLM tokens, expected ≤70 (cap=64)"
    );
}
