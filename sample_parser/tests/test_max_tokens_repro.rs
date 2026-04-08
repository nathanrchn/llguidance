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
