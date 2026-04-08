// Regression test for the maxTokens-on-JSON-string lexeme-exit bug.
//
// Previously, when a JSON string carrying `maxTokens: N` hit its budget
// mid-content, the parser silently allowed the lexeme to "phantom exit"
// without forcing the closing `"`. The resulting bytes were malformed JSON
// like `"analysis": "some content      , "rubric_scores": ...` — observed in
// the wild via SGLang grammar-constrained eval runs.
//
// The fix splits a maxTokens-bounded JSON string into three adjacent grammar
// nodes — `'"' + body_lexeme[max_tokens=N] + '"'` — so the body lexeme's regex
// is in an accepting state at every position and the surrounding rule forces
// the closing quote when the cap fires.
//
// This test feeds an over-budget value through the parser one token at a
// time and asserts that:
//   (a) the parser eventually rejects an over-budget body token, and
//   (b) at the rejection point the mask still allows a `"` (closing quote),
//       proving the parser is requiring the lexeme to terminate cleanly.

use llguidance::api::{GrammarInit, TopLevelGrammar};
use sample_parser::{get_parser_factory, get_tok_env};
use serde_json::json;

#[test]
fn maxtokens_string_forces_closing_quote() {
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

    // A string the model would normally write but which exceeds the cap.
    let target = r#"{"analysis": "this is a very long analysis that will definitely exceed four tokens of budget", "score": 5}"#;
    let tok_env = get_tok_env();
    let tokens = tok_env.tokenize(target);

    for (i, tok) in tokens.iter().enumerate() {
        let m = parser.compute_mask().unwrap();
        if !m.is_allowed(*tok) {
            // Find a token whose bytes contain a `"` — there must be at
            // least one allowed at this point, otherwise the parser would
            // have phantom-exited the string.
            let mut quote_token = None;
            for t in 0..m.len() as u32 {
                if !m.is_allowed(t) {
                    continue;
                }
                let bytes = tok_env.tok_trie().token(t);
                if bytes.contains(&b'"') {
                    quote_token = Some(t);
                    break;
                }
            }
            assert!(
                quote_token.is_some(),
                "max_tokens fired at step {i} but the mask offers no token containing `\"` — \
                 the parser allowed a phantom lexeme exit (the original bug). \
                 Number of allowed tokens: {}",
                (0..m.len() as u32).filter(|t| m.is_allowed(*t)).count()
            );
            return;
        }
        let bt = parser.consume_token(*tok).unwrap();
        assert_eq!(bt, 0, "unexpected backtrack at step {i}");
    }

    panic!(
        "parser accepted ALL {} tokens of an over-budget string — maxTokens cap not enforced",
        tokens.len()
    );
}
