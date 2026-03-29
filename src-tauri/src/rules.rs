// T1 rules engine — per-word rules applied to all ASR output.
// Since Nemotron and TDT output punctuation natively, T1 is light:
// filler removal, explicit voice commands, whitespace normalization.

/// Apply T1 rules to ASR output text.
/// Called on both Nemotron streaming and TDT refined output before display.
pub fn apply_t1(text: &str) -> String {
    let text = apply_text_commands(text);
    let text = remove_fillers(&text);
    normalize_whitespace(&text)
}

/// Check if text is a buffer command (not content to display).
/// Returns the command if detected, None if it's regular text.
pub fn detect_command(text: &str) -> Option<BufferCommand> {
    let trimmed = text.trim().to_lowercase();

    // Scratch/undo commands — exact match only
    if matches!(
        trimmed.as_str(),
        "scratch that"
            | "scratch this"
            | "delete that"
            | "undo that"
            | "undo"
            | "scratch"
    ) {
        return Some(BufferCommand::ScratchLast);
    }

    // Clear all
    if matches!(trimmed.as_str(), "clear all" | "clear everything" | "start over") {
        return Some(BufferCommand::ClearAll);
    }

    None
}

#[derive(Debug, Clone, PartialEq)]
pub enum BufferCommand {
    /// Delete the last utterance from the buffer
    ScratchLast,
    /// Clear the entire buffer
    ClearAll,
}

/// Replace explicit text commands with their output.
fn apply_text_commands(text: &str) -> String {
    let mut result = text.to_string();

    // Line break commands (case-insensitive replacement)
    let replacements = [
        ("new paragraph", "\n\n"),
        ("New paragraph", "\n\n"),
        ("new line", "\n"),
        ("New line", "\n"),
        ("newline", "\n"),
        ("Newline", "\n"),
    ];

    for (from, to) in &replacements {
        result = result.replace(from, to);
    }

    // Explicit punctuation commands — need to handle at word boundaries.
    // Add spaces around the text first, match, then trim.
    let padded = format!(" {} ", result);
    let padded = padded
        .replace(" period ", ". ")
        .replace(" Period ", ". ")
        .replace(" comma ", ", ")
        .replace(" Comma ", ", ")
        .replace(" question mark ", "? ")
        .replace(" Question mark ", "? ")
        .replace(" exclamation point ", "! ")
        .replace(" Exclamation point ", "! ")
        .replace(" exclamation mark ", "! ")
        .replace(" Exclamation mark ", "! ");
    result = padded.trim().to_string();

    result
}

/// Remove filler words from text, preserving newlines.
fn remove_fillers(text: &str) -> String {
    // Process line by line to preserve \n boundaries
    text.split('\n')
        .map(|line| {
            let words: Vec<&str> = line.split_whitespace().collect();
            let mut result = Vec::with_capacity(words.len());
            for (i, word) in words.iter().enumerate() {
                let lower = word.to_lowercase();
                let bare = lower.trim_matches(|c: char| c.is_ascii_punctuation());
                if is_filler(bare, i, &words) {
                    continue;
                }
                result.push(*word);
            }
            result.join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Determine if a word is filler.
fn is_filler(bare_lower: &str, index: usize, words: &[&str]) -> bool {
    match bare_lower {
        "um" | "uh" | "hmm" | "hm" | "uhh" | "umm" | "erm" | "er" => true,
        // "like" as filler: only when preceded by filler-adjacent words or at sentence start
        "like" => {
            if index == 0 {
                return false; // "Like, the API..." — could be intentional
            }
            let prev = words[index - 1].to_lowercase();
            let prev_bare = prev.trim_matches(|c: char| c.is_ascii_punctuation());
            matches!(prev_bare, "um" | "uh" | "so" | "and" | "but" | "just")
        }
        // "you know" — only remove when it's the exact two-word phrase
        "you" => {
            if index + 1 < words.len() {
                let next = words[index + 1].to_lowercase();
                let next_bare = next.trim_matches(|c: char| c.is_ascii_punctuation());
                next_bare == "know"
            } else {
                false
            }
        }
        "know" => {
            if index > 0 {
                let prev = words[index - 1].to_lowercase();
                let prev_bare = prev.trim_matches(|c: char| c.is_ascii_punctuation());
                prev_bare == "you"
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Clean up whitespace — collapse multiple spaces, trim.
fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_space = false;
    let mut prev_was_newline = false;

    for ch in text.chars() {
        if ch == '\n' {
            if !prev_was_newline || !result.ends_with("\n\n") {
                result.push(ch);
            }
            prev_was_newline = true;
            prev_was_space = false;
        } else if ch.is_whitespace() {
            if !prev_was_space && !prev_was_newline {
                result.push(' ');
            }
            prev_was_space = true;
        } else {
            result.push(ch);
            prev_was_space = false;
            prev_was_newline = false;
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filler_removal() {
        // Note: T1 doesn't re-capitalize — Nemotron/TDT output is already capitalized.
        // After removing "Um" the remaining "the" stays lowercase. This is fine
        // because in real use, the model outputs "The API should uh handle rate limiting"
        // and we just strip "uh".
        assert_eq!(
            apply_t1("The API should uh handle rate limiting"),
            "The API should handle rate limiting"
        );
        assert_eq!(
            apply_t1("Um the API should handle rate limiting"),
            "the API should handle rate limiting"
        );
    }

    #[test]
    fn test_filler_removal_hmm() {
        assert_eq!(apply_t1("Hmm I think so"), "I think so");
    }

    #[test]
    fn test_filler_you_know() {
        assert_eq!(
            apply_t1("The thing is you know really important"),
            "The thing is really important"
        );
    }

    #[test]
    fn test_new_line_command() {
        assert_eq!(apply_t1("First point new line second point"), "First point\nsecond point");
    }

    #[test]
    fn test_new_paragraph_command() {
        assert_eq!(
            apply_t1("First section new paragraph second section"),
            "First section\n\nsecond section"
        );
    }

    #[test]
    fn test_punctuation_commands() {
        assert_eq!(
            apply_t1("Hello period How are you question mark"),
            "Hello. How are you?"
        );
    }

    #[test]
    fn test_scratch_command() {
        assert_eq!(detect_command("scratch that"), Some(BufferCommand::ScratchLast));
        assert_eq!(detect_command("Scratch That"), Some(BufferCommand::ScratchLast));
        assert_eq!(detect_command("undo"), Some(BufferCommand::ScratchLast));
    }

    #[test]
    fn test_not_scratch() {
        assert_eq!(detect_command("don't scratch that surface"), None);
        assert_eq!(detect_command("the API should handle rate limiting"), None);
    }

    #[test]
    fn test_clear_command() {
        assert_eq!(detect_command("clear all"), Some(BufferCommand::ClearAll));
        assert_eq!(detect_command("start over"), Some(BufferCommand::ClearAll));
    }

    #[test]
    fn test_whitespace_normalization() {
        assert_eq!(apply_t1("hello   world"), "hello world");
        assert_eq!(apply_t1("  hello  "), "hello");
    }

    #[test]
    fn test_preserves_content() {
        // Should NOT strip legitimate words
        assert_eq!(
            apply_t1("I like this approach and it works well"),
            "I like this approach and it works well"
        );
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(apply_t1(""), "");
        assert_eq!(apply_t1("um uh"), "");
    }
}
