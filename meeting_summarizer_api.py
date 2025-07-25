"""
Meeting Summarization API
-------------------------

This module implements a lightweight summarization service tailored for meeting
transcripts and long-form text.  The goal of this service is to extract the
key points of a discussion without requiring any heavy machine‑learning
infrastructure.  It relies on a simple frequency‑based extractive approach
combined with heuristic action item detection.

The service exposes a single endpoint, `/summarize`, via FastAPI.  Clients
provide raw text and optional parameters controlling the summary length.  The
response contains a concise summary and any detected action items.  Action
items are sentences that either begin with an imperative verb or contain
keywords indicating required tasks (e.g. “should”, “must”, “needs”).

This API is designed to be inexpensive to run (no external API calls or model
weights) and easy to host on free tiers of popular platforms.  It can be
extended or swapped out for more advanced summarization engines as the
business scales.

Usage:
    uvicorn meeting_summarizer_api:app --reload --port 8000

"""

from __future__ import annotations

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
import re

app = FastAPI(
    title="Meeting Summarization API",
    description=(
        "A lightweight API that condenses long meeting transcripts into concise"
        " summaries and extracts actionable items."
    ),
    version="0.1.0",
)

# A simple English stopword list.  These words carry little semantic weight and
# are excluded from frequency calculations.  Feel free to extend this list
# based on your use case.
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who",
    "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
}

# Imperative verbs used to detect action items.  Sentences starting with these
# verbs likely describe tasks or next steps.
IMPERATIVE_VERBS = {
    "call", "check", "complete", "deliver", "discuss", "do", "draft", "email",
    "finalize", "finish", "follow", "implement", "inform", "meet", "prepare",
    "present", "provide", "review", "schedule", "send", "share", "start",
    "submit", "update", "write",
}

# Additional keywords that often signal tasks or obligations.
ACTION_KEYWORDS = {
    "should", "must", "need", "needs", "required", "to‑do", "todo", "action",
    "deadline", "assign", "due", "responsible",
}


class SummarizeRequest(BaseModel):
    """Request model for summarization.

    Attributes:
        text: The input document or meeting transcript to be summarized.
        ratio: Fraction of sentences to include in the summary (0 < ratio <= 1).
        max_sentences: Maximum number of sentences in the summary.
    """
    text: str = Field(..., description="The full text of the transcript to summarise.")
    ratio: float = Field(
        0.3,
        ge=0.05,
        le=1.0,
        description="Fraction of sentences to include in the summary."
    )
    max_sentences: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of sentences to include in the summary."
    )


class SummarizeResponse(BaseModel):
    """Response model for summarization."""
    summary: str = Field(..., description="Concise extractive summary of the input.")
    action_items: List[str] = Field(
        [], description="List of sentences identified as action items."
    )


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation boundaries.

    The regex looks for sentence terminators (period, exclamation point, or
    question mark) followed by whitespace.  This is a simple heuristic and
    does not handle all edge cases (e.g. abbreviations) but works well for
    meeting transcripts.
    """
    # Replace newlines with spaces to avoid splitting incorrectly
    cleaned = re.sub(r"\s+", " ", text.strip())
    # Split on punctuation followed by a space. Include the punctuation in the
    # resulting sentence by using a positive lookbehind.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    # Remove empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(sentence: str) -> List[str]:
    """Extract lowercase word tokens from a sentence."""
    return re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", sentence.lower())


def compute_word_frequencies(sentences: List[str]) -> dict[str, float]:
    """Compute normalized frequency of non‑stopwords in the provided sentences."""
    freq: dict[str, int] = {}
    for sent in sentences:
        for word in tokenize_words(sent):
            if word not in STOPWORDS:
                freq[word] = freq.get(word, 0) + 1
    # Normalize frequencies to prevent bias towards longer transcripts
    max_freq = max(freq.values()) if freq else 1
    return {word: count / max_freq for word, count in freq.items()}


def score_sentences(sentences: List[str], freq: dict[str, float]) -> List[tuple[int, float]]:
    """Assign a score to each sentence based on the average frequency of its words."""
    scores: List[tuple[int, float]] = []
    for idx, sent in enumerate(sentences):
        words = tokenize_words(sent)
        if not words:
            continue
        sentence_score = sum(freq.get(word, 0.0) for word in words)
        normalized_score = sentence_score / len(words)
        scores.append((idx, normalized_score))
    return scores


def summarize(text: str, ratio: float, max_sentences: int) -> str:
    """Generate an extractive summary for the given text.

    The algorithm selects the top sentences based on frequency scores.
    """
    sentences = tokenize_sentences(text)
    if not sentences:
        return ""
    freq = compute_word_frequencies(sentences)
    scores = score_sentences(sentences, freq)
    # Determine how many sentences to keep
    target_count = max(1, min(max_sentences, int(len(sentences) * ratio)))
    # Pick the highest scored sentences
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:target_count]
    # Sort by original order to preserve narrative flow
    selected_indices = sorted(idx for idx, _score in top)
    return " ".join(sentences[i] for i in selected_indices)


def extract_action_items(sentences: List[str]) -> List[str]:
    """Identify sentences that represent action items.

    A sentence is considered an action item if it starts with an imperative verb
    or contains certain keywords indicating responsibility or obligation.
    """
    action_items: List[str] = []
    for sent in sentences:
        words = tokenize_words(sent)
        if not words:
            continue
        # Check for imperative verb at the start
        first_word = words[0]
        if first_word in IMPERATIVE_VERBS:
            action_items.append(sent)
            continue
        # Check for presence of any action keyword
        if any(word in ACTION_KEYWORDS for word in words):
            action_items.append(sent)
    return action_items


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(req: SummarizeRequest) -> SummarizeResponse:
    """API endpoint that returns a summary and action items for a transcript."""
    sentences = tokenize_sentences(req.text)
    summary_text = summarize(req.text, req.ratio, req.max_sentences)
    actions = extract_action_items(sentences)
    return SummarizeResponse(summary=summary_text, action_items=actions)