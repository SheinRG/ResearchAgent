"""
Canned answers the pipeline falls back to when it cannot produce a real one.

These live in their own dependency-free module for one reason: the eval harness
has to be able to tell a genuine answer from a graceful failure, and detecting
them by pasting copies of the strings into the scorer would mean the detector
silently stops working the first time someone rewords a message. Importing the
same constant keeps the two honest.
"""

# Synthesizer: retrieval came back with nothing usable. Answering from an empty
# context would be an invitation to hallucinate, so we say so instead.
NO_SOURCES_MESSAGE = (
    "I couldn't find reliable sources to answer this question. "
    "This can happen with very new, niche, or ambiguous topics — "
    "try rephrasing the question or making it more specific."
)

# Synthesizer: the model call itself failed after retries.
SYNTHESIS_ERROR_MESSAGE = (
    "I encountered an error while generating the answer. "
    "Please try again or rephrase your question."
)

# Conversational node: the chat reply failed.
CHAT_ERROR_MESSAGE = "Sorry, I had trouble responding just now. Please try again."

# Every fallback, for callers that need to ask "is this a real answer?"
FALLBACK_MESSAGES: tuple[str, ...] = (
    NO_SOURCES_MESSAGE,
    SYNTHESIS_ERROR_MESSAGE,
    CHAT_ERROR_MESSAGE,
)
