"""
Conversational Node — Normal chat replies.
Handles messages the router classified as "chat": greetings, small talk, simple
questions, and self-contained requests. Streams a friendly, concise answer
directly from the model — no search, no scraping, no citations.
"""

import logging

from app.services.llm import get_llm_client
from app.agents.state import ResearchState, format_history
from app.config import get_settings

logger = logging.getLogger(__name__)

CHAT_SYSTEM = """You are goon, a friendly and helpful AI assistant. You chat naturally like a normal assistant, and you can also run deep multi-source web research with cited answers when the user asks for it.

Style:
- Be warm, natural, and concise. Match the user's energy: a short greeting gets a short, friendly reply — never an essay or a "research summary".
- Answer from your own knowledge. NEVER invent citations, sources, statistics, or facts. If you're not sure, say so plainly.
- If the message would genuinely be better with current or real-world information (news, live data, prices, recent events, comparing specific real products), answer briefly and offer to research it — e.g. "I can look that up with sources if you'd like."
- If asked what you can do: you chat and answer questions, and you can do multi-source web research with cited answers on request.
- Use light Markdown only when it actually helps. No headings or citation markers in short replies."""

CHAT_PROMPT = """{conversation}{personalization}User: {query}

Reply now as goon:"""


async def conversational_node(state: ResearchState) -> dict:
    """Stream a direct conversational reply (no research)."""
    query = state["query"]
    history = state.get("history", [])
    user_name = (state.get("user_name") or "").strip()
    sse_callback = state.get("sse_callback")
    settings = get_settings()

    logger.info("Conversational: replying to: %s", query[:80])

    if sse_callback:
        await sse_callback("phase", {"phase": "writing", "message": "Replying..."})

    conversation = ""
    history_text = format_history(history, max_answer_chars=800)
    if history_text:
        conversation = f"Conversation so far:\n{history_text}\n\n"

    personalization = ""
    if user_name:
        personalization = (
            f"The user prefers to be called \"{user_name}\"; address them by it "
            "naturally when it fits, without forcing it.\n\n"
        )

    prompt = CHAT_PROMPT.format(
        conversation=conversation,
        personalization=personalization,
        query=query,
    )

    full_answer = ""
    try:
        llm = get_llm_client()
        async for token in llm.generate_stream(
            prompt=prompt,
            system=CHAT_SYSTEM,
            temperature=0.6,
            model=settings.groq_synth_model,
            max_tokens=800,
            stage="chat",
        ):
            full_answer += token
            if sse_callback:
                await sse_callback("token", {"token": token})
    except Exception as e:
        logger.error("Conversational node failed: %s", e)
        full_answer = "Sorry, I had trouble responding just now. Please try again."
        if sse_callback:
            await sse_callback("token", {"token": full_answer})

    return {
        "draft_answer": full_answer,
        "citations": [],
        "invalid_citations": 0,
        "all_sources": [],
        "follow_up_suggestions": [],
        "confidence": 1.0,
        "phase": "writing",
        "mode": "chat",
    }
