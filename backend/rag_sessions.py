from langchain_community.chat_message_histories import ChatMessageHistory

_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Returns the in-memory chat history object for one session, creating it if needed."""

    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def clear_session_history(session_id: str) -> None:
    """Deletes all stored messages for one session."""

    if session_id in _session_store:
        _session_store[session_id].clear()


def get_history_messages(session_id: str) -> list[dict[str, str]]:
    """Converts LangChain history objects into frontend-friendly role/content dictionaries."""

    history = get_session_history(session_id)
    messages: list[dict[str, str]] = []
    for item in history.messages:
        role = "user" if item.__class__.__name__ == "HumanMessage" else "assistant"
        messages.append({"role": role, "content": item.content})
    return messages


def get_session_metrics() -> dict[str, int]:
    """Returns aggregate session and message counts for dashboard statistics."""

    return {
        "sessions": len(_session_store),
        "messages": sum(len(history.messages) for history in _session_store.values()),
    }
