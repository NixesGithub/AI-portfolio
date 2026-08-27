from app.memory.base import (
    ConversationMeta,
    ConversationStore,
    StoredMessage,
    new_id,
    utcnow,
)
from app.memory.in_memory import InMemoryConversationStore
from app.memory.locks import ConversationLocks
from app.memory.sqlite import SQLiteConversationStore

__all__ = [
    "ConversationLocks",
    "ConversationMeta",
    "ConversationStore",
    "InMemoryConversationStore",
    "SQLiteConversationStore",
    "StoredMessage",
    "new_id",
    "utcnow",
]


def build_store(settings) -> ConversationStore:
    """Fábrica del almacén según configuración."""
    if settings.store_backend == "sqlite":
        return SQLiteConversationStore(
            settings.sqlite_path, max_messages=settings.max_stored_messages
        )
    return InMemoryConversationStore(max_messages=settings.max_stored_messages)
