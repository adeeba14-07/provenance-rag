import json
import os
from datetime import datetime

CONVERSATIONS_FILE = "conversations.json"

def load_conversations():
    """Load all conversations from JSON."""
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_conversations(conversations):
    """Save all conversations to JSON."""
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(conversations, f, indent=4)

def get_conversation(document_name, chat_id=None):
    """Get a specific conversation for a document."""
    conversations = load_conversations()
    doc_conversations = conversations.get(document_name, {})
    if chat_id:
        return doc_conversations.get(chat_id, {"messages": []})
    return doc_conversations

def create_new_chat(document_name):
    """Create a new chat for a document."""
    conversations = load_conversations()
    if document_name not in conversations:
        conversations[document_name] = {}

    chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    conversations[document_name][chat_id] = {
        "chat_id": chat_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }
    save_conversations(conversations)
    return chat_id

def save_message(document_name, chat_id, role, content):
    """Save a message to a conversation."""
    conversations = load_conversations()
    if document_name not in conversations:
        conversations[document_name] = {}
    if chat_id not in conversations[document_name]:
        conversations[document_name][chat_id] = {
            "chat_id": chat_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": []
        }

    conversations[document_name][chat_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_conversations(conversations)

def delete_chat(document_name, chat_id):
    """Delete a specific chat."""
    conversations = load_conversations()
    if document_name in conversations and chat_id in conversations[document_name]:
        del conversations[document_name][chat_id]
    save_conversations(conversations)

def get_all_chats_for_document(document_name):
    """Get all chat IDs for a document."""
    conversations = load_conversations()
    doc_conversations = conversations.get(document_name, {})
    return list(doc_conversations.keys())