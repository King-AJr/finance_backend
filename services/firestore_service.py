from langchain_google_firestore import FirestoreChatMessageHistory
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional
from google.cloud import firestore
from datetime import datetime

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
COLLECTION_NAME = "chat_history"
client = firestore.Client(project=PROJECT_ID)

class FirestoreService:
    def __init__(self):
        self.project_id = PROJECT_ID
        self.collection_name = COLLECTION_NAME

    def _get_chat_history(self, user_id: str) -> FirestoreChatMessageHistory:
        """Get chat history for a specific user."""
        return FirestoreChatMessageHistory(
            session_id=user_id,
            client=client,
            collection=self.collection_name
        )

    def store_conversation(self, user_id: str, message: str, response: str) -> None:
        """Store a conversation in Firestore using langchain_google_firestore."""
        chat_history = self._get_chat_history(user_id)
        
        # Add the messages to the chat history
        chat_history.add_user_message(message)
        chat_history.add_ai_message(response)

    def get_user_conversations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve recent conversations for a user."""
        chat_history = self._get_chat_history(user_id)
        
        # Get messages from chat history
        messages = chat_history.messages[-limit:]  # Get last N messages
        
        # Convert to simplified format
        conversations = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                conversations.append({
                    'message': messages[i].content,
                    'response': messages[i + 1].content,
                    'timestamp': datetime.now()
                })
        
        return conversations

        