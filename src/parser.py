import csv
import re
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple, Optional

class ChatLogParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.conversations = []
        self.messages = []
        
    def parse(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parses the chat log CSV file and returns two DataFrames:
        1. conversations: Metadata for each conversation
        2. messages: Individual messages linked to conversations
        """
        current_conv = {}
        reading_messages = False
        
        # We need to detect the start of a new block. 
        # Based on file structure, blocks are separated by underscore lines.
        # But csv.reader will handle multiline fields correctly, so we can iterate rows.
        
        with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            
            # State variables
            conv_id = None
            date_created = None
            last_message_at = None
            source = None
            
            # Temporary storage for current conversation messages
            current_messages = []
            
            for i, row in enumerate(reader):
                if not row:
                    continue
                
                # Check for separator line (list of underscores)
                is_separator = False
                if row and len(row) > 0 and row[0].startswith('____'):
                    is_separator = True
                
                if is_separator:
                    # Save previous conversation if it exists
                    if conv_id and (date_created or current_messages):
                        self._save_conversation(conv_id, date_created, last_message_at, source, current_messages)
                    
                    # Reset state
                    conv_id = None
                    date_created = None
                    last_message_at = None
                    source = None
                    current_messages = []
                    reading_messages = False
                    continue
                
                # Metadata detection
                # Row structure for Dates: [, "Date Created", "Last Message At"]
                if len(row) >= 3 and row[1] == 'Date Created' and row[2] == 'Last Message At':
                    continue # Header row
                
                # Value row for Dates (check if previous row was header or by position? 
                # Better to regex or check format, but row structure is consistent)
                # It usually follows the header.
                
                # Check for Conversation ID header
                if len(row) >= 3 and row[0] == 'Conversation:' and row[1] == 'Conversation ID':
                    continue
                
                # Check for Messages header
                if len(row) >= 3 and row[0] == 'Messages:' and row[1] == 'Role':
                    reading_messages = True
                    continue
                
                # Logic to extract values based on context or patterns
                
                # Extract Dates: 2nd col is date-like
                if not date_created and len(row) >= 3 and self._is_date(row[1]):
                    date_created = self._parse_date(row[1])
                    if len(row) > 2:
                        last_message_at = self._parse_date(row[2])
                    continue
                    
                # Extract Conversation ID: 2nd col is UUID-like
                if not conv_id and len(row) >= 3 and self._is_uuid(row[1]):
                    conv_id = row[1]
                    if len(row) > 2:
                        source = row[2]
                    continue

                # Extract Messages
                if reading_messages:
                    # Message rows: [, role, message]
                    if len(row) >= 3:
                        role = row[1]
                        content = row[2]
                        if role and content: # Ensure valid message row
                             current_messages.append({
                                 'role': role,
                                 'content': content
                             })

            # Save last conversation
            if conv_id and (date_created or current_messages):
                self._save_conversation(conv_id, date_created, last_message_at, source, current_messages)

        return pd.DataFrame(self.conversations), pd.DataFrame(self.messages)

    def _save_conversation(self, conv_id, date_created, last_message_at, source, messages):
        # Calculate duration
        duration = 0
        if date_created and last_message_at:
            delta = last_message_at - date_created
            duration = delta.total_seconds()
            
        self.conversations.append({
            'conversation_id': conv_id,
            'date': date_created,
            'last_message_at': last_message_at,
            'duration_seconds': duration,
            'source': source,
            'message_count': len(messages)
        })
        
        for msg in messages:
            self.messages.append({
                'conversation_id': conv_id,
                'role': msg['role'],
                'content': msg['content']
            })

    def _is_date(self, text: str) -> bool:
        # Simple check: starts with 202X
        return text and text.strip().startswith('202') and ':' in text

    def _parse_date(self, text: str) -> Optional[datetime]:
        try:
            # Format: 2026-1-8 18:42:21
            return datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Try single digit month/day handling if needed (strptime handles it mostly? No, %m expects 01-12)
            # The example showed 2026-1-8, so single digits are possible.
            # Python's strptime %m matches 1 or 01 on some platforms but strictly per spec it might fail.
            # Let's try flexible parsing.
            try:
                # Fallback for "2026-1-8"
                parts = text.strip().split(' ')
                date_parts = parts[0].split('-')
                time_parts = parts[1].split(':')
                return datetime(
                    int(date_parts[0]), int(date_parts[1]), int(date_parts[2]),
                    int(time_parts[0]), int(time_parts[1]), int(time_parts[2])
                )
            except:
                return None

    def _is_uuid(self, text: str) -> bool:
        # Check for UUID format (8-4-4-4-12 hex)
        # Or just check length and dashes if we want to be loose
        return text and len(text) > 20 and '-' in text and ' ' not in text
