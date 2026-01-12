import csv
import re
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple, Optional
import uuid

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
        
        Supports multiple formats:
        - Chatbase custom format (with separators)
        - Standard CSV with columns like: conversation_id, role, content, timestamp
        - Simple CSV with: role, content (auto-generates conversation IDs)
        """
        # First, try to detect the format
        format_type = self._detect_format()
        
        if format_type == "standard_csv":
            return self._parse_standard_csv()
        elif format_type == "simple_csv":
            return self._parse_simple_csv()
        else:
            return self._parse_chatbase_format()
    
    def _detect_format(self) -> str:
        """Detects the CSV format by reading the first few lines."""
        try:
            df = pd.read_csv(self.filepath, nrows=5, encoding='utf-8', on_bad_lines='skip')
            columns_lower = [str(c).lower().strip() for c in df.columns]
            
            # Check for standard CSV format with known columns
            if 'role' in columns_lower and 'content' in columns_lower:
                if 'conversation_id' in columns_lower or 'chatid' in columns_lower or 'chat_id' in columns_lower:
                    return "standard_csv"
                else:
                    return "simple_csv"
            
            # Check for message/response format
            if 'message' in columns_lower or 'user_message' in columns_lower:
                return "standard_csv"
                
        except Exception:
            pass
        
        # Default to chatbase format
        return "chatbase"
    
    def _parse_standard_csv(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse standard CSV with columns like conversation_id, role, content."""
        df = pd.read_csv(self.filepath, encoding='utf-8', on_bad_lines='skip')
        
        # Normalize column names
        original_columns = list(df.columns)
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Map common column name variations (extended list)
        column_mappings = {
            'conversation_id': [
                'conversation_id', 'chatid', 'chat_id', 'session_id', 'thread_id', 'id',
                'conv_id', 'conversation', 'chat', 'session', 'thread', 'uuid',
                'customer_id', 'user_id', 'ticket_id'
            ],
            'role': [
                'role', 'sender', 'type', 'from', 'author', 'actor', 'speaker',
                'participant', 'source', 'direction', 'who', 'party'
            ],
            'content': [
                'content', 'message', 'text', 'body', 'user_message', 'response',
                'msg', 'chat_message', 'query', 'answer', 'question', 'reply',
                'input', 'output', 'prompt', 'completion', 'utterance',
                'customer_message', 'agent_message', 'bot_response', 'user_input',
                'assistant_response', 'human', 'ai', 'request', 'transcript'
            ],
            'timestamp': [
                'timestamp', 'created_at', 'date', 'time', 'datetime', 'created',
                'sent_at', 'received_at', 'updated_at', 'when', 'ts', 'sent_time'
            ]
        }
        
        # Find actual column names
        actual_columns = {}
        for target, variations in column_mappings.items():
            for var in variations:
                if var in df.columns:
                    actual_columns[target] = var
                    break
        
        # Handle special case: separate user_message and response columns
        if 'user_message' in df.columns and 'response' in df.columns:
            return self._parse_qa_format(df, actual_columns)
        
        # Handle special case: separate question and answer columns
        if 'question' in df.columns and 'answer' in df.columns:
            df['user_message'] = df['question']
            df['response'] = df['answer']
            return self._parse_qa_format(df, actual_columns)
        
        # Handle special case: separate input and output columns
        if 'input' in df.columns and 'output' in df.columns:
            df['user_message'] = df['input']
            df['response'] = df['output']
            return self._parse_qa_format(df, actual_columns)
        
        # Handle special case: separate prompt and completion columns
        if 'prompt' in df.columns and 'completion' in df.columns:
            df['user_message'] = df['prompt']
            df['response'] = df['completion']
            return self._parse_qa_format(df, actual_columns)
        
        # If no content column found, try to use the first text-like column
        if 'content' not in actual_columns:
            # Find first column with string data that looks like messages
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if it contains text (average length > 10 chars)
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > 10:
                        actual_columns['content'] = col
                        break
        
        # Still no content column? Show helpful error
        if 'content' not in actual_columns:
            raise ValueError(
                f"CSV muss eine Nachrichtenspalte haben.\n"
                f"Gefundene Spalten: {original_columns}\n"
                f"Erwartete Spalten (eine davon): content, message, text, body, query, answer, etc."
            )
        
        # Generate conversation_id if not present
        if 'conversation_id' not in actual_columns:
            df['conversation_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
            actual_columns['conversation_id'] = 'conversation_id'
        
        # Generate role if not present (assume alternating user/assistant)
        if 'role' not in actual_columns:
            df['role'] = ['user' if i % 2 == 0 else 'assistant' for i in range(len(df))]
            actual_columns['role'] = 'role'
        
        # Normalize role values
        role_col = actual_columns['role']
        df[role_col] = df[role_col].astype(str).str.lower().str.strip()
        df[role_col] = df[role_col].replace({
            'human': 'user', 'customer': 'user', 'client': 'user',
            'bot': 'assistant', 'ai': 'assistant', 'agent': 'assistant', 'system': 'assistant'
        })
        
        # Parse timestamp if available
        if 'timestamp' in actual_columns:
            df['parsed_date'] = pd.to_datetime(df[actual_columns['timestamp']], errors='coerce')
        else:
            df['parsed_date'] = pd.Timestamp.now()
        
        # Build messages DataFrame
        msg_df = pd.DataFrame({
            'conversation_id': df[actual_columns['conversation_id']],
            'role': df[role_col],
            'content': df[actual_columns['content']].astype(str)
        })
        
        # Build conversations DataFrame
        conv_groups = df.groupby(actual_columns['conversation_id'])
        conv_data = []
        
        for conv_id, group in conv_groups:
            first_date = group['parsed_date'].min()
            last_date = group['parsed_date'].max()
            duration = (last_date - first_date).total_seconds() if pd.notna(first_date) and pd.notna(last_date) else 0
            
            conv_data.append({
                'conversation_id': conv_id,
                'date': first_date if pd.notna(first_date) else pd.Timestamp.now(),
                'last_message_at': last_date if pd.notna(last_date) else pd.Timestamp.now(),
                'duration_seconds': max(0, duration),
                'source': 'csv_import',
                'message_count': len(group)
            })
        
        conv_df = pd.DataFrame(conv_data)
        
        return conv_df, msg_df
    
    def _parse_qa_format(self, df: pd.DataFrame, actual_columns: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse CSV with separate user_message and response columns."""
        messages = []
        conversations = []
        
        for idx, row in df.iterrows():
            conv_id = str(row.get(actual_columns.get('conversation_id', 'id'), str(uuid.uuid4())))
            
            # Parse timestamp
            timestamp = None
            if 'timestamp' in actual_columns:
                try:
                    timestamp = pd.to_datetime(row[actual_columns['timestamp']])
                except:
                    timestamp = pd.Timestamp.now()
            else:
                timestamp = pd.Timestamp.now()
            
            # Add user message
            if pd.notna(row.get('user_message', '')):
                messages.append({
                    'conversation_id': conv_id,
                    'role': 'user',
                    'content': str(row['user_message'])
                })
            
            # Add assistant response
            if pd.notna(row.get('response', '')):
                messages.append({
                    'conversation_id': conv_id,
                    'role': 'assistant',
                    'content': str(row['response'])
                })
            
            conversations.append({
                'conversation_id': conv_id,
                'date': timestamp,
                'last_message_at': timestamp,
                'duration_seconds': 0,
                'source': 'csv_import',
                'message_count': 2
            })
        
        return pd.DataFrame(conversations), pd.DataFrame(messages)
    
    def _parse_simple_csv(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse simple CSV with just role and content columns."""
        df = pd.read_csv(self.filepath, encoding='utf-8', on_bad_lines='skip')
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Generate a single conversation ID for all messages
        conv_id = str(uuid.uuid4())
        
        # Find content column
        content_col = None
        for col in ['content', 'message', 'text', 'body']:
            if col in df.columns:
                content_col = col
                break
        
        if not content_col:
            raise ValueError("CSV muss eine 'content' oder 'message' Spalte haben")
        
        # Find role column
        role_col = None
        for col in ['role', 'sender', 'type', 'from']:
            if col in df.columns:
                role_col = col
                break
        
        messages = []
        for idx, row in df.iterrows():
            role = 'user' if idx % 2 == 0 else 'assistant'
            if role_col:
                role = str(row[role_col]).lower().strip()
                if role in ['human', 'customer', 'client']:
                    role = 'user'
                elif role in ['bot', 'ai', 'agent', 'system']:
                    role = 'assistant'
            
            messages.append({
                'conversation_id': conv_id,
                'role': role,
                'content': str(row[content_col])
            })
        
        conv_df = pd.DataFrame([{
            'conversation_id': conv_id,
            'date': pd.Timestamp.now(),
            'last_message_at': pd.Timestamp.now(),
            'duration_seconds': 0,
            'source': 'csv_import',
            'message_count': len(messages)
        }])
        
        return conv_df, pd.DataFrame(messages)
    
    def _parse_chatbase_format(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse the original Chatbase export format."""
        reading_messages = False
        
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
                if len(row) >= 3 and row[1] == 'Date Created' and row[2] == 'Last Message At':
                    continue
                
                if len(row) >= 3 and row[0] == 'Conversation:' and row[1] == 'Conversation ID':
                    continue
                
                if len(row) >= 3 and row[0] == 'Messages:' and row[1] == 'Role':
                    reading_messages = True
                    continue
                
                # Extract Dates
                if not date_created and len(row) >= 3 and self._is_date(row[1]):
                    date_created = self._parse_date(row[1])
                    if len(row) > 2:
                        last_message_at = self._parse_date(row[2])
                    continue
                    
                # Extract Conversation ID
                if not conv_id and len(row) >= 3 and self._is_uuid(row[1]):
                    conv_id = row[1]
                    if len(row) > 2:
                        source = row[2]
                    continue

                # Extract Messages
                if reading_messages:
                    if len(row) >= 3:
                        role = row[1]
                        content = row[2]
                        if role and content:
                             current_messages.append({
                                 'role': role,
                                 'content': content
                             })

            # Save last conversation
            if conv_id and (date_created or current_messages):
                self._save_conversation(conv_id, date_created, last_message_at, source, current_messages)

        # If no conversations found, try standard CSV parsing as fallback
        if not self.conversations:
            return self._parse_standard_csv()

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
