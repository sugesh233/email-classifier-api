import pandas as pd
import json
import io
from datetime import datetime
from typing import List, Dict, Optional

class DataManager:
    def __init__(self):
        """Initialize the data manager with session-based storage."""
        self.emails = []
        
    def add_email(self, email_data: Dict):
        """
        Add a processed email to the storage.
        
        Args:
            email_data (dict): Dictionary containing email information
        """
        # Ensure all required fields are present
        required_fields = [
            'timestamp', 'sender', 'subject', 'content', 'masked_subject', 
            'masked_content', 'category', 'confidence', 'pii_mapping', 'pii_detected'
        ]
        
        for field in required_fields:
            if field not in email_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Add unique ID
        email_data['id'] = len(self.emails) + 1
        
        # Store the email
        self.emails.append(email_data)
    
    def get_all_emails(self) -> List[Dict]:
        """
        Get all stored emails.
        
        Returns:
            list: List of email dictionaries
        """
        return self.emails.copy()
    
    def get_email_by_id(self, email_id: int) -> Optional[Dict]:
        """
        Get a specific email by ID.
        
        Args:
            email_id (int): The email ID
            
        Returns:
            dict or None: Email data if found, None otherwise
        """
        for email in self.emails:
            if email.get('id') == email_id:
                return email.copy()
        return None
    
    def get_emails_by_category(self, category: str) -> List[Dict]:
        """
        Get all emails of a specific category.
        
        Args:
            category (str): The email category
            
        Returns:
            list: List of email dictionaries
        """
        return [email for email in self.emails if email.get('category') == category]
    
    def get_emails_with_pii(self) -> List[Dict]:
        """
        Get all emails that contain PII.
        
        Returns:
            list: List of email dictionaries containing PII
        """
        return [email for email in self.emails if email.get('pii_detected', False)]
    
    def search_emails(self, query: str, fields: List[str] = None) -> List[Dict]:
        """
        Search emails by text query.
        
        Args:
            query (str): Search query
            fields (list): Fields to search in (default: ['subject', 'content', 'sender'])
            
        Returns:
            list: List of matching email dictionaries
        """
        if fields is None:
            fields = ['subject', 'content', 'sender']
        
        query = query.lower()
        results = []
        
        for email in self.emails:
            for field in fields:
                if field in email and query in str(email[field]).lower():
                    results.append(email)
                    break
        
        return results
    
    def get_category_statistics(self) -> Dict[str, int]:
        """
        Get statistics about email categories.
        
        Returns:
            dict: Category counts
        """
        stats = {}
        for email in self.emails:
            category = email.get('category', 'unknown')
            stats[category] = stats.get(category, 0) + 1
        return stats
    
    def get_pii_statistics(self) -> Dict[str, int]:
        """
        Get statistics about PII detection.
        
        Returns:
            dict: PII statistics
        """
        stats = {
            'emails_with_pii': 0,
            'emails_without_pii': 0,
            'total_pii_items': 0
        }
        
        for email in self.emails:
            if email.get('pii_detected', False):
                stats['emails_with_pii'] += 1
                stats['total_pii_items'] += len(email.get('pii_mapping', {}))
            else:
                stats['emails_without_pii'] += 1
        
        return stats
    
    def get_confidence_statistics(self) -> Dict[str, float]:
        """
        Get statistics about classification confidence.
        
        Returns:
            dict: Confidence statistics
        """
        if not self.emails:
            return {}
        
        confidences = [email.get('confidence', 0) for email in self.emails]
        
        return {
            'mean_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'low_confidence_count': len([c for c in confidences if c < 0.7]),
            'high_confidence_count': len([c for c in confidences if c >= 0.9])
        }
    
    def export_to_dataframe(self, include_pii: bool = False, 
                           include_confidence_scores: bool = True,
                           include_timestamps: bool = True) -> pd.DataFrame:
        """
        Export emails to a pandas DataFrame.
        
        Args:
            include_pii (bool): Whether to include unmasked PII data
            include_confidence_scores (bool): Whether to include detailed confidence scores
            include_timestamps (bool): Whether to include timestamps
            
        Returns:
            pd.DataFrame: DataFrame containing email data
        """
        if not self.emails:
            return pd.DataFrame()
        
        export_data = []
        
        for email in self.emails:
            row = {
                'id': email.get('id'),
                'sender': email.get('sender'),
                'category': email.get('category'),
                'confidence': email.get('confidence'),
                'pii_detected': email.get('pii_detected'),
                'pii_count': len(email.get('pii_mapping', {}))
            }
            
            if include_timestamps:
                row['timestamp'] = email.get('timestamp')
            
            if include_pii:
                row['subject'] = email.get('subject')
                row['content'] = email.get('content')
            else:
                row['subject'] = email.get('masked_subject')
                row['content'] = email.get('masked_content')
            
            if include_confidence_scores:
                confidence_scores = email.get('confidence_scores', {})
                for category, score in confidence_scores.items():
                    row[f'confidence_{category}'] = score
            
            export_data.append(row)
        
        return pd.DataFrame(export_data)
    
    def export_to_csv(self, include_pii: bool = False,
                     include_confidence_scores: bool = True,
                     include_timestamps: bool = True) -> str:
        """
        Export emails to CSV format.
        
        Args:
            include_pii (bool): Whether to include unmasked PII data
            include_confidence_scores (bool): Whether to include detailed confidence scores
            include_timestamps (bool): Whether to include timestamps
            
        Returns:
            str: CSV data as string
        """
        df = self.export_to_dataframe(
            include_pii=include_pii,
            include_confidence_scores=include_confidence_scores,
            include_timestamps=include_timestamps
        )
        
        if df.empty:
            return ""
        
        # Convert timestamp to string format for CSV
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create CSV string
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
    
    def export_to_json(self, include_pii: bool = False) -> str:
        """
        Export emails to JSON format.
        
        Args:
            include_pii (bool): Whether to include unmasked PII data
            
        Returns:
            str: JSON data as string
        """
        export_data = []
        
        for email in self.emails:
            email_copy = email.copy()
            
            # Convert datetime to string
            if 'timestamp' in email_copy:
                email_copy['timestamp'] = email_copy['timestamp'].isoformat()
            
            # Remove PII data if not requested
            if not include_pii:
                email_copy.pop('subject', None)
                email_copy.pop('content', None)
                email_copy.pop('pii_mapping', None)
            
            export_data.append(email_copy)
        
        return json.dumps(export_data, indent=2)
    
    def clear_all_data(self):
        """Clear all stored email data."""
        self.emails.clear()
    
    def get_recent_emails(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recently processed emails.
        
        Args:
            limit (int): Maximum number of emails to return
            
        Returns:
            list: List of recent email dictionaries
        """
        # Sort by timestamp (most recent first)
        sorted_emails = sorted(
            self.emails, 
            key=lambda x: x.get('timestamp', datetime.min), 
            reverse=True
        )
        
        return sorted_emails[:limit]
    
    def update_email_category(self, email_id: int, new_category: str) -> bool:
        """
        Update the category of a specific email (for feedback/correction).
        
        Args:
            email_id (int): The email ID
            new_category (str): The new category
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        for email in self.emails:
            if email.get('id') == email_id:
                email['category'] = new_category
                email['manually_corrected'] = True
                email['correction_timestamp'] = datetime.now()
                return True
        return False
    
    def get_feedback_data(self) -> List[Dict]:
        """
        Get emails that have been manually corrected for retraining.
        
        Returns:
            list: List of corrected email data suitable for classifier retraining
        """
        feedback_data = []
        
        for email in self.emails:
            if email.get('manually_corrected', False):
                feedback_data.append({
                    'text': f"{email.get('subject', '')} {email.get('content', '')}",
                    'category': email.get('category')
                })
        
        return feedback_data
