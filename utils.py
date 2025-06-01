import re
import hashlib
from typing import Dict, List, Tuple
from datetime import datetime

def validate_email_format(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if valid email format, False otherwise
    """
    if not email:
        return False
    
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_pattern.match(email))

def get_confidence_color(confidence: float) -> str:
    """
    Get color code based on confidence level.
    
    Args:
        confidence (float): Confidence score (0-1)
        
    Returns:
        str: Color name or hex code
    """
    if confidence >= 0.9:
        return "green"
    elif confidence >= 0.7:
        return "orange"
    else:
        return "red"

def format_timestamp(timestamp: datetime) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp (datetime): Timestamp to format
        
    Returns:
        str: Formatted timestamp string
    """
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized

def generate_secure_hash(data: str, salt: str = "") -> str:
    """
    Generate a secure hash of the given data.
    
    Args:
        data (str): Data to hash
        salt (str): Optional salt for additional security
        
    Returns:
        str: Hexadecimal hash string
    """
    combined_data = f"{data}{salt}"
    return hashlib.sha256(combined_data.encode()).hexdigest()

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text (simple implementation).
    
    Args:
        text (str): Input text
        min_length (int): Minimum word length to consider
        
    Returns:
        list: List of keywords
    """
    if not text:
        return []
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter by length and remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    keywords = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words
    ]
    
    return list(set(keywords))  # Remove duplicates

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using word overlap.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(extract_keywords(text1))
    words2 = set(extract_keywords(text2))
    
    if not words1 and not words2:
        return 1.0
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def format_confidence_percentage(confidence: float) -> str:
    """
    Format confidence as percentage string.
    
    Args:
        confidence (float): Confidence score (0-1)
        
    Returns:
        str: Formatted percentage string
    """
    return f"{confidence:.1%}"

def parse_phone_number(phone: str) -> Dict[str, str]:
    """
    Parse phone number into components.
    
    Args:
        phone (str): Phone number string
        
    Returns:
        dict: Dictionary with phone number components
    """
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) == 10:
        return {
            'area_code': digits[:3],
            'exchange': digits[3:6],
            'number': digits[6:],
            'formatted': f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        }
    elif len(digits) == 11 and digits[0] == '1':
        return {
            'country_code': digits[0],
            'area_code': digits[1:4],
            'exchange': digits[4:7],
            'number': digits[7:],
            'formatted': f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        }
    else:
        return {
            'raw': phone,
            'formatted': phone
        }

def validate_confidence_score(score) -> bool:
    """
    Validate that confidence score is a valid number between 0 and 1.
    
    Args:
        score: Score to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        float_score = float(score)
        return 0.0 <= float_score <= 1.0
    except (ValueError, TypeError):
        return False

def get_email_domain(email: str) -> str:
    """
    Extract domain from email address.
    
    Args:
        email (str): Email address
        
    Returns:
        str: Domain name or empty string if invalid
    """
    if not validate_email_format(email):
        return ""
    
    return email.split('@')[1].lower()

def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Word count
    """
    if not text:
        return 0
    
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time for text.
    
    Args:
        text (str): Input text
        words_per_minute (int): Reading speed
        
    Returns:
        int: Estimated reading time in minutes
    """
    word_count = count_words(text)
    if word_count == 0:
        return 0
    
    minutes = max(1, round(word_count / words_per_minute))
    return minutes

def create_summary_stats(emails: List[Dict]) -> Dict:
    """
    Create summary statistics for a list of emails.
    
    Args:
        emails (list): List of email dictionaries
        
    Returns:
        dict: Summary statistics
    """
    if not emails:
        return {}
    
    total_emails = len(emails)
    categories = [email.get('category', 'unknown') for email in emails]
    confidences = [email.get('confidence', 0) for email in emails]
    pii_emails = sum(1 for email in emails if email.get('pii_detected', False))
    
    return {
        'total_emails': total_emails,
        'unique_categories': len(set(categories)),
        'most_common_category': max(set(categories), key=categories.count) if categories else 'N/A',
        'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
        'pii_percentage': (pii_emails / total_emails * 100) if total_emails > 0 else 0,
        'low_confidence_emails': sum(1 for c in confidences if c < 0.7),
        'high_confidence_emails': sum(1 for c in confidences if c >= 0.9)
    }
