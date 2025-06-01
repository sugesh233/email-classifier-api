import re
import hashlib
import random
import string
import spacy
from typing import Dict, Tuple, List

class PIIMasker:
    def __init__(self):
        # Load spaCy model for named entity recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if spaCy model is not available
            self.nlp = None
        
        # PII patterns
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'address': re.compile(r'\b\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|place|pl)\b', re.IGNORECASE),
            'zip_code': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
        }
        
        # Token storage for reversible masking
        self.token_mapping = {}
        self.reverse_mapping = {}
        
    def _generate_token(self, pii_type: str, original_value: str) -> str:
        """Generate a unique token for PII replacement."""
        # Create a hash of the original value for consistency
        hash_object = hashlib.md5(original_value.encode())
        hash_hex = hash_object.hexdigest()[:8]
        
        # Generate a readable token
        token = f"[{pii_type.upper()}_{hash_hex}]"
        
        return token
    
    def _detect_names_with_spacy(self, text: str) -> List[Tuple[str, int, int]]:
        """Detect person names using spaCy NER."""
        names = []
        
        if self.nlp is None:
            return names
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Filter out common false positives
                    name = ent.text.strip()
                    if len(name) > 1 and not name.lower() in ['sir', 'madam', 'mr', 'mrs', 'ms', 'dr']:
                        names.append((name, ent.start_char, ent.end_char))
                        
        except Exception:
            # Fallback if spaCy processing fails
            pass
        
        return names
    
    def _detect_names_with_regex(self, text: str) -> List[Tuple[str, int, int]]:
        """Detect potential names using regex patterns."""
        names = []
        
        # Simple pattern for detecting capitalized words that might be names
        # This is a fallback when spaCy is not available
        name_pattern = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')
        
        for match in name_pattern.finditer(text):
            name = match.group()
            # Filter out common false positives
            if not any(word.lower() in ['dear', 'hello', 'thank', 'best', 'kind'] for word in name.split()):
                names.append((name, match.start(), match.end()))
        
        return names
    
    def mask_pii(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Detect and mask PII in the given text.
        
        Args:
            text (str): Input text containing potential PII
            
        Returns:
            tuple: (masked_text, pii_mapping) where pii_mapping maps tokens to original values
        """
        if not text:
            return text, {}
        
        masked_text = text
        pii_mapping = {}
        
        # Track replacements to avoid overlapping
        replacements = []
        
        # Detect emails
        for match in self.patterns['email'].finditer(text):
            original_value = match.group()
            token = self._generate_token('EMAIL', original_value)
            replacements.append((match.start(), match.end(), token, original_value))
        
        # Detect phone numbers
        for match in self.patterns['phone'].finditer(text):
            original_value = match.group()
            token = self._generate_token('PHONE', original_value)
            replacements.append((match.start(), match.end(), token, original_value))
        
        # Detect SSNs
        for match in self.patterns['ssn'].finditer(text):
            original_value = match.group()
            token = self._generate_token('SSN', original_value)
            replacements.append((match.start(), match.end(), token, original_value))
        
        # Detect credit cards
        for match in self.patterns['credit_card'].finditer(text):
            original_value = match.group()
            # Validate credit card using Luhn algorithm
            if self._is_valid_credit_card(original_value.replace('-', '').replace(' ', '')):
                token = self._generate_token('CREDIT_CARD', original_value)
                replacements.append((match.start(), match.end(), token, original_value))
        
        # Detect addresses
        for match in self.patterns['address'].finditer(text):
            original_value = match.group()
            token = self._generate_token('ADDRESS', original_value)
            replacements.append((match.start(), match.end(), token, original_value))
        
        # Detect ZIP codes
        for match in self.patterns['zip_code'].finditer(text):
            original_value = match.group()
            token = self._generate_token('ZIP', original_value)
            replacements.append((match.start(), match.end(), token, original_value))
        
        # Detect names using spaCy or regex fallback
        if self.nlp:
            names = self._detect_names_with_spacy(text)
        else:
            names = self._detect_names_with_regex(text)
        
        for name, start, end in names:
            token = self._generate_token('NAME', name)
            replacements.append((start, end, token, name))
        
        # Sort replacements by start position (reverse order to maintain indices)
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        # Apply replacements
        for start, end, token, original_value in replacements:
            # Check for overlaps with existing replacements
            overlap = False
            for existing_start, existing_end, _, _ in replacements:
                if (existing_start, existing_end) != (start, end):
                    if not (end <= existing_start or start >= existing_end):
                        overlap = True
                        break
            
            if not overlap:
                masked_text = masked_text[:start] + token + masked_text[end:]
                pii_mapping[token] = original_value
                
                # Store in class mappings for persistence
                self.token_mapping[token] = original_value
                self.reverse_mapping[original_value] = token
        
        return masked_text, pii_mapping
    
    def unmask_pii(self, masked_text: str, pii_mapping: Dict[str, str]) -> str:
        """
        Restore original PII values in masked text.
        
        Args:
            masked_text (str): Text with PII tokens
            pii_mapping (dict): Mapping from tokens to original values
            
        Returns:
            str: Text with PII restored
        """
        if not masked_text or not pii_mapping:
            return masked_text
        
        unmasked_text = masked_text
        
        # Replace tokens with original values
        for token, original_value in pii_mapping.items():
            unmasked_text = unmasked_text.replace(token, original_value)
        
        return unmasked_text
    
    def get_pii_type(self, value: str) -> str:
        """
        Determine the type of PII based on the value.
        
        Args:
            value (str): The PII value
            
        Returns:
            str: The type of PII
        """
        if self.patterns['email'].match(value):
            return "Email Address"
        elif self.patterns['phone'].match(value):
            return "Phone Number"
        elif self.patterns['ssn'].match(value):
            return "Social Security Number"
        elif self.patterns['credit_card'].match(value.replace('-', '').replace(' ', '')):
            return "Credit Card"
        elif self.patterns['address'].match(value):
            return "Address"
        elif self.patterns['zip_code'].match(value):
            return "ZIP Code"
        else:
            return "Name"
    
    def _is_valid_credit_card(self, card_number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm.
        
        Args:
            card_number (str): Credit card number (digits only)
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not card_number.isdigit() or len(card_number) < 13 or len(card_number) > 19:
            return False
        
        # Luhn algorithm
        def luhn_checksum(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10
        
        return luhn_checksum(card_number) == 0
    
    def get_pii_statistics(self, text: str) -> Dict[str, int]:
        """
        Get statistics about PII types found in text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Statistics of PII types found
        """
        _, pii_mapping = self.mask_pii(text)
        
        stats = {
            'Email Address': 0,
            'Phone Number': 0,
            'Social Security Number': 0,
            'Credit Card': 0,
            'Address': 0,
            'ZIP Code': 0,
            'Name': 0
        }
        
        for original_value in pii_mapping.values():
            pii_type = self.get_pii_type(original_value)
            stats[pii_type] += 1
        
        return stats
    
    def clear_mappings(self):
        """Clear all stored PII mappings."""
        self.token_mapping.clear()
        self.reverse_mapping.clear()
