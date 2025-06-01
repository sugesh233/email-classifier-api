import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string

class EmailClassifier:
    def __init__(self):
        self.categories = [
            'technical',
            'billing', 
            'general_inquiry',
            'complaint',
            'feature_request'
        ]
        
        self.pipeline = None
        self.is_trained = False
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize and train the classifier with synthetic training data."""
        # Create synthetic training data for initial model training
        training_data = self._generate_training_data()
        
        # Create the classification pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents='ascii'
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Prepare training data
        X = [self._preprocess_text(item['text']) for item in training_data]
        y = [item['category'] for item in training_data]
        
        # Train the model
        self.pipeline.fit(X, y)
        self.is_trained = True
    
    def _generate_training_data(self):
        """Generate synthetic training data for different email categories."""
        training_data = [
            # Technical support emails
            {'text': 'having trouble logging into my account password reset not working', 'category': 'technical'},
            {'text': 'application crashes when I try to upload files error message appears', 'category': 'technical'},
            {'text': 'website is loading very slowly timeout errors connection issues', 'category': 'technical'},
            {'text': 'mobile app not syncing data showing old information', 'category': 'technical'},
            {'text': 'cannot connect to database server error 500 internal error', 'category': 'technical'},
            {'text': 'installation failed missing dependencies setup problems', 'category': 'technical'},
            {'text': 'API returning wrong response format documentation unclear', 'category': 'technical'},
            {'text': 'backup restore process failing data corruption detected', 'category': 'technical'},
            {'text': 'SSL certificate expired security warning browser', 'category': 'technical'},
            {'text': 'integration with third party service not working authentication', 'category': 'technical'},
            
            # Billing inquiries
            {'text': 'question about my invoice charges seem incorrect billing period', 'category': 'billing'},
            {'text': 'need to update payment method credit card expired', 'category': 'billing'},
            {'text': 'when will I be charged for next billing cycle subscription', 'category': 'billing'},
            {'text': 'requesting refund for unused service credits', 'category': 'billing'},
            {'text': 'upgrade plan pricing options cost comparison', 'category': 'billing'},
            {'text': 'billing address change update account information', 'category': 'billing'},
            {'text': 'duplicate charges on credit card statement', 'category': 'billing'},
            {'text': 'tax invoice required for business accounting purposes', 'category': 'billing'},
            {'text': 'subscription cancellation how to stop recurring payments', 'category': 'billing'},
            {'text': 'discount coupon code not working promotional offer', 'category': 'billing'},
            
            # General inquiries
            {'text': 'how do I use this feature getting started guide', 'category': 'general_inquiry'},
            {'text': 'information about your service plans available options', 'category': 'general_inquiry'},
            {'text': 'operating hours customer support availability', 'category': 'general_inquiry'},
            {'text': 'product documentation user manual download', 'category': 'general_inquiry'},
            {'text': 'company information contact details office locations', 'category': 'general_inquiry'},
            {'text': 'privacy policy data handling security measures', 'category': 'general_inquiry'},
            {'text': 'terms of service agreement understanding', 'category': 'general_inquiry'},
            {'text': 'partnership opportunities business collaboration', 'category': 'general_inquiry'},
            {'text': 'training materials educational resources available', 'category': 'general_inquiry'},
            {'text': 'system requirements compatibility information', 'category': 'general_inquiry'},
            
            # Complaints
            {'text': 'very disappointed with service quality poor experience', 'category': 'complaint'},
            {'text': 'customer service representative was unhelpful rude', 'category': 'complaint'},
            {'text': 'system went down during important presentation lost work', 'category': 'complaint'},
            {'text': 'waited too long for response slow support team', 'category': 'complaint'},
            {'text': 'promised features not delivered false advertising', 'category': 'complaint'},
            {'text': 'data lost due to system failure poor backup', 'category': 'complaint'},
            {'text': 'unexpected service interruption no advance notice', 'category': 'complaint'},
            {'text': 'billing errors consistently wrong charges', 'category': 'complaint'},
            {'text': 'interface confusing difficult to navigate poor design', 'category': 'complaint'},
            {'text': 'security breach concerned about data protection', 'category': 'complaint'},
            
            # Feature requests
            {'text': 'would like dark mode theme option user interface', 'category': 'feature_request'},
            {'text': 'request integration with popular calendar applications', 'category': 'feature_request'},
            {'text': 'need bulk export functionality data management', 'category': 'feature_request'},
            {'text': 'mobile notifications push alerts important updates', 'category': 'feature_request'},
            {'text': 'advanced search filters better data discovery', 'category': 'feature_request'},
            {'text': 'collaboration tools team sharing workspace', 'category': 'feature_request'},
            {'text': 'automated reporting scheduled email reports', 'category': 'feature_request'},
            {'text': 'custom dashboard widgets personalized interface', 'category': 'feature_request'},
            {'text': 'real time chat support instant messaging', 'category': 'feature_request'},
            {'text': 'API rate limiting controls developer tools', 'category': 'feature_request'},
        ]
        
        return training_data
    
    def _preprocess_text(self, text):
        """Preprocess text for classification."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def classify_email(self, subject, content):
        """
        Classify an email into one of the predefined categories.
        
        Args:
            subject (str): Email subject line
            content (str): Email content
            
        Returns:
            dict: Classification result with category, confidence, and all confidence scores
        """
        if not self.is_trained:
            raise Exception("Classifier is not trained")
        
        # Combine subject and content for classification
        combined_text = f"{subject} {content}"
        processed_text = self._preprocess_text(combined_text)
        
        if not processed_text:
            return {
                'category': 'general_inquiry',
                'confidence': 0.5,
                'confidence_scores': {cat: 0.2 for cat in self.categories}
            }
        
        # Get prediction probabilities
        probabilities = self.pipeline.predict_proba([processed_text])[0]
        
        # Get predicted category
        predicted_category = self.pipeline.predict([processed_text])[0]
        
        # Get confidence score
        max_confidence = max(probabilities)
        
        # Create confidence scores dictionary
        confidence_scores = {}
        for i, category in enumerate(self.pipeline.classes_):
            confidence_scores[category] = probabilities[i]
        
        return {
            'category': predicted_category,
            'confidence': max_confidence,
            'confidence_scores': confidence_scores
        }
    
    def retrain_with_feedback(self, feedback_data):
        """
        Retrain the classifier with user feedback data.
        
        Args:
            feedback_data (list): List of dictionaries with 'text' and 'category' keys
        """
        if not feedback_data:
            return
        
        # Combine original training data with feedback
        original_training = self._generate_training_data()
        combined_data = original_training + feedback_data
        
        # Prepare training data
        X = [self._preprocess_text(item['text']) for item in combined_data]
        y = [item['category'] for item in combined_data]
        
        # Retrain the model
        self.pipeline.fit(X, y)
        
    def get_feature_importance(self, category, top_n=20):
        """
        Get the most important features (words) for a given category.
        
        Args:
            category (str): The category to analyze
            top_n (int): Number of top features to return
            
        Returns:
            list: List of (feature, importance_score) tuples
        """
        if not self.is_trained:
            return []
        
        try:
            # Get the category index
            category_idx = list(self.pipeline.classes_).index(category)
            
            # Get feature names and weights
            feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
            feature_weights = self.pipeline.named_steps['classifier'].coef_[category_idx]
            
            # Sort features by importance
            feature_importance = list(zip(feature_names, feature_weights))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return feature_importance[:top_n]
            
        except (ValueError, IndexError):
            return []
    
    def evaluate_model(self, test_data):
        """
        Evaluate model performance on test data.
        
        Args:
            test_data (list): List of dictionaries with 'text' and 'category' keys
            
        Returns:
            dict: Evaluation metrics
        """
        if not test_data:
            return {}
        
        X_test = [self._preprocess_text(item['text']) for item in test_data]
        y_test = [item['category'] for item in test_data]
        
        # Get predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate accuracy
        accuracy = sum(1 for true, pred in zip(y_test, y_pred) if true == pred) / len(y_test)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'actual': y_test
        }
