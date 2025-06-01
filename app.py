import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

from email_classifier import EmailClassifier
from pii_masker import PIIMasker
from data_manager import DataManager
from utils import validate_email_format, get_confidence_color

# Initialize session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
if 'email_classifier' not in st.session_state:
    st.session_state.email_classifier = EmailClassifier()
if 'pii_masker' not in st.session_state:
    st.session_state.pii_masker = PIIMasker()

def main():
    st.set_page_config(
        page_title="Email Classification System",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 Email Classification System")
    st.subheader("Support Team Email Processing with PII Protection")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Email Processing", "Processed Emails", "Analytics", "Export Data"]
    )
    
    if page == "Email Processing":
        email_processing_page()
    elif page == "Processed Emails":
        processed_emails_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Export Data":
        export_data_page()

def email_processing_page():
    st.header("📝 Process New Email")
    
    # Email input form
    with st.form("email_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sender_email = st.text_input("Sender Email Address", placeholder="customer@example.com")
            subject = st.text_input("Email Subject", placeholder="Enter email subject")
            email_content = st.text_area(
                "Email Content", 
                height=200,
                placeholder="Enter the email content here..."
            )
        
        with col2:
            st.write("**Email Categories:**")
            st.write("• Technical Support")
            st.write("• Billing Inquiry")
            st.write("• General Inquiry")
            st.write("• Complaint")
            st.write("• Feature Request")
            
            st.write("**PII Detection:**")
            st.write("• Email addresses")
            st.write("• Phone numbers")
            st.write("• Names (using NLP)")
            st.write("• Addresses")
        
        submitted = st.form_submit_button("Process Email", type="primary")
        
        if submitted:
            # Validate inputs
            if not sender_email or not subject or not email_content:
                st.error("Please fill in all fields")
                return
            
            if not validate_email_format(sender_email):
                st.error("Please enter a valid email address")
                return
            
            # Process the email
            process_email(sender_email, subject, email_content)

def process_email(sender_email, subject, email_content):
    try:
        # Show processing status
        with st.spinner("Processing email..."):
            # Step 1: Detect and mask PII
            st.info("🔍 Detecting and masking PII...")
            masked_content, pii_mapping = st.session_state.pii_masker.mask_pii(email_content)
            masked_subject, subject_pii_mapping = st.session_state.pii_masker.mask_pii(subject)
            
            # Combine PII mappings
            combined_pii_mapping = {**pii_mapping, **subject_pii_mapping}
            
            # Step 2: Classify the masked email
            st.info("🤖 Classifying email...")
            classification_result = st.session_state.email_classifier.classify_email(
                masked_subject, masked_content
            )
            
            # Step 3: Store the processed email
            email_data = {
                'timestamp': datetime.now(),
                'sender': sender_email,
                'subject': subject,
                'content': email_content,
                'masked_subject': masked_subject,
                'masked_content': masked_content,
                'category': classification_result['category'],
                'confidence': classification_result['confidence'],
                'confidence_scores': classification_result['confidence_scores'],
                'pii_mapping': combined_pii_mapping,
                'pii_detected': len(combined_pii_mapping) > 0
            }
            
            st.session_state.data_manager.add_email(email_data)
        
        # Display results
        st.success("✅ Email processed successfully!")
        
        # Show classification results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Category", classification_result['category'])
        with col2:
            confidence_pct = f"{classification_result['confidence']:.1%}"
            st.metric("Confidence", confidence_pct)
        with col3:
            pii_count = len(combined_pii_mapping)
            st.metric("PII Items Detected", pii_count)
        
        # Show detailed results in expandable sections
        with st.expander("🔒 PII Masking Results", expanded=True):
            if combined_pii_mapping:
                st.write("**Detected PII:**")
                for token, original_value in combined_pii_mapping.items():
                    pii_type = st.session_state.pii_masker.get_pii_type(original_value)
                    st.write(f"• {pii_type}: `{original_value}` → `{token}`")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Masked Content:**")
                    st.text_area("Masked Subject", masked_subject, disabled=True, key="masked_subj")
                    st.text_area("Masked Content", masked_content, height=100, disabled=True, key="masked_cont")
                
                with col2:
                    st.write("**Original Content:**")
                    st.text_area("Original Subject", subject, disabled=True, key="orig_subj")
                    st.text_area("Original Content", email_content, height=100, disabled=True, key="orig_cont")
            else:
                st.info("No PII detected in this email")
        
        with st.expander("📊 Classification Details"):
            st.write("**Confidence Scores for All Categories:**")
            confidence_df = pd.DataFrame([
                {"Category": cat, "Confidence": score}
                for cat, score in classification_result['confidence_scores'].items()
            ])
            confidence_df = confidence_df.sort_values('Confidence', ascending=False)
            
            # Create confidence chart
            fig = px.bar(
                confidence_df, 
                x='Confidence', 
                y='Category',
                orientation='h',
                color='Confidence',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(confidence_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error processing email: {str(e)}")

def processed_emails_page():
    st.header("📋 Processed Emails")
    
    emails = st.session_state.data_manager.get_all_emails()
    
    if not emails:
        st.info("No emails have been processed yet. Go to the Email Processing page to get started.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories = ["All"] + list(set([email['category'] for email in emails]))
        selected_category = st.selectbox("Filter by Category", categories)
    
    with col2:
        pii_filter = st.selectbox("PII Status", ["All", "Contains PII", "No PII"])
    
    with col3:
        search_term = st.text_input("Search in subject/content")
    
    # Apply filters
    filtered_emails = emails
    
    if selected_category != "All":
        filtered_emails = [e for e in filtered_emails if e['category'] == selected_category]
    
    if pii_filter == "Contains PII":
        filtered_emails = [e for e in filtered_emails if e['pii_detected']]
    elif pii_filter == "No PII":
        filtered_emails = [e for e in filtered_emails if not e['pii_detected']]
    
    if search_term:
        search_term = search_term.lower()
        filtered_emails = [
            e for e in filtered_emails 
            if search_term in e['subject'].lower() or search_term in e['content'].lower()
        ]
    
    st.write(f"Showing {len(filtered_emails)} of {len(emails)} emails")
    
    # Display emails
    for i, email in enumerate(filtered_emails):
        with st.expander(
            f"📧 {email['subject'][:50]}{'...' if len(email['subject']) > 50 else ''} "
            f"({email['category']} - {email['confidence']:.1%})"
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**From:** {email['sender']}")
                st.write(f"**Timestamp:** {email['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Subject:** {email['subject']}")
                
                # Toggle between masked and original content
                show_original = st.checkbox(f"Show original content (unmasked)", key=f"toggle_{i}")
                
                if show_original:
                    if email['pii_detected']:
                        st.warning("⚠️ Showing unmasked content with PII")
                        unmasked_subject = st.session_state.pii_masker.unmask_pii(
                            email['masked_subject'], email['pii_mapping']
                        )
                        unmasked_content = st.session_state.pii_masker.unmask_pii(
                            email['masked_content'], email['pii_mapping']
                        )
                        st.text_area("Subject", unmasked_subject, disabled=True, key=f"unmask_subj_{i}")
                        st.text_area("Content", unmasked_content, height=100, disabled=True, key=f"unmask_cont_{i}")
                    else:
                        st.text_area("Subject", email['subject'], disabled=True, key=f"orig_subj_{i}")
                        st.text_area("Content", email['content'], height=100, disabled=True, key=f"orig_cont_{i}")
                else:
                    if email['pii_detected']:
                        st.info("🔒 Showing masked content (PII protected)")
                    st.text_area("Subject", email['masked_subject'], disabled=True, key=f"mask_subj_{i}")
                    st.text_area("Content", email['masked_content'], height=100, disabled=True, key=f"mask_cont_{i}")
            
            with col2:
                st.metric("Category", email['category'])
                confidence_color = get_confidence_color(email['confidence'])
                st.metric("Confidence", f"{email['confidence']:.1%}")
                
                if email['pii_detected']:
                    st.write("**PII Detected:**")
                    for token, original in email['pii_mapping'].items():
                        pii_type = st.session_state.pii_masker.get_pii_type(original)
                        st.write(f"• {pii_type}")
                else:
                    st.success("No PII detected")

def analytics_page():
    st.header("📊 Analytics Dashboard")
    
    emails = st.session_state.data_manager.get_all_emails()
    
    if not emails:
        st.info("No data available for analytics. Process some emails first.")
        return
    
    # Create analytics dataframe
    df = pd.DataFrame([
        {
            'category': email['category'],
            'confidence': email['confidence'],
            'pii_detected': email['pii_detected'],
            'pii_count': len(email['pii_mapping']),
            'timestamp': email['timestamp'],
            'date': email['timestamp'].date()
        }
        for email in emails
    ])
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails", len(emails))
    
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col3:
        pii_emails = df['pii_detected'].sum()
        pii_percentage = pii_emails / len(emails) * 100
        st.metric("Emails with PII", f"{pii_emails} ({pii_percentage:.1f}%)")
    
    with col4:
        most_common_category = df['category'].mode().iloc[0]
        st.metric("Top Category", most_common_category)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Email Categories Distribution")
        category_counts = df['category'].value_counts()
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Distribution of Email Categories"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Classification Confidence Distribution")
        fig_hist = px.histogram(
            df,
            x='confidence',
            nbins=20,
            title="Distribution of Classification Confidence",
            labels={'confidence': 'Confidence Score', 'count': 'Number of Emails'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Time series analysis
    if len(df) > 1:
        st.subheader("Emails Processed Over Time")
        daily_counts = df.groupby('date').size().reset_index(name='count')
        fig_line = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Daily Email Processing Volume",
            labels={'date': 'Date', 'count': 'Number of Emails'}
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    # PII Analysis
    st.subheader("PII Detection Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pii_by_category = df.groupby('category')['pii_detected'].agg(['sum', 'count']).reset_index()
        pii_by_category['pii_rate'] = pii_by_category['sum'] / pii_by_category['count']
        
        fig_pii = px.bar(
            pii_by_category,
            x='category',
            y='pii_rate',
            title="PII Detection Rate by Category",
            labels={'pii_rate': 'PII Detection Rate', 'category': 'Email Category'}
        )
        st.plotly_chart(fig_pii, use_container_width=True)
    
    with col2:
        pii_count_dist = df[df['pii_detected']]['pii_count'].value_counts().sort_index()
        fig_pii_count = px.bar(
            x=pii_count_dist.index,
            y=pii_count_dist.values,
            title="Distribution of PII Items per Email",
            labels={'x': 'Number of PII Items', 'y': 'Number of Emails'}
        )
        st.plotly_chart(fig_pii_count, use_container_width=True)

def export_data_page():
    st.header("📤 Export Data")
    
    emails = st.session_state.data_manager.get_all_emails()
    
    if not emails:
        st.info("No data available for export. Process some emails first.")
        return
    
    st.write(f"Available data: {len(emails)} processed emails")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Options")
        include_pii = st.checkbox("Include unmasked PII data", value=False)
        if include_pii:
            st.warning("⚠️ Exporting unmasked data may expose sensitive information")
        
        include_confidence_scores = st.checkbox("Include detailed confidence scores", value=True)
        include_timestamps = st.checkbox("Include timestamps", value=True)
    
    with col2:
        st.subheader("Data Preview")
        preview_df = st.session_state.data_manager.export_to_dataframe(
            include_pii=include_pii,
            include_confidence_scores=include_confidence_scores,
            include_timestamps=include_timestamps
        )
        st.write(f"Export will contain {len(preview_df)} rows and {len(preview_df.columns)} columns")
        st.dataframe(preview_df.head(), use_container_width=True)
    
    # Generate CSV
    if st.button("Generate CSV Export", type="primary"):
        try:
            csv_data = st.session_state.data_manager.export_to_csv(
                include_pii=include_pii,
                include_confidence_scores=include_confidence_scores,
                include_timestamps=include_timestamps
            )
            
            # Create download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"email_classification_export_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )
            
            st.success(f"✅ CSV export generated successfully! File: {filename}")
            
        except Exception as e:
            st.error(f"Error generating export: {str(e)}")

if __name__ == "__main__":
    main()
