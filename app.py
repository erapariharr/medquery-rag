import streamlit as st
from retriever import get_pubmed_articles
from embedder import create_faiss_index, search_similar_documents
from qa_chain import generate_answer
import traceback

# Configure Streamlit page
st.set_page_config(
    page_title="MedQuery: Clinical Decision Support",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2e7d32;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1e88e5;
    }
    .summary-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .relevance-score {
        background-color: #e8f5e8;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🩺 MedQuery: Evidence-Based Clinical Decision Support</h1>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("### About MedQuery")
    st.write("""
    This advanced tool helps clinicians by:
    
    1. ** Searching PubMed** for relevant research
    2. ** AI Matching** using semantic similarity  
    3. ** Generating Summaries** for each article
    4. ** Creating Answers** with proper citations
    5. ** Providing Links** to source papers
    """)
    
    st.markdown("### ✨ New Features")
    st.success("**Article Summaries**: Get quick overviews of what each paper covers!")
    st.info("**Smart Matching**: Improved relevance scoring for better results")
    
    st.markdown("### Example Questions")
    example_questions = [
        "What are the first-line treatments for type 2 diabetes?",
        "Contraindications for ACE inhibitors",
        "Side effects of metformin in elderly patients",
        "Treatment guidelines for hypertension",
        "Antibiotic resistance in pneumonia"
    ]
    
    for q in example_questions:
        if st.button(q, key=f"example_{hash(q)}"):
            st.session_state.query = q

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""

# Main query input
st.markdown("### 🔍 Enter Your Clinical Question")
query = st.text_input(
    "Ask a clinical question:",
    value=st.session_state.query,
    placeholder="e.g., What are the first-line treatments for type 2 diabetes in adults?",
    help="Enter your medical question in natural language"
)

# Advanced options
with st.expander("⚙️ Advanced Options"):
    col1, col2 = st.columns(2)
    with col1:
        max_articles = st.slider("Max articles to retrieve", 3, 15, 5)
    with col2:
        top_k = st.slider("Top results to analyze", 2, 5, 3)

# Main processing
if query:
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Retrieve articles from PubMed
        status_text.text("🔍 Searching PubMed database...")
        progress_bar.progress(10)
        
        with st.spinner("Fetching evidence from PubMed..."):
            articles = get_pubmed_articles(query, max_results=max_articles)
        
        progress_bar.progress(30)
        
        if not articles:
            st.error("❌ No relevant PubMed articles found. Try rephrasing your question or using different keywords.")
            st.stop()
        
        st.success(f"✅ Found {len(articles)} relevant articles")
        
        # Step 2: Create embeddings and search index
        status_text.text("🧠 Creating semantic embeddings...")
        progress_bar.progress(50)
        
        with st.spinner("Processing articles with AI embeddings..."):
            index, metadatas, texts = create_faiss_index(articles)
        
        if index is None or len(texts) == 0:
            st.error("❌ Failed to create search index. Please try again.")
            st.stop()
        
        progress_bar.progress(70)
        
        # Step 3: Search for most relevant documents
        status_text.text("🎯 Finding most relevant evidence...")
        
        with st.spinner("Finding most relevant evidence..."):
            similar_docs = search_similar_documents(index, metadatas, texts, query, k=top_k)
        
        progress_bar.progress(85)
        
        if not similar_docs:
            st.error("❌ No relevant matches found in the retrieved articles.")
            st.stop()
        
        # Step 4: Generate answer with summaries
        status_text.text("📝 Generating summaries and evidence-based answer...")
        
        with st.spinner("Generating article summaries and evidence-based recommendation..."):
            result = generate_answer(query, similar_docs)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display results
        st.markdown('<h2 class="section-header">🔬 Evidence-Based Recommendation</h2>', unsafe_allow_html=True)
        
        # Main answer
        st.markdown(result)
        
        # Medical disclaimer
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for clinical decisions.
        </div>
        """, unsafe_allow_html=True)
        
        # Show article summaries in a more prominent way
        st.markdown('<h2 class="section-header"> Quick Article Overview</h2>', unsafe_allow_html=True)
        
        for i, doc in enumerate(similar_docs, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"** Article {i}: {doc['metadata']['title']}**")
                
                with col2:
                    score = doc.get('similarity_score', 0)
                    st.markdown(f'<div class="relevance-score">Relevance: {score:.3f}</div>', unsafe_allow_html=True)
                
                # Show summary if available (this will be added by the enhanced qa_chain)
                if 'summary' in doc:
                    st.markdown(f"""
                    <div class="summary-box">
                     <strong>Summary:</strong> {doc['summary']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Article details
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**PMID:** {doc['metadata'].get('pmid', 'Unknown')}")
                with col_b:
                    st.markdown(f"**[📎 View on PubMed]({doc['metadata']['url']})**")
                
                st.markdown("---")
        
        # Brief explanation of relevance scores
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #4CAF50;">
        <strong> About Relevance Scores:</strong> These scores (0.0-1.0) indicate how well each article matches your query using semantic similarity. 
        Higher scores mean better matches. Our AI considers title relevance, abstract content, and keyword matching to rank articles.
        <em>Click the expandable section below for detailed scoring methodology.</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Add explanation of relevance scoring
        with st.expander("ℹ️ Understanding Relevance Scores - Detailed Methodology"):
            st.markdown("""
            ### 🎯 How We Calculate Relevance Scores
            
            Our system uses **two complementary scoring methods** to find the most relevant articles:
            
            ---
            
            #### 🔍 **Step 1: PubMed Retrieval Scoring** *(If using improved retriever)*
            
            When fetching articles, we score them based on:
            
            1. **🎯 Exact Query Match** *(Highest Priority)*
               - **Title match**: +10.0 points if your exact query appears in the article title
               - **Abstract match**: +5.0 points if found in the abstract
               
            2. **🏥 Medical Terms** *(High Priority)*
               - **Title**: +3.0 points per medical term (e.g., "type 2 diabetes", "first-line treatment")
               - **Abstract**: +1.5 points per medical term
               
            3. **🔤 Individual Keywords** *(Medium Priority)*
               - **Title words**: +2.0 points per matching word from your query
               - **Abstract words**: +0.5 points per matching word
               
            4. **📄 Content Quality** *(Base Priority)*
               - **Has abstract**: +1.0 point (articles with abstracts are more informative)
               - **Recent publication**: +0.5 points (newer guidelines and findings)
            
            ---
            
            #### 🧠 **Step 2: AI Semantic Similarity** *(Final Ranking)*
            
            The **Relevance Scores** you see (0.0-1.0) come from our AI semantic analysis:
            
            - **AI Embeddings**: We convert your query and each article into high-dimensional vectors
            - **Semantic Matching**: We measure how "close" the meanings are in this vector space
            - **Normalized Scoring**: Converted to 0.0-1.0 scale where:
              - **0.8-1.0**: Extremely relevant - perfect semantic match
              - **0.6-0.8**: Highly relevant - strong semantic similarity  
              - **0.4-0.6**: Moderately relevant - some semantic overlap
              - **0.2-0.4**: Weakly relevant - limited semantic connection
              - **0.0-0.2**: Minimally relevant - very different topics
            
            ---
            
            #### 🏆 **Why This Two-Step Approach?**
            
            1. **Step 1** ensures we get articles that contain your specific medical terms
            2. **Step 2** ensures we understand the *meaning* and *context* of your question
            3. **Combined**: You get articles that are both keyword-relevant AND semantically meaningful
            
            #### 💡 **Example for Query: *"What are the first-line treatments for type 2 diabetes?"***
            
            - **High Score (0.85)**: *"Metformin as first-line therapy for type 2 diabetes mellitus"*
            - **Medium Score (0.62)**: *"Comparative effectiveness of diabetes medications in adults"*  
            - **Lower Score (0.34)**: *"Insulin resistance mechanisms in metabolic syndrome"*
            
            **This dual scoring system ensures you get the most clinically relevant evidence for your question! 🎯**
            """)
            
            
        
        # Show detailed retrieved articles 
        with st.expander("📚 Detailed Article Information"):
            for i, doc in enumerate(similar_docs, 1):
                with st.container():
                    st.markdown(f"### 📖 Article {i}")
                    st.markdown(f"**Title:** {doc['metadata']['title']}")
                    st.markdown(f"**Similarity Score:** {doc['similarity_score']:.3f}")
                    st.markdown(f"**PMID:** {doc['metadata'].get('pmid', 'Unknown')}")
                    st.markdown(f"**URL:** {doc['metadata']['url']}")
                    
                    # Show snippet of text
                    snippet = doc['text'][:400] + "..." if len(doc['text']) > 400 else doc['text']
                    st.markdown(f"**Abstract Preview:**")
                    st.text_area("", snippet, height=100, key=f"abstract_{i}")
                    
                    if i < len(similar_docs):
                        st.markdown("---")
        
        # Show search statistics
        with st.expander("📊 Search Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Articles Retrieved", len(articles))
            with col2:
                st.metric("Articles Analyzed", len(similar_docs))
            with col3:
                avg_similarity = sum(d['similarity_score'] for d in similar_docs) / len(similar_docs)
                st.metric("Average Similarity", f"{avg_similarity:.3f}")
            with col4:
                # Show relevance scores from improved retriever if available
                if 'relevance_score' in articles[0]:
                    avg_relevance = sum(a.get('relevance_score', 0) for a in articles) / len(articles)
                    st.metric("Average Relevance", f"{avg_relevance:.3f}")
                else:
                    st.metric("Processing Time", "< 30s")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        with st.expander("Error Details (for debugging)"):
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    🚀 <strong>Enhanced with AI Summaries</strong> | Built with ❤️ for healthcare professionals | Data from PubMed/NCBI
</div>
""", unsafe_allow_html=True)

# # only give defination for relevance score
# import streamlit as st
# from retriever import get_pubmed_articles
# from embedder import create_faiss_index, search_similar_documents
# from qa_chain import generate_answer
# import traceback

# # Configure Streamlit page
# st.set_page_config(
#     page_title="MedQuery: Clinical Decision Support",
#     page_icon="🩺",
#     layout="wide"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1e88e5;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .section-header {
#         color: #2e7d32;
#         border-bottom: 2px solid #2e7d32;
#         padding-bottom: 0.5rem;
#         margin-top: 2rem;
#         margin-bottom: 1rem;
#     }
#     .source-box {
#         background-color: #f5f5f5;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#         border-left: 4px solid #1e88e5;
#     }
#     .summary-box {
#         background-color: #e3f2fd;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#         border-left: 4px solid #2196f3;
#     }
#     .warning-box {
#         background-color: #fff3e0;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #ff9800;
#         margin: 1rem 0;
#     }
#     .relevance-score {
#         background-color: #e8f5e8;
#         padding: 0.2rem 0.5rem;
#         border-radius: 0.3rem;
#         font-size: 0.8rem;
#         font-weight: bold;
#         color: #2e7d32;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Main title
# st.markdown('<h1 class="main-header">🩺 MedQuery: Evidence-Based Clinical Decision Support</h1>', unsafe_allow_html=True)

# # Sidebar with information
# with st.sidebar:
#     st.markdown("### About MedQuery")
#     st.write("""
#     This advanced tool helps clinicians by:
    
#     1. **🔍 Searching PubMed** for relevant research
#     2. **🧠 AI Matching** using semantic similarity  
#     3. **📝 Generating Summaries** for each article
#     4. **💡 Creating Answers** with proper citations
#     5. **🔗 Providing Links** to source papers
#     """)
    
#     st.markdown("### ✨ New Features")
#     st.success("**Article Summaries**: Get quick overviews of what each paper covers!")
#     st.info("**Smart Matching**: Improved relevance scoring for better results")
    
#     st.markdown("### Example Questions")
#     example_questions = [
#         "What are the first-line treatments for type 2 diabetes?",
#         "Contraindications for ACE inhibitors",
#         "Side effects of metformin in elderly patients",
#         "Treatment guidelines for hypertension",
#         "Antibiotic resistance in pneumonia"
#     ]
    
#     for q in example_questions:
#         if st.button(q, key=f"example_{hash(q)}"):
#             st.session_state.query = q

# # Initialize session state
# if 'query' not in st.session_state:
#     st.session_state.query = ""

# # Main query input
# st.markdown("### 🔍 Enter Your Clinical Question")
# query = st.text_input(
#     "Ask a clinical question:",
#     value=st.session_state.query,
#     placeholder="e.g., What are the first-line treatments for type 2 diabetes in adults?",
#     help="Enter your medical question in natural language"
# )

# # Advanced options
# with st.expander("⚙️ Advanced Options"):
#     col1, col2 = st.columns(2)
#     with col1:
#         max_articles = st.slider("Max articles to retrieve", 3, 15, 5)
#     with col2:
#         top_k = st.slider("Top results to analyze", 2, 5, 3)

# # Main processing
# if query:
#     try:
#         # Initialize progress tracking
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         # Step 1: Retrieve articles from PubMed
#         status_text.text("🔍 Searching PubMed database...")
#         progress_bar.progress(10)
        
#         with st.spinner("Fetching evidence from PubMed..."):
#             articles = get_pubmed_articles(query, max_results=max_articles)
        
#         progress_bar.progress(30)
        
#         if not articles:
#             st.error("❌ No relevant PubMed articles found. Try rephrasing your question or using different keywords.")
#             st.stop()
        
#         st.success(f"✅ Found {len(articles)} relevant articles")
        
#         # Step 2: Create embeddings and search index
#         status_text.text("🧠 Creating semantic embeddings...")
#         progress_bar.progress(50)
        
#         with st.spinner("Processing articles with AI embeddings..."):
#             index, metadatas, texts = create_faiss_index(articles)
        
#         if index is None or len(texts) == 0:
#             st.error("❌ Failed to create search index. Please try again.")
#             st.stop()
        
#         progress_bar.progress(70)
        
#         # Step 3: Search for most relevant documents
#         status_text.text("🎯 Finding most relevant evidence...")
        
#         with st.spinner("Finding most relevant evidence..."):
#             similar_docs = search_similar_documents(index, metadatas, texts, query, k=top_k)
        
#         progress_bar.progress(85)
        
#         if not similar_docs:
#             st.error("❌ No relevant matches found in the retrieved articles.")
#             st.stop()
        
#         # Step 4: Generate answer with summaries
#         status_text.text("📝 Generating summaries and evidence-based answer...")
        
#         with st.spinner("Generating article summaries and evidence-based recommendation..."):
#             result = generate_answer(query, similar_docs)
        
#         progress_bar.progress(100)
#         status_text.text("✅ Analysis complete!")
        
#         # Display results
#         st.markdown('<h2 class="section-header">🔬 Evidence-Based Recommendation</h2>', unsafe_allow_html=True)
        
#         # Main answer
#         st.markdown(result)
        
#         # Medical disclaimer
#         st.markdown("""
#         <div class="warning-box">
#         ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for clinical decisions.
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Show article summaries in a more prominent way
#         st.markdown('<h2 class="section-header">📋 Quick Article Overview</h2>', unsafe_allow_html=True)
        
#         # Explain relevance score
#         st.markdown("""
#         <div style="background-color: #f8f9fa; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 3px solid #6c757d;">
#         📊 <strong>Relevance Score Explained:</strong> This score (0.000-1.000) indicates how well each article matches your question. 
#         Higher scores mean the article is more likely to contain information directly relevant to your query. 
#         Scores above 0.800 indicate excellent matches, 0.600-0.800 are good matches, and below 0.600 may have limited relevance.
#         </div>
#         """, unsafe_allow_html=True)
        
#         for i, doc in enumerate(similar_docs, 1):
#             with st.container():
#                 col1, col2 = st.columns([3, 1])
                
#                 with col1:
#                     st.markdown(f"**📖 Article {i}: {doc['metadata']['title']}**")
                
#                 with col2:
#                     score = doc.get('similarity_score', 0)
#                     st.markdown(f'<div class="relevance-score">Relevance: {score:.3f}</div>', unsafe_allow_html=True)
                
#                 # Show summary if available (this will be added by the enhanced qa_chain)
#                 if 'summary' in doc:
#                     st.markdown(f"""
#                     <div class="summary-box">
#                     💡 <strong>Summary:</strong> {doc['summary']}
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 # Article details
#                 col_a, col_b = st.columns(2)
#                 with col_a:
#                     st.markdown(f"**PMID:** {doc['metadata'].get('pmid', 'Unknown')}")
#                 with col_b:
#                     st.markdown(f"**[📎 View on PubMed]({doc['metadata']['url']})**")
                
#                 st.markdown("---")
        
#         # Show detailed retrieved articles 
#         with st.expander("📚 Detailed Article Information"):
#             for i, doc in enumerate(similar_docs, 1):
#                 with st.container():
#                     st.markdown(f"### 📖 Article {i}")
#                     st.markdown(f"**Title:** {doc['metadata']['title']}")
#                     st.markdown(f"**Similarity Score:** {doc['similarity_score']:.3f}")
#                     st.markdown(f"**PMID:** {doc['metadata'].get('pmid', 'Unknown')}")
#                     st.markdown(f"**URL:** {doc['metadata']['url']}")
                    
#                     # Show snippet of text
#                     snippet = doc['text'][:400] + "..." if len(doc['text']) > 400 else doc['text']
#                     st.markdown(f"**Abstract Preview:**")
#                     st.text_area("", snippet, height=100, key=f"abstract_{i}")
                    
#                     if i < len(similar_docs):
#                         st.markdown("---")
        
#         # Show search statistics
#         with st.expander("📊 Search Statistics"):
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Articles Retrieved", len(articles))
#             with col2:
#                 st.metric("Articles Analyzed", len(similar_docs))
#             with col3:
#                 avg_similarity = sum(d['similarity_score'] for d in similar_docs) / len(similar_docs)
#                 st.metric("Average Similarity", f"{avg_similarity:.3f}")
#             with col4:
#                 # Show relevance scores from improved retriever if available
#                 if 'relevance_score' in articles[0]:
#                     avg_relevance = sum(a.get('relevance_score', 0) for a in articles) / len(articles)
#                     st.metric("Average Relevance", f"{avg_relevance:.3f}")
#                 else:
#                     st.metric("Processing Time", "< 30s")
        
#     except Exception as e:
#         st.error(f"❌ An error occurred: {str(e)}")
#         with st.expander("Error Details (for debugging)"):
#             st.code(traceback.format_exc())

# # Footer
# st.markdown("---")
# st.markdown("""
# <div style="text-align: center; color: #666; margin-top: 2rem;">
#     🚀 <strong>Enhanced with AI Summaries</strong> | Built with ❤️ for healthcare professionals | Data from PubMed/NCBI
# </div>
# """, unsafe_allow_html=True)

# this does show summary but does not show relevance score meaning
# import streamlit as st
# from retriever import get_pubmed_articles
# from embedder import create_faiss_index, search_similar_documents
# from qa_chain import generate_answer
# import traceback

# # Configure Streamlit page
# st.set_page_config(
#     page_title="MedQuery: Clinical Decision Support",
#     page_icon="🩺",
#     layout="wide"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1e88e5;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .section-header {
#         color: #2e7d32;
#         border-bottom: 2px solid #2e7d32;
#         padding-bottom: 0.5rem;
#         margin-top: 2rem;
#         margin-bottom: 1rem;
#     }
#     .source-box {
#         background-color: #f5f5f5;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#         border-left: 4px solid #1e88e5;
#     }
#     .summary-box {
#         background-color: #e3f2fd;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#         border-left: 4px solid #2196f3;
#     }
#     .warning-box {
#         background-color: #fff3e0;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #ff9800;
#         margin: 1rem 0;
#     }
#     .relevance-score {
#         background-color: #e8f5e8;
#         padding: 0.2rem 0.5rem;
#         border-radius: 0.3rem;
#         font-size: 0.8rem;
#         font-weight: bold;
#         color: #2e7d32;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Main title
# st.markdown('<h1 class="main-header">🩺 MedQuery: Evidence-Based Clinical Decision Support</h1>', unsafe_allow_html=True)

# # Sidebar with information
# with st.sidebar:
#     st.markdown("### About MedQuery")
#     st.write("""
#     This advanced tool helps clinicians by:
    
#     1. **🔍 Searching PubMed** for relevant research
#     2. **🧠 AI Matching** using semantic similarity  
#     3. **📝 Generating Summaries** for each article
#     4. **💡 Creating Answers** with proper citations
#     5. **🔗 Providing Links** to source papers
#     """)
    
#     st.markdown("### ✨ New Features")
#     st.success("**Article Summaries**: Get quick overviews of what each paper covers!")
#     st.info("**Smart Matching**: Improved relevance scoring for better results")
    
#     st.markdown("### Example Questions")
#     example_questions = [
#         "What are the first-line treatments for type 2 diabetes?",
#         "Contraindications for ACE inhibitors",
#         "Side effects of metformin in elderly patients",
#         "Treatment guidelines for hypertension",
#         "Antibiotic resistance in pneumonia"
#     ]
    
#     for q in example_questions:
#         if st.button(q, key=f"example_{hash(q)}"):
#             st.session_state.query = q

# # Initialize session state
# if 'query' not in st.session_state:
#     st.session_state.query = ""

# # Main query input
# st.markdown("### 🔍 Enter Your Clinical Question")
# query = st.text_input(
#     "Ask a clinical question:",
#     value=st.session_state.query,
#     placeholder="e.g., What are the first-line treatments for type 2 diabetes in adults?",
#     help="Enter your medical question in natural language"
# )

# # Advanced options
# with st.expander("⚙️ Advanced Options"):
#     col1, col2 = st.columns(2)
#     with col1:
#         max_articles = st.slider("Max articles to retrieve", 3, 15, 5)
#     with col2:
#         top_k = st.slider("Top results to analyze", 2, 5, 3)

# # Main processing
# if query:
#     try:
#         # Initialize progress tracking
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         # Step 1: Retrieve articles from PubMed
#         status_text.text("🔍 Searching PubMed database...")
#         progress_bar.progress(10)
        
#         with st.spinner("Fetching evidence from PubMed..."):
#             articles = get_pubmed_articles(query, max_results=max_articles)
        
#         progress_bar.progress(30)
        
#         if not articles:
#             st.error("❌ No relevant PubMed articles found. Try rephrasing your question or using different keywords.")
#             st.stop()
        
#         st.success(f"✅ Found {len(articles)} relevant articles")
        
#         # Step 2: Create embeddings and search index
#         status_text.text("🧠 Creating semantic embeddings...")
#         progress_bar.progress(50)
        
#         with st.spinner("Processing articles with AI embeddings..."):
#             index, metadatas, texts = create_faiss_index(articles)
        
#         if index is None or len(texts) == 0:
#             st.error("❌ Failed to create search index. Please try again.")
#             st.stop()
        
#         progress_bar.progress(70)
        
#         # Step 3: Search for most relevant documents
#         status_text.text("🎯 Finding most relevant evidence...")
        
#         with st.spinner("Finding most relevant evidence..."):
#             similar_docs = search_similar_documents(index, metadatas, texts, query, k=top_k)
        
#         progress_bar.progress(85)
        
#         if not similar_docs:
#             st.error("❌ No relevant matches found in the retrieved articles.")
#             st.stop()
        
#         # Step 4: Generate answer with summaries
#         status_text.text("📝 Generating summaries and evidence-based answer...")
        
#         with st.spinner("Generating article summaries and evidence-based recommendation..."):
#             result = generate_answer(query, similar_docs)
        
#         progress_bar.progress(100)
#         status_text.text("✅ Analysis complete!")
        
#         # Display results
#         st.markdown('<h2 class="section-header">🔬 Evidence-Based Recommendation</h2>', unsafe_allow_html=True)
        
#         # Main answer
#         st.markdown(result)
        
#         # Medical disclaimer
#         st.markdown("""
#         <div class="warning-box">
#         ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for clinical decisions.
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Show article summaries in a more prominent way
#         st.markdown('<h2 class="section-header">📋 Quick Article Overview</h2>', unsafe_allow_html=True)
        
#         for i, doc in enumerate(similar_docs, 1):
#             with st.container():
#                 col1, col2 = st.columns([3, 1])
                
#                 with col1:
#                     st.markdown(f"**📖 Article {i}: {doc['metadata']['title']}**")
                
#                 with col2:
#                     score = doc.get('similarity_score', 0)
#                     st.markdown(f'<div class="relevance-score">Relevance: {score:.3f}</div>', unsafe_allow_html=True)
                
#                 # Show summary if available (this will be added by the enhanced qa_chain)
#                 if 'summary' in doc:
#                     st.markdown(f"""
#                     <div class="summary-box">
#                     💡 <strong>Summary:</strong> {doc['summary']}
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 # Article details
#                 col_a, col_b = st.columns(2)
#                 with col_a:
#                     st.markdown(f"**PMID:** {doc['metadata'].get('pmid', 'Unknown')}")
#                 with col_b:
#                     st.markdown(f"**[📎 View on PubMed]({doc['metadata']['url']})**")
                
#                 st.markdown("---")
        
#         # Show detailed retrieved articles 
#         with st.expander("📚 Detailed Article Information"):
#             for i, doc in enumerate(similar_docs, 1):
#                 with st.container():
#                     st.markdown(f"### 📖 Article {i}")
#                     st.markdown(f"**Title:** {doc['metadata']['title']}")
#                     st.markdown(f"**Similarity Score:** {doc['similarity_score']:.3f}")
#                     st.markdown(f"**PMID:** {doc['metadata'].get('pmid', 'Unknown')}")
#                     st.markdown(f"**URL:** {doc['metadata']['url']}")
                    
#                     # Show snippet of text
#                     snippet = doc['text'][:400] + "..." if len(doc['text']) > 400 else doc['text']
#                     st.markdown(f"**Abstract Preview:**")
#                     st.text_area("", snippet, height=100, key=f"abstract_{i}")
                    
#                     if i < len(similar_docs):
#                         st.markdown("---")
        
#         # Show search statistics
#         with st.expander("📊 Search Statistics"):
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Articles Retrieved", len(articles))
#             with col2:
#                 st.metric("Articles Analyzed", len(similar_docs))
#             with col3:
#                 avg_similarity = sum(d['similarity_score'] for d in similar_docs) / len(similar_docs)
#                 st.metric("Average Similarity", f"{avg_similarity:.3f}")
#             with col4:
#                 # Show relevance scores from improved retriever if available
#                 if 'relevance_score' in articles[0]:
#                     avg_relevance = sum(a.get('relevance_score', 0) for a in articles) / len(articles)
#                     st.metric("Average Relevance", f"{avg_relevance:.3f}")
#                 else:
#                     st.metric("Processing Time", "< 30s")
        
#     except Exception as e:
#         st.error(f"❌ An error occurred: {str(e)}")
#         with st.expander("Error Details (for debugging)"):
#             st.code(traceback.format_exc())

# # Footer
# st.markdown("---")
# st.markdown("""
# <div style="text-align: center; color: #666; margin-top: 2rem;">
#     🚀 <strong>Enhanced with AI Summaries</strong> | Built with ❤️ for healthcare professionals | Data from PubMed/NCBI
# </div>
# """, unsafe_allow_html=True)




# # for version 1 and 2 -> wihout summaries
# # import streamlit as st
# # from retriever import get_pubmed_articles
# # from embedder import create_faiss_index, search_similar_documents
# # from qa_chain import generate_answer
# # import traceback

# # # Configure Streamlit page
# # st.set_page_config(
# #     page_title="MedQuery: Clinical Decision Support",
# #     page_icon="🩺",
# #     layout="wide"
# # )

# # # Custom CSS for better styling
# # st.markdown("""
# # <style>
# #     .main-header {
# #         font-size: 2.5rem;
# #         color: #1e88e5;
# #         text-align: center;
# #         margin-bottom: 2rem;
# #     }
# #     .section-header {
# #         color: #2e7d32;
# #         border-bottom: 2px solid #2e7d32;
# #         padding-bottom: 0.5rem;
# #         margin-top: 2rem;
# #         margin-bottom: 1rem;
# #     }
# #     .source-box {
# #         background-color: #f5f5f5;
# #         padding: 1rem;
# #         border-radius: 0.5rem;
# #         margin: 0.5rem 0;
# #         border-left: 4px solid #1e88e5;
# #     }
# #     .warning-box {
# #         background-color: #fff3e0;
# #         padding: 1rem;
# #         border-radius: 0.5rem;
# #         border-left: 4px solid #ff9800;
# #         margin: 1rem 0;
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # # Main title
# # st.markdown('<h1 class="main-header">🩺 MedQuery: Evidence-Based Clinical Decision Support</h1>', unsafe_allow_html=True)

# # # Sidebar with information
# # with st.sidebar:
# #     st.markdown("### About MedQuery")
# #     st.write("""
# #     This tool helps clinicians get evidence-based answers by:
    
# #     1. **Searching PubMed** for relevant research
# #     2. **Analyzing abstracts** using AI embeddings
# #     3. **Generating answers** with proper citations
# #     4. **Providing direct links** to source papers
# #     """)
    
# #     st.markdown("### Example Questions")
# #     example_questions = [
# #         "What are the first-line treatments for type 2 diabetes?",
# #         "Contraindications for ACE inhibitors",
# #         "Side effects of metformin in elderly patients",
# #         "Treatment guidelines for hypertension",
# #         "Antibiotic resistance in pneumonia"
# #     ]
    
# #     for q in example_questions:
# #         if st.button(q, key=f"example_{hash(q)}"):
# #             st.session_state.query = q

# # # Initialize session state
# # if 'query' not in st.session_state:
# #     st.session_state.query = ""

# # # Main query input
# # st.markdown("### 🔍 Enter Your Clinical Question")
# # query = st.text_input(
# #     "Ask a clinical question:",
# #     value=st.session_state.query,
# #     placeholder="e.g., What are the first-line treatments for type 2 diabetes in adults?",
# #     help="Enter your medical question in natural language"
# # )

# # # Advanced options
# # with st.expander("⚙️ Advanced Options"):
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         max_articles = st.slider("Max articles to retrieve", 3, 15, 5)
# #     with col2:
# #         top_k = st.slider("Top results to analyze", 2, 5, 3)

# # # Main processing
# # if query:
# #     try:
# #         # Initialize progress tracking
# #         progress_bar = st.progress(0)
# #         status_text = st.empty()
        
# #         # Step 1: Retrieve articles from PubMed
# #         status_text.text("🔍 Searching PubMed database...")
# #         progress_bar.progress(10)
        
# #         with st.spinner("Fetching evidence from PubMed..."):
# #             articles = get_pubmed_articles(query, max_results=max_articles)
        
# #         progress_bar.progress(30)
        
# #         if not articles:
# #             st.error("❌ No relevant PubMed articles found. Try rephrasing your question or using different keywords.")
# #             st.stop()
        
# #         st.success(f"✅ Found {len(articles)} relevant articles")
        
# #         # Step 2: Create embeddings and search index
# #         status_text.text("🧠 Creating semantic embeddings...")
# #         progress_bar.progress(50)
        
# #         with st.spinner("Processing articles with AI embeddings..."):
# #             index, metadatas, texts = create_faiss_index(articles)
        
# #         if index is None or len(texts) == 0:
# #             st.error("❌ Failed to create search index. Please try again.")
# #             st.stop()
        
# #         progress_bar.progress(70)
        
# #         # Step 3: Search for most relevant documents
# #         status_text.text("🎯 Finding most relevant evidence...")
        
# #         with st.spinner("Finding most relevant evidence..."):
# #             similar_docs = search_similar_documents(index, metadatas, texts, query, k=top_k)
        
# #         progress_bar.progress(85)
        
# #         if not similar_docs:
# #             st.error("❌ No relevant matches found in the retrieved articles.")
# #             st.stop()
        
# #         # Step 4: Generate answer
# #         status_text.text("💭 Generating evidence-based answer...")
        
# #         with st.spinner("Generating evidence-based recommendation..."):
# #             result = generate_answer(query, similar_docs)
        
# #         progress_bar.progress(100)
# #         status_text.text("✅ Analysis complete!")
        
# #         # Display results
# #         st.markdown('<h2 class="section-header">🔬 Evidence-Based Recommendation</h2>', unsafe_allow_html=True)
        
# #         # Main answer
# #         st.markdown(result)
        
# #         # Medical disclaimer
# #         st.markdown("""
# #         <div class="warning-box">
# #         ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for clinical decisions.
# #         </div>
# #         """, unsafe_allow_html=True)
        
# #         # Show retrieved articles details
# #         with st.expander("📚 View Retrieved Articles Details"):
# #             for i, doc in enumerate(similar_docs, 1):
# #                 with st.container():
# #                     st.markdown(f"**Article {i}:** {doc['metadata']['title']}")
# #                     st.markdown(f"**Similarity Score:** {doc['similarity_score']:.3f}")
# #                     st.markdown(f"**PMID:** {doc['metadata'].get('pmid', 'Unknown')}")
# #                     st.markdown(f"**URL:** {doc['metadata']['url']}")
                    
# #                     # Show snippet of text
# #                     snippet = doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
# #                     st.markdown(f"**Abstract Preview:** {snippet}")
# #                     st.markdown("---")
        
# #         # Show search statistics
# #         with st.expander("📊 Search Statistics"):
# #             col1, col2, col3 = st.columns(3)
# #             with col1:
# #                 st.metric("Articles Retrieved", len(articles))
# #             with col2:
# #                 st.metric("Articles Analyzed", len(similar_docs))
# #             with col3:
# #                 st.metric("Average Similarity", f"{sum(d['similarity_score'] for d in similar_docs) / len(similar_docs):.3f}")
        
# #     except Exception as e:
# #         st.error(f"❌ An error occurred: {str(e)}")
# #         with st.expander("Error Details (for debugging)"):
# #             st.code(traceback.format_exc())

# # # Footer
# # st.markdown("---")
# # st.markdown("""
# # <div style="text-align: center; color: #666; margin-top: 2rem;">
# #     Built with ❤️ for healthcare professionals | Data from PubMed/NCBI
# # </div>
# # """, unsafe_allow_html=True)