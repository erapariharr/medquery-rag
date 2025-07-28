from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
import re

# Global pipeline to avoid reloading
_qa_pipeline = None
_summarizer_pipeline = None

def get_qa_pipeline():
    """Get or create the QA pipeline with a lightweight model"""
    global _qa_pipeline
    if _qa_pipeline is None:
        print("🔄 Loading language model...")
        
        # Use a smaller, more manageable model
        model_name = "microsoft/DialoGPT-medium"
        
        try:
            device = 0 if torch.cuda.is_available() else -1
            _qa_pipeline = pipeline(
                "text-generation", 
                model=model_name, 
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                max_length=1024
            )
            print(f"✅ Language model loaded on {'GPU' if device == 0 else 'CPU'}")
            
        except Exception as e:
            print(f"⚠️ Error loading {model_name}, falling back to distilgpt2: {e}")
            _qa_pipeline = pipeline(
                "text-generation", 
                model="distilgpt2", 
                device=-1
            )
            print("✅ Fallback model loaded")
            
    return _qa_pipeline

def get_summarizer_pipeline():
    """Get or create the summarization pipeline"""
    global _summarizer_pipeline
    if _summarizer_pipeline is None:
        print("🔄 Loading summarization model...")
        
        try:
            # Use BART for summarization (good balance of quality and speed)
            device = 0 if torch.cuda.is_available() else -1
            _summarizer_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print(f"✅ Summarization model loaded on {'GPU' if device == 0 else 'CPU'}")
            
        except Exception as e:
            print(f"⚠️ Error loading BART, trying smaller model: {e}")
            try:
                # Fallback to smaller model
                _summarizer_pipeline = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=-1  # Force CPU for stability
                )
                print("✅ Fallback summarization model loaded")
            except Exception as e2:
                print(f"⚠️ Summarization model failed, will use extractive summarization: {e2}")
                _summarizer_pipeline = None
                
    return _summarizer_pipeline

def generate_article_summaries(contexts: List[Dict]) -> List[Dict]:
    """
    Generate summaries for each article
    """
    print("📝 Generating article summaries...")
    
    enhanced_contexts = []
    summarizer = get_summarizer_pipeline()
    
    for i, ctx in enumerate(contexts, 1):
        try:
            print(f"🔄 Summarizing article {i}/{len(contexts)}...")
            
            # Get article content
            title = ctx['metadata']['title']
            abstract = ctx['text']
            
            # Generate summary
            if summarizer is not None:
                summary = generate_ai_summary(abstract, title, summarizer)
            else:
                summary = generate_extractive_summary(abstract, title)
            
            # Add summary to context
            enhanced_ctx = ctx.copy()
            enhanced_ctx['summary'] = summary
            enhanced_contexts.append(enhanced_ctx)
            
        except Exception as e:
            print(f"⚠️ Error summarizing article {i}: {e}")
            # Add context without summary
            enhanced_ctx = ctx.copy()
            enhanced_ctx['summary'] = generate_fallback_summary(ctx['text'], ctx['metadata']['title'])
            enhanced_contexts.append(enhanced_ctx)
    
    print("✅ Article summaries generated")
    return enhanced_contexts

def generate_ai_summary(abstract: str, title: str, summarizer) -> str:
    """
    Generate AI-powered summary using transformer model
    """
    try:
        # Prepare text for summarization
        # Use both title and abstract for context
        input_text = f"{title}. {abstract}"
        
        # Limit input length for the model
        max_input_length = 1024
        if len(input_text) > max_input_length:
            input_text = input_text[:max_input_length]
        
        # Generate summary
        summary_result = summarizer(
            input_text,
            max_length=100,  # Concise summary
            min_length=30,
            do_sample=False,
            truncation=True
        )
        
        summary = summary_result[0]['summary_text']
        
        # Clean up the summary
        summary = clean_summary_text(summary)
        
        return summary
        
    except Exception as e:
        print(f"⚠️ AI summarization failed: {e}")
        return generate_extractive_summary(abstract, title)

def generate_extractive_summary(abstract: str, title: str) -> str:
    """
    Generate extractive summary by selecting key sentences
    """
    try:
        # Split abstract into sentences
        sentences = re.split(r'[.!?]+', abstract)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return f"This study focuses on {title.lower()}."
        
        # Score sentences for importance
        scored_sentences = []
        
        # Keywords that indicate important information
        important_keywords = [
            'objective', 'aim', 'purpose', 'goal',
            'method', 'approach', 'design',
            'result', 'finding', 'outcome', 'conclusion',
            'significant', 'effective', 'treatment', 'therapy',
            'recommendation', 'guideline', 'protocol'
        ]
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Score based on important keywords
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Prefer sentences with numbers (often key findings)
            if re.search(r'\d+', sentence):
                score += 1
            
            # Prefer longer sentences (more informative)
            if len(sentence) > 100:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:2]]  # Top 2 sentences
        
        summary = '. '.join(top_sentences)
        if not summary.endswith('.'):
            summary += '.'
        
        # Add context about what the study covers
        study_focus = extract_study_focus(title, abstract)
        if study_focus:
            summary = f"{study_focus} {summary}"
        
        return clean_summary_text(summary)
        
    except Exception as e:
        print(f"⚠️ Extractive summarization failed: {e}")
        return generate_fallback_summary(abstract, title)

def extract_study_focus(title: str, abstract: str) -> str:
    """
    Extract what the study focuses on
    """
    title_lower = title.lower()
    
    if any(word in title_lower for word in ['review', 'systematic review', 'meta-analysis']):
        return "This systematic review examines"
    elif any(word in title_lower for word in ['trial', 'study', 'clinical']):
        return "This clinical study investigates"
    elif any(word in title_lower for word in ['guideline', 'recommendation']):
        return "These guidelines address"
    elif any(word in title_lower for word in ['treatment', 'therapy', 'management']):
        return "This research focuses on treatment approaches for"
    else:
        return "This study explores"

def generate_fallback_summary(text: str, title: str) -> str:
    """
    Generate a simple fallback summary when other methods fail
    """
    # Extract first meaningful sentence from abstract
    sentences = re.split(r'[.!?]+', text)
    first_sentence = ""
    
    for sentence in sentences:
        if len(sentence.strip()) > 30:
            first_sentence = sentence.strip()
            break
    
    if first_sentence:
        return f"This article discusses {title.lower()}. {first_sentence}."
    else:
        return f"This article provides information about {title.lower()}."

def clean_summary_text(summary: str) -> str:
    """
    Clean and format summary text
    """
    # Remove extra whitespace
    summary = re.sub(r'\s+', ' ', summary)
    
    # Fix punctuation
    summary = summary.replace(' .', '.')
    summary = summary.replace(' ,', ',')
    
    # Ensure proper capitalization
    if summary and not summary[0].isupper():
        summary = summary[0].upper() + summary[1:]
    
    # Ensure it ends with punctuation
    if summary and summary[-1] not in '.!?':
        summary += '.'
    
    return summary.strip()

def generate_answer(question: str, contexts: List[Dict]) -> str:
    """
    Generate an evidence-based answer using retrieved contexts with summaries
    """
    if not contexts:
        return "❌ No relevant evidence found to answer this question."
    
    print("🔄 Generating evidence-based answer...")
    
    try:
        # First, generate summaries for each article
        enhanced_contexts = generate_article_summaries(contexts)
        
        # Prepare context string with summaries
        context_str = ""
        for i, ctx in enumerate(enhanced_contexts, 1):
            title = ctx['metadata']['title']
            summary = ctx.get('summary', 'Summary not available')
            text = ctx['text'][:300]  # Limit context length
            pmid = ctx['metadata'].get('pmid', 'Unknown')
            
            context_str += f"\n[Source {i}] {title} (PMID: {pmid})\nSummary: {summary}\nDetails: {text}...\n"
        
        # Create prompt
        prompt = f"""Based on the following medical literature with summaries, provide an evidence-based answer to the clinical question. Include specific citations and reasoning.

Question: {question}

Medical Literature with Summaries:
{context_str}

Evidence-Based Answer:"""
        
        # Generate response
        pipeline = get_qa_pipeline()
        
        response = pipeline(
            prompt, 
            max_new_tokens=350,
            temperature=0.7,
            do_sample=True,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            truncation=True
        )
        
        # Extract the generated answer
        generated_text = response[0]['generated_text']
        answer = generated_text.split("Evidence-Based Answer:")[-1].strip()
        
        # Format answer with summaries and citations
        formatted_answer = format_answer_with_summaries(answer, enhanced_contexts)
        
        return formatted_answer
        
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return generate_fallback_answer_with_summaries(question, contexts)

def format_answer_with_summaries(answer: str, contexts: List[Dict]) -> str:
    """Format the answer with article summaries and citations"""
    
    # Add article summaries section
    summaries_section = "\n\n📚 **Article Summaries:**\n"
    for i, ctx in enumerate(contexts, 1):
        title = ctx['metadata']['title']
        summary = ctx.get('summary', 'Summary not available')
        url = ctx['metadata']['url']
        
        summaries_section += f"\n**{i}. {title}**\n"
        summaries_section += f"   💡 *{summary}*\n"
        summaries_section += f"   🔗 {url}\n"
    
    return answer + summaries_section

def generate_fallback_answer_with_summaries(question: str, contexts: List[Dict]) -> str:
    """Generate a fallback answer with summaries when main pipeline fails"""
    
    print("🔄 Generating fallback answer with summaries...")
    
    # Generate summaries even for fallback
    enhanced_contexts = generate_article_summaries(contexts)
    
    answer = f"""Based on the retrieved medical literature, here are the key findings relevant to: "{question}"

**Summary of Evidence:**
"""
    
    for i, ctx in enumerate(enhanced_contexts, 1):
        summary = ctx.get('summary', 'Summary not available')
        answer += f"\n{i}. {summary}\n"
    
    # Add detailed sources
    answer += "\n\n📚 **Detailed Sources:**\n"
    for i, ctx in enumerate(enhanced_contexts, 1):
        title = ctx['metadata']['title']
        url = ctx['metadata']['url']
        summary = ctx.get('summary', 'Summary not available')
        
        answer += f"\n**{i}. {title}**\n"
        answer += f"   💡 *{summary}*\n"
        answer += f"   🔗 {url}\n"
    
    answer += "\n⚠️ *This is a summary of available evidence. Please consult with healthcare professionals for clinical decisions.*"
    
    return answer


# for version 1 and 2
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import torch
# from typing import List, Dict
# import re

# # Global pipeline to avoid reloading
# _qa_pipeline = None

# def get_qa_pipeline():
#     """Get or create the QA pipeline with a lightweight model"""
#     global _qa_pipeline
#     if _qa_pipeline is None:
#         print("🔄 Loading language model...")
        
#         # Use a smaller, more manageable model
#         model_name = "microsoft/DialoGPT-medium"  # Smaller alternative
#         # Alternative options:
#         # model_name = "google/flan-t5-base"  # Good for QA
#         # model_name = "facebook/opt-350m"    # Very lightweight
        
#         try:
#             device = 0 if torch.cuda.is_available() else -1
#             _qa_pipeline = pipeline(
#                 "text-generation", 
#                 model=model_name, 
#                 device=device,
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                 max_length=1024
#             )
#             print(f"✅ Language model loaded on {'GPU' if device == 0 else 'CPU'}")
            
#         except Exception as e:
#             print(f"⚠️ Error loading {model_name}, falling back to distilgpt2: {e}")
#             # Fallback to even smaller model
#             _qa_pipeline = pipeline(
#                 "text-generation", 
#                 model="distilgpt2", 
#                 device=-1  # Force CPU for stability
#             )
#             print("✅ Fallback model loaded")
            
#     return _qa_pipeline

# def generate_answer(question: str, contexts: List[Dict]) -> str:
#     """
#     Generate an evidence-based answer using retrieved contexts
    
#     Args:
#         question: The clinical question
#         contexts: List of context documents with metadata
        
#     Returns:
#         Generated answer with citations
#     """
#     if not contexts:
#         return "❌ No relevant evidence found to answer this question."
    
#     print("🔄 Generating evidence-based answer...")
    
#     try:
#         # Prepare context string
#         context_str = ""
#         for i, ctx in enumerate(contexts, 1):
#             title = ctx['metadata']['title']
#             text = ctx['text'][:500]  # Limit context length
#             pmid = ctx['metadata'].get('pmid', 'Unknown')
            
#             context_str += f"\n[Source {i}] {title} (PMID: {pmid})\n{text}\n"
        
#         # Create prompt
#         prompt = f"""Based on the following medical literature, provide an evidence-based answer to the clinical question. Include specific citations and reasoning.

# Question: {question}

# Medical Literature:
# {context_str}

# Evidence-Based Answer:"""
        
#         # Generate response
#         pipeline = get_qa_pipeline()
        
#         # Generate with appropriate parameters
#         response = pipeline(
#             prompt, 
#             max_new_tokens=300,
#             temperature=0.7,
#             do_sample=True,
#             pad_token_id=pipeline.tokenizer.eos_token_id,
#             truncation=True
#         )
        
#         # Extract the generated answer
#         generated_text = response[0]['generated_text']
#         answer = generated_text.split("Evidence-Based Answer:")[-1].strip()
        
#         # Post-process and add citations
#         formatted_answer = format_answer_with_citations(answer, contexts)
        
#         return formatted_answer
        
#     except Exception as e:
#         print(f"❌ Error generating answer: {e}")
#         return generate_fallback_answer(question, contexts)

# def format_answer_with_citations(answer: str, contexts: List[Dict]) -> str:
#     """Format the answer with proper citations"""
    
#     # Add citations at the end
#     citations = "\n\n📚 **Sources:**\n"
#     for i, ctx in enumerate(contexts, 1):
#         title = ctx['metadata']['title']
#         url = ctx['metadata']['url']
#         citations += f"{i}. {title}\n   {url}\n"
    
#     return answer + citations

# def generate_fallback_answer(question: str, contexts: List[Dict]) -> str:
#     """Generate a simple fallback answer when the main pipeline fails"""
    
#     print("🔄 Generating fallback answer...")
    
#     # Extract key information from contexts
#     key_findings = []
#     for ctx in contexts:
#         text = ctx['text'][:300]  # First 300 chars
#         key_findings.append(text)
    
#     # Create a simple template-based response
#     answer = f"""Based on the retrieved medical literature, here are the key findings relevant to: "{question}"

# **Summary of Evidence:**
# """
    
#     for i, finding in enumerate(key_findings, 1):
#         answer += f"\n{i}. {finding}...\n"
    
#     # Add citations
#     answer += "\n\n📚 **Sources:**\n"
#     for i, ctx in enumerate(contexts, 1):
#         title = ctx['metadata']['title']
#         url = ctx['metadata']['url']
#         answer += f"{i}. {title}\n   {url}\n"
    
#     answer += "\n⚠️ *This is a simplified summary. Please consult with healthcare professionals for clinical decisions.*"
    
#     return answer

# def clean_medical_text(text: str) -> str:
#     """Clean and format medical text for better readability"""
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text)
    
#     # Fix common formatting issues
#     text = text.replace(' .', '.')
#     text = text.replace(' ,', ',')
    
#     return text.strip()