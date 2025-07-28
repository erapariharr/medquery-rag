# version 2: gives out 5 to 15 articles
import requests
import json
import time
from typing import List, Dict, Tuple

def get_pubmed_articles(query: str, max_results: int = 5) -> List[Dict]:
    """
    Retrieve PubMed articles using improved matching strategy
    """
    print(f"➡️ Searching PubMed for: {query}")
    
    # Clean query for PubMed
    cleaned_query = clean_medical_query(query)
    print(f"🔍 PubMed search query: {cleaned_query}")
    
    # Strategy: Get more articles initially, then filter for best matches
    initial_fetch = min(max_results * 3, 15)  # Get 3x more articles to choose from
    
    # Step 1: Search with multiple strategies for better coverage
    all_articles = []
    
    # Primary search - most relevant
    articles_primary = search_pubmed_articles(cleaned_query, initial_fetch, "relevance")
    if articles_primary:
        all_articles.extend(articles_primary)
        print(f"✅ Primary search: {len(articles_primary)} articles")
    
    # Secondary search - recent articles for current guidelines
    if len(all_articles) < initial_fetch:
        articles_recent = search_pubmed_articles(cleaned_query, 
                                               initial_fetch - len(all_articles), 
                                               "pub_date")
        if articles_recent:
            # Avoid duplicates
            existing_pmids = {a['pmid'] for a in all_articles}
            new_articles = [a for a in articles_recent if a['pmid'] not in existing_pmids]
            all_articles.extend(new_articles)
            print(f"✅ Recent search: {len(new_articles)} new articles")
    
    if not all_articles:
        print("❌ No articles found")
        return []
    
    print(f"📚 Total articles retrieved: {len(all_articles)}")
    
    # Step 2: Re-rank articles by query relevance using our own scoring
    scored_articles = score_articles_by_relevance(all_articles, query)
    
    # Step 3: Return top matches
    top_articles = scored_articles[:max_results]
    
    print(f"🎯 Selected top {len(top_articles)} most relevant articles")
    for i, article in enumerate(top_articles, 1):
        print(f"   {i}. {article['title'][:60]}... (Score: {article.get('relevance_score', 0):.3f})")
    
    return top_articles

def search_pubmed_articles(query: str, max_results: int, sort_by: str) -> List[Dict]:
    """
    Search PubMed with specific sorting strategy
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": sort_by,
        "tool": "MedQuery",
        "email": "medquery@example.com"
    }
    
    try:
        # Search for article IDs
        search_data = make_request_with_retry(search_url, search_params, f"search-{sort_by}")
        if not search_data:
            return []
        
        if 'esearchresult' not in search_data or not search_data['esearchresult']['idlist']:
            return []
        
        article_ids = search_data['esearchresult']['idlist']
        
        # Fetch article details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(article_ids),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": "MedQuery",
            "email": "medquery@example.com"
        }
        
        time.sleep(0.5)  # Rate limiting
        
        fetch_response = make_request_with_retry(fetch_url, fetch_params, f"fetch-{sort_by}", return_json=False)
        if not fetch_response:
            return []
        
        # Parse articles
        articles = parse_pubmed_xml(fetch_response.text, article_ids)
        return articles
        
    except Exception as e:
        print(f"❌ Error in {sort_by} search: {e}")
        return []

def score_articles_by_relevance(articles: List[Dict], original_query: str) -> List[Dict]:
    """
    Score articles by relevance to the original query using multiple factors
    """
    print("🧮 Scoring articles for relevance...")
    
    query_lower = original_query.lower()
    query_words = set(query_lower.split())
    
    # Important medical terms that should boost relevance
    medical_terms = extract_medical_terms(query_lower)
    
    scored_articles = []
    
    for article in articles:
        score = calculate_relevance_score(article, query_words, medical_terms, query_lower)
        article_copy = article.copy()
        article_copy['relevance_score'] = score
        scored_articles.append(article_copy)
    
    # Sort by relevance score (highest first)
    scored_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return scored_articles

def calculate_relevance_score(article: Dict, query_words: set, medical_terms: List[str], full_query: str) -> float:
    """
    Calculate relevance score for an article
    """
    title = article.get('title', '').lower()
    abstract = article.get('abstract', '').lower()
    
    score = 0.0
    
    # 1. Exact phrase matching (highest weight)
    if full_query in title:
        score += 10.0
    elif full_query in abstract:
        score += 5.0
    
    # 2. Medical terms matching (high weight)
    for term in medical_terms:
        if term in title:
            score += 3.0
        elif term in abstract:
            score += 1.5
    
    # 3. Individual word matching
    title_words = set(title.split())
    abstract_words = set(abstract.split())
    
    # Title word matches (higher weight)
    title_matches = len(query_words.intersection(title_words))
    score += title_matches * 2.0
    
    # Abstract word matches
    abstract_matches = len(query_words.intersection(abstract_words))
    score += abstract_matches * 0.5
    
    # 4. Completeness bonus (articles with abstracts are better)
    if abstract and abstract != "no abstract available":
        score += 1.0
    
    # 5. Recency bonus (prefer recent articles, assume recent if no clear date)
    # This could be enhanced with actual publication date parsing
    score += 0.5
    
    return score

def extract_medical_terms(query: str) -> List[str]:
    """
    Extract important medical terms from query
    """
    # Common medical phrases that should be preserved as units
    medical_phrases = [
        "type 2 diabetes", "type 1 diabetes",
        "first-line treatment", "second-line treatment",
        "side effects", "adverse effects",
        "contraindications", "drug interactions",
        "ace inhibitors", "beta blockers",
        "blood pressure", "heart failure",
        "clinical trial", "systematic review",
        "meta-analysis", "guidelines"
    ]
    
    found_terms = []
    for phrase in medical_phrases:
        if phrase in query:
            found_terms.append(phrase)
    
    # Also add individual important words
    important_words = [
        "diabetes", "hypertension", "metformin", "insulin",
        "treatment", "therapy", "medication", "drug",
        "elderly", "pediatric", "pregnancy", "renal",
        "cardiovascular", "nephropathy", "retinopathy"
    ]
    
    query_words = query.split()
    for word in important_words:
        if word in query_words:
            found_terms.append(word)
    
    return list(set(found_terms))  # Remove duplicates

def make_request_with_retry(url: str, params: dict, request_type: str, max_retries: int = 3, return_json: bool = True):
    """
    Make HTTP request with retry logic for rate limiting
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt
                print(f"⏳ Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} attempts")
                    return None
            
            response.raise_for_status()
            
            if return_json:
                return response.json()
            else:
                return response
                
        except requests.RequestException as e:
            if "429" in str(e):
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"❌ Rate limit exceeded for {request_type}: {e}")
                    return None
            else:
                print(f"❌ Network error during {request_type}: {e}")
                return None
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON response: {e}")
            return None
    
    return None

def clean_medical_query(query: str) -> str:
    """
    Clean medical query while preserving important terms
    """
    query_lower = query.lower().strip()
    
    # Remove question words but keep medical context
    words_to_remove = [
        "what are the", "what is the", "what are", "what is",
        "how do", "how does", "how can", "how to",
        "when should", "when do", "when is",
        "why do", "why does", "why is",
        "where do", "where does", "where is"
    ]
    
    cleaned = query_lower
    for phrase in words_to_remove:
        cleaned = cleaned.replace(phrase, "")
    
    # Keep important medical terms together
    medical_stopwords = ["a", "an", "the", "of", "at", "by"]
    words = cleaned.split()
    filtered_words = [word for word in words if word not in medical_stopwords or len(words) <= 3]
    
    result = " ".join(filtered_words).strip()
    
    # Ensure we have enough content
    if len(result.split()) < 2:
        words = query_lower.replace("what are the", "").replace("what is the", "").split()
        result = " ".join(words[:6])
    
    return result

def parse_pubmed_xml(xml_content: str, article_ids: List[str]) -> List[Dict]:
    """
    Parse PubMed XML response and extract article information
    """
    import xml.etree.ElementTree as ET
    
    articles = []
    
    try:
        root = ET.fromstring(xml_content)
        
        for i, article_elem in enumerate(root.findall('.//PubmedArticle')):
            try:
                # Extract PMID
                pmid_elem = article_elem.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else article_ids[i] if i < len(article_ids) else "Unknown"
                
                # Extract title
                title_elem = article_elem.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else "No title available"
                
                # Extract abstract
                abstract_parts = []
                abstract_elems = article_elem.findall('.//AbstractText')
                
                if abstract_elems:
                    for abs_elem in abstract_elems:
                        label = abs_elem.get('Label', '')
                        text = abs_elem.text or ""
                        
                        if abs_elem.text is None:
                            text = ''.join(abs_elem.itertext())
                        
                        if label and text:
                            abstract_parts.append(f"{label}: {text}")
                        elif text:
                            abstract_parts.append(text)
                
                if abstract_parts:
                    abstract = " ".join(abstract_parts)
                else:
                    abstract_elem = article_elem.find('.//Abstract')
                    if abstract_elem is not None:
                        abstract = ''.join(abstract_elem.itertext())
                    else:
                        abstract = "No abstract available"
                
                abstract = abstract.strip()
                if not abstract:
                    abstract = "No abstract available"
                
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
                
                articles.append({
                    "title": title.strip() if title else "No title available",
                    "abstract": abstract,
                    "url": url,
                    "pmid": pmid
                })
                
            except Exception as e:
                print(f"⚠️ Error parsing individual article: {e}")
                continue
    
    except ET.ParseError as e:
        print(f"❌ Error parsing XML: {e}")
        return []
    
    return articles

# Test the improved matching
if __name__ == "__main__":
    print("🧪 Testing improved article matching...")
    
    query = "What are the first-line treatments for type 2 diabetes?"
    articles = get_pubmed_articles(query, max_results=5)
    
    if articles:
        print(f"\n✅ Retrieved {len(articles)} top-matched articles:")
        for i, article in enumerate(articles, 1):
            print(f"\n📖 Article {i} (Score: {article.get('relevance_score', 0):.3f}):")
            print(f"Title: {article['title']}")
            print(f"PMID: {article['pmid']}")
            print(f"Abstract: {article['abstract'][:150]}...")
    else:
        print("❌ No articles retrieved")

# # version 1: this worked- but without summary!!!
# import requests
# import json
# import time
# from typing import List, Dict

# def get_pubmed_articles(query: str, max_results: int = 5) -> List[Dict]:
#     """
#     Retrieve PubMed articles using the Entrez API with rate limiting
#     """
#     print(f"➡️ Searching PubMed for: {query}")
    
#     # Better query cleaning - preserve important medical terms
#     cleaned_query = clean_medical_query(query)
#     print(f"🔍 PubMed search query: {cleaned_query}")
    
#     # Step 1: Search for article IDs with rate limiting
#     search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
#     search_params = {
#         "db": "pubmed",
#         "term": cleaned_query,
#         "retmode": "json",
#         "retmax": max_results,
#         "sort": "relevance",
#         "tool": "MedQuery",
#         "email": "medquery@example.com"  # NCBI recommends providing tool and email
#     }
    
#     try:
#         # Get article IDs with retry logic
#         search_data = make_request_with_retry(search_url, search_params, "search")
#         if not search_data:
#             return []
        
#         if 'esearchresult' not in search_data or not search_data['esearchresult']['idlist']:
#             print("❌ No articles found for this query")
#             return []
        
#         article_ids = search_data['esearchresult']['idlist']
#         count = search_data['esearchresult'].get('count', 0)
#         print(f"✅ Found {len(article_ids)} articles (total available: {count})")
        
#         # Step 2: Fetch article details with rate limiting
#         fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
#         fetch_params = {
#             "db": "pubmed",
#             "id": ",".join(article_ids),
#             "retmode": "xml",
#             "rettype": "abstract",
#             "tool": "MedQuery",
#             "email": "medquery@example.com"
#         }
        
#         # Add delay before fetch request
#         time.sleep(0.5)  # 500ms delay between requests
        
#         fetch_response = make_request_with_retry(fetch_url, fetch_params, "fetch", return_json=False)
#         if not fetch_response:
#             return []
        
#         # Step 3: Parse XML and extract article information
#         articles = parse_pubmed_xml(fetch_response.text, article_ids)
#         print(f"📚 Successfully processed {len(articles)} articles")
        
#         return articles
        
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")
#         return []

# def make_request_with_retry(url: str, params: dict, request_type: str, max_retries: int = 3, return_json: bool = True):
#     """
#     Make HTTP request with retry logic for rate limiting
#     """
#     for attempt in range(max_retries):
#         try:
#             if attempt > 0:
#                 wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
#                 print(f"⏳ Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
#                 time.sleep(wait_time)
            
#             response = requests.get(url, params=params, timeout=20)
            
#             if response.status_code == 429:  # Rate limited
#                 if attempt < max_retries - 1:
#                     continue
#                 else:
#                     print(f"❌ Rate limit exceeded after {max_retries} attempts")
#                     return None
            
#             response.raise_for_status()
            
#             if return_json:
#                 return response.json()
#             else:
#                 return response
                
#         except requests.RequestException as e:
#             if "429" in str(e):  # Rate limiting
#                 if attempt < max_retries - 1:
#                     continue
#                 else:
#                     print(f"❌ Rate limit exceeded for {request_type}: {e}")
#                     return None
#             else:
#                 print(f"❌ Network error during {request_type}: {e}")
#                 return None
#         except json.JSONDecodeError as e:
#             print(f"❌ Error parsing JSON response: {e}")
#             return None
    
#     return None

# def clean_medical_query(query: str) -> str:
#     """
#     Clean medical query while preserving important terms
#     """
#     # Convert to lowercase for processing
#     query_lower = query.lower().strip()
    
#     # Remove common question words but keep medical terms
#     words_to_remove = [
#         "what are the", "what is the", "what are", "what is",
#         "how do", "how does", "how can", "how to",
#         "when should", "when do", "when is",
#         "why do", "why does", "why is",
#         "where do", "where does", "where is"
#     ]
    
#     cleaned = query_lower
#     for phrase in words_to_remove:
#         cleaned = cleaned.replace(phrase, "")
    
#     # Remove extra words but keep important medical context
#     medical_stopwords = ["a", "an", "the", "of", "at", "by", "for", "with", "without"]
#     words = cleaned.split()
#     filtered_words = [word for word in words if word not in medical_stopwords or len(words) <= 3]
    
#     result = " ".join(filtered_words).strip()
    
#     # If result is too short, use more of the original
#     if len(result.split()) < 2:
#         # Keep more words from original
#         words = query_lower.replace("what are the", "").replace("what is the", "").split()
#         result = " ".join(words[:6])  # Take first 6 words
    
#     return result

# def parse_pubmed_xml(xml_content: str, article_ids: List[str]) -> List[Dict]:
#     """
#     Parse PubMed XML response and extract article information
#     """
#     import xml.etree.ElementTree as ET
    
#     articles = []
    
#     try:
#         root = ET.fromstring(xml_content)
        
#         for i, article_elem in enumerate(root.findall('.//PubmedArticle')):
#             try:
#                 # Extract PMID
#                 pmid_elem = article_elem.find('.//PMID')
#                 pmid = pmid_elem.text if pmid_elem is not None else article_ids[i] if i < len(article_ids) else "Unknown"
                
#                 # Extract title
#                 title_elem = article_elem.find('.//ArticleTitle')
#                 title = title_elem.text if title_elem is not None else "No title available"
                
#                 # Extract abstract - handle different structures
#                 abstract_parts = []
                
#                 # Try structured abstract first
#                 abstract_elems = article_elem.findall('.//AbstractText')
#                 if abstract_elems:
#                     for abs_elem in abstract_elems:
#                         label = abs_elem.get('Label', '')
#                         text = abs_elem.text or ""
                        
#                         # Handle XML elements within abstract text
#                         if abs_elem.text is None:
#                             # Get all text content including from child elements
#                             text = ''.join(abs_elem.itertext())
                        
#                         if label and text:
#                             abstract_parts.append(f"{label}: {text}")
#                         elif text:
#                             abstract_parts.append(text)
                
#                 if abstract_parts:
#                     abstract = " ".join(abstract_parts)
#                 else:
#                     # Try to get abstract from other locations
#                     abstract_elem = article_elem.find('.//Abstract')
#                     if abstract_elem is not None:
#                         abstract = ''.join(abstract_elem.itertext())
#                     else:
#                         abstract = "No abstract available"
                
#                 # Clean up abstract
#                 abstract = abstract.strip()
#                 if not abstract or abstract == "":
#                     abstract = "No abstract available"
                
#                 # Create URL
#                 url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
                
#                 articles.append({
#                     "title": title.strip() if title else "No title available",
#                     "abstract": abstract,
#                     "url": url,
#                     "pmid": pmid
#                 })
                
#             except Exception as e:
#                 print(f"⚠️ Error parsing individual article: {e}")
#                 continue
    
#     except ET.ParseError as e:
#         print(f"❌ Error parsing XML: {e}")
#         return []
    
#     return articles

# # Test function - now with proper delays
# if __name__ == "__main__":
#     print("🧪 Testing PubMed retrieval with rate limiting...")
    
#     # Test with single query first
#     query = "What are the first-line treatments for type 2 diabetes?"
#     print(f"\n{'='*50}")
#     print(f"Testing: {query}")
#     print('='*50)
    
#     articles = get_pubmed_articles(query, max_results=3)
    
#     if articles:
#         print(f"✅ Retrieved {len(articles)} articles")
#         for i, article in enumerate(articles, 1):
#             print(f"\n📖 Article {i}:")
#             print(f"Title: {article['title']}")
#             print(f"PMID: {article['pmid']}")
#             print(f"Abstract: {article['abstract'][:200]}...")
#             print(f"URL: {article['url']}")
#     else:
#         print("❌ No articles retrieved")