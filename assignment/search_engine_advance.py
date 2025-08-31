#!/usr/bin/env python3
"""
Advanced Academic Search Engine
==============================
Author: Ram Sapkota
Description: High-performance search with BM25 + Cross-Encoder reranking and advanced filtering
"""

import math, re, time, hashlib
from collections import Counter, defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies
try:
    import nltk
    from nltk.corpus import stopwords, wordnet as wn
    from nltk.stem import PorterStemmer
    HAVE_NLTK = True
except:
    HAVE_NLTK = False

try:
    from sentence_transformers import CrossEncoder
    HAVE_SBERT = True
except:
    HAVE_SBERT = False

# Initialize NLTK
def init_nltk():
    if not HAVE_NLTK:
        return
    try:
        stopwords.words("english")
        nltk.word_tokenize("test")
        wn.synsets("test")
    except:
        for resource in ["stopwords", "punkt", "wordnet"]:
            nltk.download(resource, quiet=True)

if HAVE_NLTK:
    init_nltk()
    STOPS = set(stopwords.words("english"))
    STEMMER = PorterStemmer()
else:
    STOPS = {"a", "an", "the", "and", "or", "in", "of", "for", "to", "with", "on", "at", "by", "from", "as", "is", "are", "was", "were"}
    STEMMER = None

TOKEN_RE = re.compile(r"[a-z0-9]+")

def process_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> List[str]:
    """Extract and stem tokens"""
    text = process_text(text)
    tokens = TOKEN_RE.findall(text)
    filtered = [t for t in tokens if t not in STOPS and len(t) > 1]
    return [STEMMER.stem(t) for t in filtered] if STEMMER else filtered

def parse_authors(data: Any) -> List[Dict[str, Optional[str]]]:
    """Parse author data into standardized format"""
    result = []
    if not data:
        return result
    
    if isinstance(data, str):
        name = data.strip()
        if name:
            result.append({"name": name, "profile": None})
    elif isinstance(data, dict):
        name = str(data.get("name", "")).strip()
        profile = str(data.get("profile", "") or "").strip() or None
        if name:
            result.append({"name": name, "profile": profile})
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                profile = str(item.get("profile", "") or "").strip() or None
                if name:
                    result.append({"name": name, "profile": profile})
            elif isinstance(item, str):
                name = item.strip()
                if name:
                    result.append({"name": name, "profile": None})
    return result

def get_year(date_field: Any) -> Optional[int]:
    """Extract year from various date formats"""
    if not date_field:
        return None
    text = str(date_field)
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if match:
        year = int(match.group(0))
        return year if 1900 <= year <= 2100 else None
    return None

# Query parsing
QUOTE_RE = re.compile(r'"([^"]+)"')
YEAR_RE = re.compile(r"\byear:(\d{4})(?:\.\.(\d{4}))?\b", re.I)
AUTHOR_RE = re.compile(r'\bauthor:(".*?"|\S+)\b', re.I)
DOI_RE = re.compile(r'\bdoi:([^\s"]+)\b', re.I)

class QueryData:
    """Parsed search query"""
    def __init__(self, text: str, phrases: List[str], authors: List[str], 
                 year_from: Optional[int], year_to: Optional[int], doi: Optional[str]):
        self.text = text
        self.phrases = phrases
        self.authors = authors
        self.year_from = year_from
        self.year_to = year_to
        self.doi = doi

def parse_advanced_query(query: str) -> QueryData:
    """Parse complex query with filters"""
    query = " ".join((query or "").split())
    phrases = [m.group(1) for m in QUOTE_RE.finditer(query)]
    
    # Extract authors
    authors = []
    for m in AUTHOR_RE.finditer(query):
        author = m.group(1).strip('"').strip()
        if author:
            authors.append(author)
    query = AUTHOR_RE.sub(" ", query)
    
    # Extract DOI
    doi = None
    doi_match = DOI_RE.search(query)
    if doi_match:
        doi = doi_match.group(1).strip().lower()
        query = DOI_RE.sub(" ", query)
    
    # Extract year range
    year_from = year_to = None
    year_match = YEAR_RE.search(query)
    if year_match:
        year_from = int(year_match.group(1))
        year_to = int(year_match.group(2) or year_match.group(1))
        query = YEAR_RE.sub(" ", query)
    
    clean_text = QUOTE_RE.sub(" ", query).strip()
    return QueryData(clean_text, phrases, authors, year_from, year_to, doi)

# BM25 Implementation
class BM25Index:
    """Efficient BM25 scoring for a field"""
    def __init__(self, doc_tokens: List[List[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(doc_tokens)
        self.doc_lengths = [len(tokens) for tokens in doc_tokens]
        self.avg_len = sum(self.doc_lengths) / max(1, self.N)
        self.term_freqs = [Counter(tokens) for tokens in doc_tokens]
        
        # Calculate IDF
        doc_freqs = Counter(term for tf in self.term_freqs for term in tf.keys())
        self.idf_scores = {term: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0) 
                          for term, df in doc_freqs.items()}
        
        # Build inverted index
        self.inverted_index = defaultdict(list)
        for doc_id, tf in enumerate(self.term_freqs):
            for term, freq in tf.items():
                self.inverted_index[term].append((doc_id, freq))
    
    def score_query(self, query_terms: List[Tuple[str, float]]) -> Dict[int, float]:
        """Score documents for query terms"""
        scores = defaultdict(float)
        if not self.N:
            return scores
            
        for term, weight in query_terms:
            idf = self.idf_scores.get(term, 0.0)
            if idf <= 0:
                continue
                
            for doc_id, tf in self.inverted_index.get(term, []):
                doc_len = self.doc_lengths[doc_id]
                normalized_tf = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len))
                scores[doc_id] += weight * idf * normalized_tf
        return scores

# Cache for reranker
class SimpleCache:
    """LRU cache for reranking scores"""
    def __init__(self, max_size=4096):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[float]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: float):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

class AdvancedSearchEngine:
    """High-performance search with BM25 + Cross-Encoder"""
    
    # Field weights
    TITLE_WEIGHT = 2.8
    AUTHOR_WEIGHT = 1.6
    ABSTRACT_WEIGHT = 1.0
    
    # Score fusion
    BM25_ALPHA = 0.55
    RERANK_BETA = 0.45
    
    # Bonuses
    TITLE_EXACT_BONUS = 0.18
    TITLE_PARTIAL_BONUS = 0.10
    PHRASE_BONUS = 0.14
    AUTHOR_BONUS = 0.10
    DOI_BONUS = 0.30
    RECENCY_BONUS = 0.20

    def __init__(self, publications: List[Dict[str, Any]], enable_rerank=True, rerank_k=75, use_synonyms=True):
        # Process records
        self.records = []
        title_docs, author_docs, abstract_docs = [], [], []
        
        for pub in publications:
            title = str(pub.get("title", ""))
            abstract = str(pub.get("abstract", ""))
            authors = parse_authors(pub.get("authors"))
            
            record = {
                **pub,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": get_year(pub.get("date") or pub.get("published_date")),
                "_title_norm": process_text(title),
                "_authors_norm": process_text(" ".join(a["name"] for a in authors)),
                "_abstract_norm": process_text(abstract),
                "doi": str(pub.get("doi", "")).lower(),
                "link": pub.get("link") or pub.get("page_url") or ""
            }
            self.records.append(record)
            
            title_docs.append(tokenize(title))
            author_docs.append(tokenize(" ".join(a["name"] for a in authors)))
            abstract_docs.append(tokenize(abstract))
        
        # Build BM25 indices
        self.title_index = BM25Index(title_docs)
        self.author_index = BM25Index(author_docs)
        self.abstract_index = BM25Index(abstract_docs)
        
        # Initialize reranker
        self.use_rerank = enable_rerank and HAVE_SBERT
        self.rerank_k = max(1, rerank_k)
        self.reranker = None
        
        if self.use_rerank:
            try:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                print(f"Reranker load failed: {e}")
                self.use_rerank = False
        
        self.cache = SimpleCache(4096)
        self.use_synonyms = use_synonyms and HAVE_NLTK
        self.current_year = time.gmtime().tm_year

    def expand_query_terms(self, query: str) -> List[Tuple[str, float]]:
        """Expand query with synonyms"""
        base_terms = tokenize(query)
        if not self.use_synonyms:
            return [(t, 1.0) for t in base_terms]
        
        expanded = []
        seen = set()
        
        for term in base_terms:
            if term in seen:
                continue
            expanded.append((term, 1.0))
            seen.add(term)
            
            # Add synonyms
            try:
                synonyms = set()
                for synset in wn.synsets(term):
                    for lemma in synset.lemmas():
                        syn = lemma.name().replace("_", " ").lower()
                        if STEMMER:
                            syn = STEMMER.stem(syn)
                        if syn != term and syn.isalpha():
                            synonyms.add(syn)
                
                for syn in list(synonyms)[:2]:
                    if syn not in seen:
                        expanded.append((syn, 0.6))
                        seen.add(syn)
            except:
                pass
        
        return expanded

    def apply_filters(self, author=None, year=None, year_from=None, year_to=None, doi=None) -> set:
        """Get indices that pass filters"""
        all_indices = set(range(len(self.records)))
        if not any([author, year, year_from, year_to, doi]):
            return all_indices
        
        valid = all_indices
        if author:
            author_norm = process_text(str(author))
            valid &= {i for i, r in enumerate(self.records) 
                     if author_norm in r["_authors_norm"]}
        
        if year is not None:
            valid &= {i for i, r in enumerate(self.records) 
                     if r.get("year") == int(year)}
        
        if year_from is not None:
            valid &= {i for i, r in enumerate(self.records) 
                     if r.get("year") and r["year"] >= int(year_from)}
        
        if year_to is not None:
            valid &= {i for i, r in enumerate(self.records) 
                     if r.get("year") and r["year"] <= int(year_to)}
        
        if doi:
            doi_lower = str(doi).lower()
            valid &= {i for i, r in enumerate(self.records) 
                     if doi_lower in r["doi"]}
        
        return valid

    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """Min-max normalization"""
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 1e-9:
            return [0.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def recency_prior(self, year: Optional[int]) -> float:
        """Compute recency prior"""
        if year is None:
            return 0.5
        pivot = self.current_year - 3
        return 1.0 / (1.0 + math.exp(-(year - pivot) / 2.0))

    def execute_search(self, query: str, *, author=None, year=None, year_from=None, 
                      year_to=None, k=None, page=1, page_size=50) -> Dict[str, Any]:
        """Main search function with pagination"""
        query = (query or "").strip()
        parsed = parse_advanced_query(query)
        
        # Merge filters
        author_filter = author or (" ".join(parsed.authors) if parsed.authors else None)
        year_from_filter = year_from if year_from is not None else parsed.year_from
        year_to_filter = year_to if year_to is not None else parsed.year_to
        year_filter = int(year) if year is not None else None
        doi_filter = parsed.doi
        
        if not any([query, author_filter, year_filter, year_from_filter, year_to_filter, doi_filter]):
            return {"results": [], "total_results": 0}
        
        # BM25 retrieval
        query_text = " ".join([parsed.text] + parsed.phrases + parsed.authors).strip()
        query_terms = self.expand_query_terms(query_text)
        
        title_scores = self.title_index.score_query(query_terms)
        author_scores = self.author_index.score_query(query_terms)
        abstract_scores = self.abstract_index.score_query(query_terms)
        
        # Combine field scores
        combined = defaultdict(float)
        for i, s in title_scores.items(): combined[i] += self.TITLE_WEIGHT * s
        for i, s in author_scores.items(): combined[i] += self.AUTHOR_WEIGHT * s
        for i, s in abstract_scores.items(): combined[i] += self.ABSTRACT_WEIGHT * s
        
        # Apply filters
        valid_indices = self.apply_filters(author_filter, year_filter, year_from_filter, year_to_filter, doi_filter)
        candidates = [(i, float(s)) for i, s in combined.items() if i in valid_indices and s > 0]
        
        if not candidates:
            return {"results": [], "total_results": 0}
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        total_count = len(candidates)
        
        # Pagination setup
        page = max(1, int(page))
        page_size = max(1, int(page_size))
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        
        if start_idx >= total_count:
            return {"results": [], "total_results": total_count}
        
        # Rerank subset
        pool_size = max(self.rerank_k, end_idx)
        rerank_candidates = candidates[:pool_size]
        
        bm25_scores = [s for _, s in rerank_candidates]
        bm25_norm = self.normalize_scores(bm25_scores)
        
        # Cross-encoder reranking
        rerank_scores = [0.0] * len(rerank_candidates)
        if self.use_rerank and self.reranker and parsed.text:
            try:
                pairs = []
                for doc_idx, _ in rerank_candidates:
                    doc = self.records[doc_idx]
                    text = f"{doc['title']}. {doc['abstract']}".strip()
                    pairs.append([parsed.text, text])
                
                if pairs:
                    raw_scores = self.reranker.predict(pairs, show_progress_bar=False)
                    rerank_scores = self.normalize_scores([float(s) for s in raw_scores])
            except Exception as e:
                print(f"Reranking failed: {e}")
        
        # Final scoring with bonuses
        final_scores = []
        query_title_norm = process_text(parsed.text)
        phrase_list = [p.lower() for p in parsed.phrases]
        
        for idx, (doc_idx, _) in enumerate(rerank_candidates):
            record = self.records[doc_idx]
            bonus = 0.0
            
            # Title bonuses
            if query_title_norm:
                if record["_title_norm"] == query_title_norm:
                    bonus += self.TITLE_EXACT_BONUS
                elif query_title_norm in record["_title_norm"]:
                    bonus += self.TITLE_PARTIAL_BONUS
            
            # Phrase bonus
            for phrase in phrase_list:
                if phrase and (phrase in record["_title_norm"] or phrase in record["_abstract_norm"]):
                    bonus += min(self.PHRASE_BONUS, 0.06 + 0.02 * len(phrase.split()))
            
            # Author bonus
            if author_filter:
                author_norm = process_text(str(author_filter))
                if author_norm in record["_authors_norm"]:
                    bonus += self.AUTHOR_BONUS
            
            # DOI bonus
            if doi_filter and record["doi"] and doi_filter in record["doi"]:
                bonus += self.DOI_BONUS
            
            # Recency bonus
            recency = self.recency_prior(record.get("year"))
            bonus += self.RECENCY_BONUS * recency
            
            # Combine scores
            final_score = (self.BM25_ALPHA * bm25_norm[idx] + 
                          (self.RERANK_BETA * rerank_scores[idx] if self.use_rerank else 0.0) + 
                          bonus)
            final_scores.append((doc_idx, final_score))
        
        # Sort and paginate
        final_scores.sort(key=lambda x: x[1], reverse=True)
        page_results = final_scores[start_idx:end_idx]
        
        # Format output
        def highlight_text(text: str, terms: List[str]) -> str:
            if not text or not terms:
                return text or ""
            result = text
            for term in sorted(set(terms), key=len, reverse=True):
                if term:
                    try:
                        result = re.sub(f"(?i)({re.escape(term)})", r"<mark>\1</mark>", result)
                    except:
                        pass
            return result
        
        highlight_terms = []
        if parsed.text:
            highlight_terms.extend([w for w in parsed.text.split() if len(w) > 1])
        highlight_terms.extend(parsed.phrases)
        
        results = []
        for doc_idx, score in page_results:
            record = self.records[doc_idx]
            results.append({
                "title": record.get("title", ""),
                "link": record.get("link", ""),
                "authors": record.get("authors", []),
                "date": record.get("date", "") or record.get("published_date", ""),
                "abstract": highlight_text(record.get("abstract", ""), highlight_terms),
                "score": round(float(score), 3),
                "doi": record.get("doi", ""),
                "oa_url": record.get("oa_url", "") or record.get("pdf", "")
            })
        
        return {"results": results, "total_results": total_count}