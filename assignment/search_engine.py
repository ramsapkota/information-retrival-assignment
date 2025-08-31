#!/usr/bin/env python3
"""
Academic Publication Search Engine
=================================
Author: Ram Sapkota
Description: Compact TF-IDF search engine for academic publications with fielded filters
"""

import json, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional NLTK
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    HAVE_NLTK = True
except:
    HAVE_NLTK = False

def setup_nltk() -> bool:
    """Initialize NLTK resources"""
    if not HAVE_NLTK:
        return False
    try:
        stopwords.words("english")
        nltk.word_tokenize("test")
        return True
    except:
        try:
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)
            return True
        except:
            return False

NLTK_READY = setup_nltk()
STEMMER = PorterStemmer() if NLTK_READY else None
STOPWORDS = set(stopwords.words("english")) if NLTK_READY else {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "for", "on", 
    "with", "by", "from", "as", "at", "is", "are", "was", "were", "be", "it",
    "this", "that", "these", "i", "you", "he", "she", "we", "they"
}

def read_publications(primary="data/publications.json", fallback="data/publications_detailed.json") -> List[Dict]:
    """Load publication data with fallback"""
    try:
        with open(primary, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except FileNotFoundError:
        with open(fallback, "r", encoding="utf-8") as f:
            return json.load(f) or []

def clean_text(text: str) -> str:
    """Basic text normalization"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    filtered = [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]
    if STEMMER:
        filtered = [STEMMER.stem(t) for t in filtered]
    return " ".join(filtered)

def extract_authors(auth_data: Any) -> List[str]:
    """Extract author names from varied JSON structures"""
    if not auth_data:
        return []
    if isinstance(auth_data, str):
        return [auth_data.strip()] if auth_data.strip() else []
    if isinstance(auth_data, dict):
        name = str(auth_data.get("name", "")).strip()
        return [name] if name else []
    if isinstance(auth_data, list):
        names = []
        for item in auth_data:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    names.append(name)
            elif isinstance(item, str) and item.strip():
                names.append(item.strip())
        return names
    return []

@dataclass
class PublicationRecord:
    """Normalized publication data"""
    raw: Dict[str, Any]
    title: str
    abstract: str  
    authors_raw: Any
    author_names: List[str]
    date: str
    url: str

def normalize_publication(data: Dict[str, Any]) -> PublicationRecord:
    """Convert raw JSON to normalized record"""
    title = str(data.get("title", "")).strip()
    abstract = str(data.get("abstract") or data.get("summary", "")).strip()
    date = str(data.get("date") or data.get("published_date") or data.get("year", "")).strip()
    authors_raw = data.get("authors", [])
    author_names = extract_authors(authors_raw)
    url = (data.get("link") or data.get("page_url") or data.get("url", "")).strip()
    
    return PublicationRecord(data, title, abstract, authors_raw, author_names, date, url)

# Query parsing
AUTHOR_PATTERN = re.compile(r'author:\s*"(.*?)"|author:\s*([^\s]+)', re.I)
YEAR_PATTERN = re.compile(r'year:\s*(\d{4})(?:\.\.(\d{4}))?', re.I)

@dataclass  
class SearchFilters:
    author: Optional[str] = None
    year_start: Optional[int] = None  
    year_end: Optional[int] = None

def parse_search_query(query: str) -> Tuple[str, SearchFilters]:
    """Extract filters from query string"""
    filters = SearchFilters()
    
    # Extract author filter
    def replace_author(match):
        filters.author = (match.group(1) or match.group(2) or "").strip().lower()
        return " "
    query = AUTHOR_PATTERN.sub(replace_author, query)
    
    # Extract year filter  
    def replace_year(match):
        y1, y2 = match.group(1), match.group(2)
        if y1:
            filters.year_start = filters.year_end = int(y1)
        if y2:
            filters.year_start = min(int(y1), int(y2))
            filters.year_end = max(int(y1), int(y2))
        return " "
    query = YEAR_PATTERN.sub(replace_year, query)
    
    return query.strip(), filters

class PublicationSearchEngine:
    """Compact TF-IDF based search engine"""
    
    def __init__(self, publications: List[Dict[str, Any]]):
        # Normalize records
        self.records = [normalize_publication(pub) for pub in publications]
        
        # Build searchable text blobs
        search_texts = []
        for record in self.records:
            title_clean = clean_text(record.title)
            authors_clean = clean_text(" ".join(record.author_names))
            abstract_clean = clean_text(record.abstract)
            search_texts.append(f"{title_clean} {authors_clean} {abstract_clean}".strip())
        
        # Create TF-IDF index
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            norm="l2"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(search_texts)
    
    def extract_year(self, record: PublicationRecord) -> Optional[int]:
        """Extract 4-digit year from record"""
        # Check raw data for explicit year
        for key in ("year", "published_year"):
            value = record.raw.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and re.match(r"^\d{4}$", value.strip()):
                return int(value)
        
        # Search date string
        year_match = re.search(r"\b(19|20)\d{2}\b", record.date or "")
        return int(year_match.group(0)) if year_match else None
    
    def passes_filters(self, record: PublicationRecord, filters: SearchFilters) -> bool:
        """Check if record passes search filters"""
        # Author filter
        if filters.author:
            author_text = " ".join(record.author_names).lower()
            if filters.author not in author_text:
                return False
        
        # Year filters
        if filters.year_start or filters.year_end:
            year = self.extract_year(record)
            if year is None:
                return False
            if filters.year_start and year < filters.year_start:
                return False  
            if filters.year_end and year > filters.year_end:
                return False
        
        return True
    
    def perform_search(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Execute search with optional filters"""
        if not query or not query.strip():
            return []
        
        # Parse query and filters
        clean_query, filters = parse_search_query(query)
        clean_query = clean_query if clean_query else "*"
        
        # Compute similarities
        query_vector = self.vectorizer.transform([clean_text(clean_query)])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top candidates (larger pool for filtering)
        pool_size = min(len(self.records), max(max_results * 5, max_results))
        top_indices = similarities.argsort()[-pool_size:][::-1]
        
        # Filter and format results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.01:  # Skip low-relevance results
                continue
                
            record = self.records[idx]
            if not self.passes_filters(record, filters):
                continue
            
            # Format result
            year = self.extract_year(record)
            result = {
                "title": record.title,
                "link": record.url,
                "authors": record.authors_raw,  # Preserve original format
                "date": record.date or (str(year) if year else ""),
                "abstract": record.abstract,
                "score": round(score, 3)
            }
            results.append(result)
            
            if len(results) >= max_results:
                break
        
        # Sort by score and recency
        results.sort(key=lambda x: (x["score"], extract_year_safely(x.get("date"))), reverse=True)
        return results

def extract_year_safely(date_str: Any) -> int:
    """Safely extract year for sorting"""
    if isinstance(date_str, int):
        return date_str
    if isinstance(date_str, str):
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        return int(match.group(0)) if match else -1
    return -1

# Standalone usage
if __name__ == "__main__":
    pubs = read_publications()
    engine = PublicationSearchEngine(pubs)
    
    test_query = 'inflation author:"smith" year:2020..2024'
    for result in engine.perform_search(test_query, max_results=10):
        print(f"{result['score']:.3f} | {result['date']} | {result['title'][:80]}...")