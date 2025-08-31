#!/usr/bin/env python3
"""
Academic Search & Classification API
====================================
Author: Ram Sapkota
Description: FastAPI server combining publication search engines and news classification
"""

import sys, os, math, time, subprocess, traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import search engines
from search_engine import PublicationSearchEngine, read_publications
from search_engine_advance import AdvancedSearchEngine

# Import classifier (required for unpickling)
try:
    import train_classifier  # noqa: F401
    # Import the tokenizer class into current namespace for pickle compatibility
    from train_classifier import TextProcessor
    # Add compatibility aliases for old class names that might be in saved models
    LemmaTokenizer = TextProcessor
    CustomTokenizer = TextProcessor
    
    # Also add to train_classifier module for safety
    train_classifier.LemmaTokenizer = TextProcessor
    train_classifier.CustomTokenizer = TextProcessor
except:
    pass

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
MODEL_DIR = ROOT_DIR / "models"
MODEL_FILE = MODEL_DIR / "news_clf.joblib"

# Global state
publications_data = read_publications()
basic_search_engine = PublicationSearchEngine(publications_data)
advanced_search_engine = AdvancedSearchEngine(publications_data, enable_rerank=True, rerank_k=75, use_synonyms=True)

classifier_model = None
model_classes = None
model_metadata = {}
last_load_error = None

def ensure_nltk_resources() -> Optional[str]:
    """Ensure NLTK resources are available"""
    try:
        import nltk
        for resource in ("punkt", "wordnet", "omw-1.4", "stopwords"):
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        return None
    except Exception as e:
        return f"NLTK error: {type(e).__name__}: {e}"

def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax"""
    x = x.astype(np.float64)
    x -= np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum() if np.isfinite(exp_x.sum()) else np.ones_like(x) / len(x)

def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Convert decision function scores to probabilities"""
    scores = np.atleast_2d(scores)
    if scores.shape[1] == 1:
        scores = np.concatenate([-scores, scores], axis=1)
    return np.apply_along_axis(stable_softmax, 1, scores)

def predict_with_probabilities(model, texts: List[str]):
    """Get predictions with confidence scores"""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        preds = np.argmax(probs, axis=1)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        probs = scores_to_probabilities(scores)
        preds = np.argmax(probs, axis=1)
    else:
        preds = model.predict(texts)
        n_classes = len(getattr(model, "classes_", [])) or 3
        probs = np.full((len(preds), n_classes), 1.0 / n_classes)
    return preds, probs

def load_classifier_model():
    """Load classification model with error handling"""
    global classifier_model, model_classes, model_metadata, last_load_error
    
    if classifier_model is not None:
        return
    
    last_load_error = ensure_nltk_resources()
    if not MODEL_FILE.exists():
        last_load_error = f"Model file missing: {MODEL_FILE}"
        return
    
    try:
        # Fix pickle compatibility - create dummy classes in __main__ module
        import __main__
        import sys
        
        # Add the TextProcessor class to __main__ under old names for pickle compatibility
        if not hasattr(__main__, 'LemmaTokenizer'):
            __main__.LemmaTokenizer = train_classifier.TextProcessor
        if not hasattr(__main__, 'CustomTokenizer'):
            __main__.CustomTokenizer = train_classifier.TextProcessor
        if not hasattr(__main__, 'TextProcessor'):
            __main__.TextProcessor = train_classifier.TextProcessor
            
        # Also ensure train_classifier module has the old class names
        if not hasattr(train_classifier, 'LemmaTokenizer'):
            train_classifier.LemmaTokenizer = train_classifier.TextProcessor
        
        print('DEBUG: Pickle compatibility classes set up')
        print('DEBUG: About to call joblib.load()...')
        
        model_payload = joblib.load(MODEL_FILE)
        print('DEBUG: joblib.load() completed successfully')
        
        classifier_model = model_payload.get("pipeline")
        model_classes = list(model_payload.get("labels", []))
        model_metadata = model_payload.get("meta", {})
        
        if hasattr(classifier_model, "classes_"):
            model_classes = list(classifier_model.classes_)
        
        if not classifier_model:
            raise RuntimeError("No pipeline in model file")
        
        if not model_classes:
            model_classes = list(getattr(classifier_model, "classes_", [])) or ["politics", "business", "health"]
        
        print('DEBUG: Model loading completed successfully!')
        print(f'DEBUG: Model algorithm: {model_metadata.get("algo", "unknown")}')
        print(f'DEBUG: Model classes: {model_classes}')
            
    except Exception as e:
        print(f'DEBUG: Exception occurred: {type(e).__name__}: {e}')
        last_load_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        classifier_model = None
        model_classes = None
        model_metadata = {}

# Initialize FastAPI
app = FastAPI(
    title="Academic Search & Classification API",
    description="Search publications and classify news articles - by Ram Sapkota",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Static file serving
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8"))

@app.get("/styles.css")
def serve_css():
    css_file = FRONTEND_DIR / "styles.css"
    return FileResponse(css_file, media_type="text/css") if css_file.exists() else JSONResponse({"error": "CSS not found"}, 404)

@app.get("/script.js")
def serve_js():
    js_file = FRONTEND_DIR / "script.js"
    return FileResponse(js_file, media_type="application/javascript") if js_file.exists() else JSONResponse({"error": "JS not found"}, 404)

# Health and status endpoints
@app.get("/healthz")
def health_check():
    load_classifier_model()
    return {
        "status": "healthy",
        "publications_count": len(publications_data),
        "model_loaded": classifier_model is not None,
        "model_error": last_load_error,
        "model_algorithm": model_metadata.get("algo"),
        "model_classes": model_classes
    }

@app.get("/model_info")
def get_model_info():
    load_classifier_model()
    if classifier_model is None:
        return JSONResponse({"error": last_load_error or "Model unavailable"}, 503)
    
    return {
        "loaded": True,
        "algorithm": model_metadata.get("algo"),
        "classes": model_classes,
        "parameters": model_metadata.get("params", {})
    }

# Publication endpoints
@app.get("/publications/")
def list_publications(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1), engine: str = Query("bm25")):
    start = (page - 1) * page_size
    end = start + page_size
    total = len(publications_data)
    
    return {
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size),
        "total_publications": total,
        "publications": publications_data[start:end],
        "engine": engine.lower()
    }

@app.get("/search/")
def search_publications(
    q: Optional[str] = Query(None, min_length=1),
    query: Optional[str] = Query(None, min_length=1),
    engine: str = Query("bm25"),
    author: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    year_from: Optional[int] = Query(None),
    year_to: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    search_query = (q or query or "").strip()
    has_filters = any([search_query, author, year, year_from, year_to])
    
    if not has_filters:
        return {
            "query": "",
            "engine": engine.lower(),
            "reranker_active": False,
            "results_count": 0,
            "total_results": 0,
            "total_pages": 0,
            "page": page,
            "page_size": page_size,
            "search_time_ms": 0,
            "results": []
        }
    
    start_time = time.perf_counter()
    engine_type = (engine or "bm25").strip().lower()
    
    if engine_type == "tfidf":
        # Basic TF-IDF search
        search_results = basic_search_engine.perform_search(search_query, max_results=10000)
        total_results = len(search_results)
        start_idx = (page - 1) * page_size
        end_idx = min(page * page_size, total_results)
        results = search_results[start_idx:end_idx]
        reranker_active = False
    else:
        # Advanced BM25 search
        search_response = advanced_search_engine.execute_search(
            search_query,
            author=author,
            year=year,
            year_from=year_from,
            year_to=year_to,
            page=page,
            page_size=page_size
        )
        results = search_response["results"]
        total_results = search_response["total_results"]
        reranker_active = (
            getattr(advanced_search_engine, "use_rerank", False) and 
            getattr(advanced_search_engine, "reranker", None) is not None
        )
    
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    
    return {
        "query": search_query,
        "engine": engine_type,
        "reranker_active": reranker_active,
        "results_count": len(results),
        "total_results": total_results,
        "total_pages": math.ceil(total_results / page_size) if total_results else 0,
        "page": page,
        "page_size": page_size,
        "search_time_ms": elapsed_ms,
        "results": results
    }

# Classification endpoints
class ClassificationRequest(BaseModel):
    text: str

@app.post("/classify")
def classify_text(request: ClassificationRequest):
    load_classifier_model()
    print('came here    ')
    if classifier_model is None:
        print('came here as none   ') 
        print(last_load_error) 
        error_msg = last_load_error or "Classification model unavailable"
        return JSONResponse({"error": error_msg, "loaded": False}, 503)
    
    input_text = (request.text or "").strip()
    if not input_text:
        return {
            "error": "Empty input text",
            "loaded": False,
            "label": None,
            "probabilities": {}
        }
    
    try:
        predictions, probabilities = predict_with_probabilities(classifier_model, [input_text])
        classes = model_classes or [f"class_{i}" for i in range(probabilities.shape[1])]
        
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities[0])}
        predicted_label = classes[int(predictions[0])]
        
        return {
            "loaded": True,
            "label": predicted_label,
            "probabilities": prob_dict,
            "error": None
        }
        
    except Exception as e:
        error_details = f"Classification failed: {type(e).__name__}: {e}"
        tb_lines = traceback.format_exc().splitlines()[-3:]
        
        return JSONResponse({
            "error": error_details,
            "traceback": tb_lines,
            "loaded": False
        }, 500)

@app.post("/retrain")
def retrain_model():
    """Retrain classification model"""
    try:
        training_command = [
            sys.executable, 
            str(ROOT_DIR / "train_classifier.py"),
            "--model", "all",
            "--use_lemmatization",
            "--ngram_max", "3"
        ]
        
        process = subprocess.run(training_command, capture_output=True, text=True)
        
        # Reset global model state
        global classifier_model, model_classes, model_metadata, last_load_error
        classifier_model = None
        model_classes = None
        model_metadata = {}
        last_load_error = None
        
        return {
            "success": process.returncode == 0,
            "return_code": process.returncode,
            "stdout": process.stdout[-4000:],  # Last 4K chars
            "stderr": process.stderr[-4000:]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)