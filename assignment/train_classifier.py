#!/usr/bin/env python3
"""
Text Classification Training System
===================================
Author: Ram Sapkota
Description: Multi-algorithm news classifier with NLTK preprocessing and cross-validation
"""

import os, glob, json, argparse, random, re
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib

# Configuration
CONFIG = {
    'data_dir': Path("data/classification"),
    'model_dir': Path("models"),
    'model_path': Path("models/news_clf.joblib"),
    'summary_path': Path("models/news_clf_summary.json"),
    'categories': ["politics", "business", "health"],
    'random_seed': 42,
    'word_regex': re.compile(r"[A-Za-z]+")
}

# NLTK Integration
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_READY = True
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except:
        STOP_WORDS = set()
    LEMMATIZER = WordNetLemmatizer()
except:
    NLTK_READY = False
    STOP_WORDS = set()
    LEMMATIZER = None

class TextProcessor:
    """Compact tokenizer with lemmatization support"""
    
    def __init__(self, lowercase=True, remove_stops=True, use_alpha=True):
        self.lowercase = lowercase
        self.remove_stops = remove_stops
        self.use_alpha = use_alpha

    def __call__(self, text: str) -> List[str]:
        if not NLTK_READY:
            return self._basic_tokenize(text)
        
        text = text.lower() if self.lowercase else text
        tokens = CONFIG['word_regex'].findall(text) if self.use_alpha else nltk.word_tokenize(text)
        
        result = []
        for token in tokens:
            if self.remove_stops and token in STOP_WORDS:
                continue
            processed = LEMMATIZER.lemmatize(token) if LEMMATIZER else token
            if processed:
                result.append(processed)
        return result

    def _basic_tokenize(self, text: str) -> List[str]:
        text = text.lower() if self.lowercase else text
        return CONFIG['word_regex'].findall(text)

def load_text_data() -> Tuple[List[str], List[str]]:
    """Load documents and labels from category directories"""
    texts, labels = [], []
    for category in CONFIG['categories']:
        folder = CONFIG['data_dir'] / category
        if not folder.exists():
            continue
        for file_path in glob.glob(str(folder / "*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if content:
                    texts.append(content)
                    labels.append(category)
            except Exception as e:
                print(f"Warning: {file_path} failed: {e}")
    return texts, labels

def create_model_pipeline(algo: str, min_df=2, max_features=80000, ngram_max=2, 
                         use_lemma=False, nb_alpha=0.3, lr_C=2.0, svm_C=1.0) -> Pipeline:
    """Create ML pipeline for given algorithm"""
    tokenizer = TextProcessor() if use_lemma else None
    
    vec_params = {
        'lowercase': True, 'strip_accents': 'unicode', 'analyzer': 'word',
        'ngram_range': (1, ngram_max), 'min_df': min_df, 'max_features': max_features,
        'sublinear_tf': True, 'dtype': np.float32
    }
    
    if tokenizer:
        vec_params.update({'tokenizer': tokenizer, 'token_pattern': None})
    else:
        vec_params.update({'stop_words': 'english'})

    vectorizer = TfidfVectorizer(**vec_params)

    classifiers = {
        'nb': MultinomialNB(alpha=nb_alpha),
        'lr': LogisticRegression(solver='saga', max_iter=2000, n_jobs=-1, C=lr_C),
        'svm': LinearSVC(C=svm_C)
    }
    
    if algo not in classifiers:
        raise ValueError(f"Unknown algorithm: {algo}")
        
    return Pipeline([('tfidf', vectorizer), ('clf', classifiers[algo])])

def run_cross_validation(pipeline: Pipeline, X: List[str], y: List[str], folds=5) -> Dict:
    """Execute stratified k-fold cross validation"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=CONFIG['random_seed'])
    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=['f1_macro', 'accuracy'], 
                               n_jobs=-1, return_train_score=False)
    return {
        'cv_f1_mean': float(np.mean(cv_results['test_f1_macro'])),
        'cv_f1_std': float(np.std(cv_results['test_f1_macro'])),
        'cv_acc_mean': float(np.mean(cv_results['test_accuracy'])),
        'cv_acc_std': float(np.std(cv_results['test_accuracy']))
    }

def evaluate_test_set(pipeline: Pipeline, X_test: List[str], y_test: List[str]) -> Dict:
    """Evaluate on holdout test set"""
    predictions = pipeline.predict(X_test)
    return {
        'test_accuracy': float(accuracy_score(y_test, predictions)),
        'test_f1_macro': float(f1_score(y_test, predictions, average='macro')),
        'classification_report': classification_report(y_test, predictions, 
                                                     labels=CONFIG['categories'], 
                                                     zero_division=0, digits=4),
        'confusion_matrix_labels': CONFIG['categories'],
        'confusion_matrix': confusion_matrix(y_test, predictions, 
                                           labels=CONFIG['categories']).tolist()
    }

def train_single_model(algo: str, X_train: List[str], y_train: List[str], 
                      X_test: List[str], y_test: List[str], folds: int, **kwargs):
    """Train and evaluate single algorithm"""
    pipeline = create_model_pipeline(algo, **kwargs)
    
    print(f"\n[CV] {algo.upper()} cross-validation...")
    cv_stats = run_cross_validation(pipeline, X_train, y_train, folds)
    print(f"  F1: {cv_stats['cv_f1_mean']:.4f}±{cv_stats['cv_f1_std']:.4f} | "
          f"Acc: {cv_stats['cv_acc_mean']:.4f}±{cv_stats['cv_acc_std']:.4f}")

    pipeline.fit(X_train, y_train)
    test_results = evaluate_test_set(pipeline, X_test, y_test)
    
    print(f"\n[TEST] Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"[TEST] F1-macro: {test_results['test_f1_macro']:.4f}")
    print(test_results['classification_report'])
    
    return pipeline, cv_stats, test_results

def save_model_artifacts(best_pipeline, best_algo, cv_stats, test_results, params, data_info):
    """Save model and metadata"""
    CONFIG['model_dir'].mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_data = {
        'pipeline': best_pipeline,
        'labels': CONFIG['categories'],
        'meta': {
            'algo': best_algo,
            'params': params,
            'cv': cv_stats,
            'heldout': {'test_accuracy': test_results['test_accuracy'], 
                       'test_f1_macro': test_results['test_f1_macro']}
        }
    }
    joblib.dump(model_data, CONFIG['model_path'])
    print(f"\n[SAVED] {CONFIG['model_path']}")

    # Save summary
    summary = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'labels': CONFIG['categories'],
        'class_counts': data_info['counts'],
        'split': data_info['split'],
        'best_model': best_algo,
        'cv_best': cv_stats,
        'heldout': test_results,
        'notes': {'vectorizer': 'TF-IDF', 'preprocessing': 'NLTK lemmatization (optional)'}
    }
    
    with open(CONFIG['summary_path'], 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {CONFIG['summary_path']}")

def execute_training(test_size, folds, model_choice, min_df, max_features, ngram_max, 
                    use_lemma, alpha, lr_C, svm_C):
    """Main training execution"""
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    # Load data
    X, y = load_text_data()
    n_docs = len(X)
    assert n_docs >= 100, f"Need ≥100 documents, got {n_docs}"
    
    counts = {cat: y.count(cat) for cat in CONFIG['categories']}
    print(f"[DATA] {n_docs} documents: {counts}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=CONFIG['random_seed'])
    print(f"[SPLIT] Train={len(X_train)} Test={len(X_test)}")

    # Train models
    algos = [model_choice] if model_choice in ['nb', 'lr', 'svm'] else ['nb', 'lr', 'svm']
    
    results = {}
    best = {'algo': None, 'pipeline': None, 'cv': None, 'test': None, 'f1': -1}
    
    params = {
        'min_df': min_df, 'max_features': max_features, 'ngram_max': ngram_max,
        'use_lemma': use_lemma, 'nb_alpha': alpha, 'lr_C': lr_C, 'svm_C': svm_C
    }

    for algo in algos:
        pipeline, cv_stats, test_results = train_single_model(
            algo, X_train, y_train, X_test, y_test, folds, **params)
        
        f1_score = test_results['test_f1_macro']
        results[algo] = {'cv': cv_stats, 'test': {'f1_macro': f1_score, 
                                                 'accuracy': test_results['test_accuracy']}}
        
        if f1_score > best['f1']:
            best.update({'algo': algo, 'pipeline': pipeline, 'cv': cv_stats, 
                        'test': test_results, 'f1': f1_score})

    # Save best model
    data_info = {
        'counts': counts,
        'split': {'train': len(X_train), 'test': len(X_test), 'test_size': test_size}
    }
    save_model_artifacts(best['pipeline'], best['algo'], best['cv'], best['test'], params, data_info)

def main():
    parser = argparse.ArgumentParser(description="News Classification Training")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", choices=["nb","lr","svm","all"], default="all")
    parser.add_argument("--min_df", type=int, default=5)
    parser.add_argument("--max_features", type=int, default=120000)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--use_lemmatization", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--lr_C", type=float, default=2.0)
    parser.add_argument("--svm_C", type=float, default=1.0)
    
    args = parser.parse_args()
    
    execute_training(args.test_size, args.folds, args.model, args.min_df, 
                    args.max_features, args.ngram_max, args.use_lemmatization,
                    args.alpha, args.lr_C, args.svm_C)

if __name__ == "__main__":
    main()