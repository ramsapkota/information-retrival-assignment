#!/usr/bin/env python3
"""
Academic Publications Web Scraper
=================================
Author: Ram Sapkota
Description: Selenium-based scraper for Coventry University Pure Portal publications
"""

import argparse, json, os, time, re, unicodedata, difflib
from math import ceil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PORTAL_BASE = "https://pureportal.coventry.ac.uk"
PERSONS_PATH = "/en/persons/"
PUBLICATIONS_URL = f"{PORTAL_BASE}/en/organisations/fbl-school-of-economics-finance-and-accounting/publications/"

# Patterns
FIRST_DIGIT_RE = re.compile(r"\d")
NAME_FORMAT_RE = re.compile(r"[A-Z][A-Za-z'—\-]+,\s*(?:[A-Z](?:\.)?)(?:\s*[A-Z](?:\.)?)*", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

def create_chrome_driver(headless=True, legacy=False) -> webdriver.Chrome:
    """Create configured Chrome WebDriver"""
    options = Options()
    if headless:
        options.add_argument("--headless" + ("" if legacy else "=new"))
    
    # Performance options
    for arg in [
        "--window-size=1366,900", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
        "--lang=en-US", "--disable-notifications", "--no-first-run", "--disable-extensions",
        "--disable-popup-blocking", "--disable-renderer-backgrounding",
        "--disable-features=CalculateNativeWinOcclusion,MojoVideoDecoder",
        "--disable-blink-features=AutomationControlled"
    ]:
        options.add_argument(arg)
    
    options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.page_load_strategy = "eager"
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    service = ChromeService(ChromeDriverManager().install(), log_output=os.devnull)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(40)
    
    # Hide webdriver property
    try:
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
    except:
        pass
    
    return driver

def handle_cookie_consent(driver: webdriver.Chrome):
    """Accept cookies if consent banner appears"""
    try:
        cookie_btn = WebDriverWait(driver, 6).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        driver.execute_script("arguments[0].click();", cookie_btn)
        time.sleep(0.2)
    except TimeoutException:
        pass

def normalize_text(text: str) -> str:
    """Normalize Unicode text"""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s\-']", " ", text, flags=re.UNICODE).strip().lower()
    return WHITESPACE_RE.sub(" ", text)

def is_person_url(url: str) -> bool:
    """Check if URL is a valid person profile"""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        if parsed.netloc and "coventry.ac.uk" not in parsed.netloc:
            return False
        path = parsed.path.rstrip("/")
        if not path.startswith(PERSONS_PATH):
            return False
        slug = path[len(PERSONS_PATH):].strip("/")
        return bool(slug and not slug.startswith("?"))
    except:
        return False

def is_person_name(text: str) -> bool:
    """Check if text looks like a person's name"""
    if not text:
        return False
    text = text.strip()
    excluded = {"profiles", "persons", "people", "overview"}
    if text.lower() in excluded:
        return False
    return ((" " in text) or ("," in text)) and sum(ch.isalpha() for ch in text) >= 4

def unique_strings(items: List[str]) -> List[str]:
    """Remove duplicate strings while preserving order"""
    seen, result = set(), []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result

def unique_authors(authors: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
    """Remove duplicate author objects"""
    seen = set()
    result = []
    for author in authors:
        name = (author.get("name") or "").strip()
        profile = (author.get("profile") or "").strip()
        key = (name, profile)
        if name and key not in seen:
            seen.add(key)
            result.append({"name": name, "profile": profile or None})
    return result

class ListingScraper:
    """Scraper for publication listing pages"""
    
    def __init__(self, headless=False, legacy=False):
        self.driver = create_chrome_driver(headless, legacy)
    
    def scrape_page(self, page_num: int) -> List[Dict]:
        """Scrape single listing page"""
        url = f"{PUBLICATIONS_URL}?page={page_num}"
        self.driver.get(url)
        handle_cookie_consent(self.driver)
        
        try:
            WebDriverWait(self.driver, 15).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, ".result-container h3.title a") or 
                         "No results" in d.page_source
            )
        except TimeoutException:
            pass
        
        publications = []
        for container in self.driver.find_elements(By.CLASS_NAME, "result-container"):
            try:
                link_elem = container.find_element(By.CSS_SELECTOR, "h3.title a")
                title = link_elem.text.strip()
                url = link_elem.get_attribute("href")
                if title and url:
                    publications.append({"title": title, "link": url})
            except:
                continue
        
        return publications
    
    def scrape_all_pages(self, max_pages: int) -> List[Dict]:
        """Scrape all listing pages"""
        try:
            self.driver.get(PUBLICATIONS_URL)
            handle_cookie_consent(self.driver)
            
            all_publications = []
            for page in range(max_pages):
                print(f"[LISTING] Page {page+1}/{max_pages}")
                page_pubs = self.scrape_page(page)
                if not page_pubs:
                    print(f"[LISTING] No results at page {page}, stopping")
                    break
                all_publications.extend(page_pubs)
            
            # Remove duplicates
            unique_pubs = {}
            for pub in all_publications:
                unique_pubs[pub["link"]] = pub
            
            return list(unique_pubs.values())
        finally:
            try:
                self.driver.quit()
            except:
                pass

class DetailScraper:
    """Scraper for individual publication detail pages"""
    
    def __init__(self, headless=True, legacy=False):
        self.driver = create_chrome_driver(headless, legacy)
    
    def expand_author_lists(self):
        """Click 'show more' buttons to reveal all authors"""
        try:
            for btn in self.driver.find_elements(By.XPATH, 
                "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'show') or "
                "contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'more')]")[:2]:
                try:
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                    time.sleep(0.1)
                    btn.click()
                    time.sleep(0.2)
                except:
                    continue
        except:
            pass
    
    def extract_authors_from_links(self) -> List[Dict]:
        """Extract authors from person profile links in header"""
        # Find navigation tabs to establish boundary
        tabs_y = None
        for xpath in [
            "//a[normalize-space()='Overview']",
            "//nav[contains(@class,'tabbed-navigation')]",
            "//div[contains(@class,'navigation') and .//a[contains(.,'Overview')]]"
        ]:
            try:
                tab_elem = self.driver.find_element(By.XPATH, xpath)
                tabs_y = tab_elem.location.get("y")
                if tabs_y:
                    break
            except:
                continue
        
        tabs_y = tabs_y or 900  # Conservative fallback
        
        authors = []
        seen = set()
        
        for link in self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/en/persons/']"):
            try:
                y_pos = link.location.get("y", 99999)
                if y_pos >= tabs_y:  # Skip links below navigation
                    continue
                
                href = (link.get_attribute("href") or "").strip()
                if not is_person_url(href):
                    continue
                
                # Get author name
                try:
                    name = link.find_element(By.CSS_SELECTOR, "span").text.strip()
                except NoSuchElementException:
                    name = (link.text or "").strip()
                
                if not is_person_name(name):
                    continue
                
                key = (name, href)
                if key not in seen:
                    seen.add(key)
                    authors.append({
                        "name": name, 
                        "profile": urljoin(self.driver.current_url, href)
                    })
            except:
                continue
        
        return unique_authors(authors)
    
    def get_meta_content(self, names: List[str]) -> List[str]:
        """Extract content from meta tags"""
        values = []
        for name in names:
            for elem in self.driver.find_elements(By.CSS_SELECTOR, f'meta[name="{name}"], meta[property="{name}"]'):
                content = (elem.get_attribute("content") or "").strip()
                if content:
                    values.append(content)
        return unique_strings(values)
    
    def extract_authors_from_jsonld(self) -> List[str]:
        """Extract authors from JSON-LD structured data"""
        import json as _json
        authors = []
        
        for script in self.driver.find_elements(By.CSS_SELECTOR, 'script[type="application/ld+json"]'):
            content = (script.get_attribute("textContent") or "").strip()
            if not content:
                continue
            
            try:
                data = _json.loads(content)
                objects = data if isinstance(data, list) else [data]
                
                for obj in objects:
                    author_data = obj.get("author")
                    if not author_data:
                        continue
                    
                    if isinstance(author_data, list):
                        for author in author_data:
                            if isinstance(author, dict):
                                name = author.get("name")
                                if name:
                                    authors.append(name)
                            elif isinstance(author, str):
                                authors.append(author)
                    elif isinstance(author_data, dict):
                        name = author_data.get("name")
                        if name:
                            authors.append(name)
                    elif isinstance(author_data, str):
                        authors.append(author_data)
            except:
                continue
        
        return unique_strings(authors)
    
    def extract_authors_from_subtitle(self, title: str) -> List[str]:
        """Extract authors from subtitle line using pattern matching"""
        try:
            date_elem = self.driver.find_element(By.CSS_SELECTOR, "span.date")
        except NoSuchElementException:
            return []
        
        try:
            subtitle_elem = date_elem.find_element(By.XPATH, "ancestor::*[contains(@class,'subtitle')][1]")
        except:
            try:
                subtitle_elem = date_elem.find_element(By.XPATH, "..")
            except:
                return []
        
        subtitle_text = subtitle_elem.text if subtitle_elem else ""
        
        # Remove title from subtitle
        if title and title in subtitle_text:
            subtitle_text = subtitle_text.replace(title, "")
        
        subtitle_text = " ".join(subtitle_text.split()).strip()
        
        # Extract text before first digit (usually publication date)
        digit_match = FIRST_DIGIT_RE.search(subtitle_text)
        pre_date = subtitle_text[:digit_match.start()].strip(" -—–·•,;|") if digit_match else subtitle_text
        
        # Normalize conjunctions
        pre_date = pre_date.replace(" & ", ", ").replace(" and ", ", ")
        
        # Extract "Surname, Initials" patterns
        author_patterns = NAME_FORMAT_RE.findall(pre_date)
        return unique_strings(author_patterns)
    
    def wrap_names_as_objects(self, names: List[str]) -> List[Dict]:
        """Convert name strings to author objects"""
        return unique_authors([{"name": name, "profile": None} for name in names])
    
    def scrape_publication_detail(self, url: str, title_hint: str) -> Dict:
        """Scrape detailed information from publication page"""
        self.driver.get(url)
        handle_cookie_consent(self.driver)
        
        try:
            WebDriverWait(self.driver, 18).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
        except TimeoutException:
            pass
        
        # Extract title
        try:
            title = self.driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
        except NoSuchElementException:
            title = title_hint or ""
        
        self.expand_author_lists()
        
        # Extract authors using multiple strategies
        author_objects = self.extract_authors_from_links()
        # Filter out any navigation links that may have slipped through
        author_objects = [
            author for author in author_objects 
            if is_person_name(author.get("name", "")) and is_person_url(author.get("profile", ""))
        ]
        
        if not author_objects:
            names = self.extract_authors_from_subtitle(title)
            author_objects = self.wrap_names_as_objects(names)
        
        if not author_objects:
            names = self.get_meta_content(["citation_author", "dc.contributor", "dc.contributor.author"])
            author_objects = self.wrap_names_as_objects(names)
        
        if not author_objects:
            names = self.extract_authors_from_jsonld()
            author_objects = self.wrap_names_as_objects(names)
        
        # Extract publication date
        pub_date = None
        for selector in ["span.date", "time[datetime]", "time"]:
            try:
                date_elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                pub_date = date_elem.get_attribute("datetime") or date_elem.text.strip()
                if pub_date:
                    break
            except NoSuchElementException:
                continue
        
        if not pub_date:
            meta_dates = self.get_meta_content(["citation_publication_date", "dc.date", "article:published_time"])
            if meta_dates:
                pub_date = meta_dates[0]
        
        # Extract abstract
        abstract = None
        for selector in [
            "section#abstract .textblock", "section.abstract .textblock", "div.abstract .textblock",
            "div#abstract", "section#abstract", "div.textblock"
        ]:
            try:
                abstract_elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                text = abstract_elem.text.strip()
                if text and len(text) > 15:
                    abstract = text
                    break
            except NoSuchElementException:
                continue
        
        # Fallback abstract extraction
        if not abstract:
            try:
                for heading in self.driver.find_elements(By.CSS_SELECTOR, "h2, h3"):
                    if "abstract" in heading.text.strip().lower():
                        next_elem = heading.find_element(By.XPATH, "./following::*[self::div or self::p or self::section][1]")
                        text = next_elem.text.strip()
                        if text:
                            abstract = text
                            break
            except:
                pass
        
        return {
            "title": title,
            "link": url,
            "authors": unique_authors(author_objects),
            "published_date": pub_date,
            "abstract": abstract or ""
        }
    
    def close(self):
        """Close the browser"""
        try:
            self.driver.quit()
        except:
            pass

def process_detail_batch(publications: List[Dict], headless=True, legacy=False) -> List[Dict]:
    """Process a batch of publications for detailed scraping"""
    scraper = DetailScraper(headless, legacy)
    results = []
    
    try:
        for i, pub in enumerate(publications, 1):
            try:
                detail = scraper.scrape_publication_detail(pub["link"], pub.get("title", ""))
                results.append(detail)
                if i % 5 == 0:
                    print(f"[WORKER] Processed {i}/{len(publications)}")
            except WebDriverException as e:
                print(f"[WORKER] Error processing {pub['link']}: {e}")
                continue
    finally:
        scraper.close()
    
    return results

def split_into_chunks(items: List[Dict], num_chunks: int) -> List[List[Dict]]:
    """Split list into roughly equal chunks"""
    if num_chunks <= 1:
        return [items]
    chunk_size = ceil(len(items) / num_chunks)
    return [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

def main():
    parser = argparse.ArgumentParser(description="Coventry Pure Portal Publications Scraper")
    parser.add_argument("--outdir", default="data", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum listing pages to scrape")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for detail scraping")
    parser.add_argument("--listing-headless", action="store_true", help="Run listing scraper in headless mode")
    parser.add_argument("--legacy-headless", action="store_true", help="Use legacy headless mode")
    
    args = parser.parse_args()
    
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Scrape listing pages
    print(f"[STAGE 1] Scraping publication listings (max {args.max_pages} pages)...")
    listing_scraper = ListingScraper(headless=args.listing_headless, legacy=args.legacy_headless)
    publication_links = listing_scraper.scrape_all_pages(args.max_pages)
    
    if not publication_links:
        print("No publications found in listings")
        return
    
    # Save links
    links_file = output_dir / "publications_links.json"
    links_file.write_text(json.dumps(publication_links, indent=2), encoding="utf-8")
    print(f"[STAGE 1] Found {len(publication_links)} unique publications")
    
    # Stage 2: Scrape detailed information in parallel
    print(f"[STAGE 2] Scraping publication details with {args.workers} workers...")
    chunks = split_into_chunks(publication_links, max(1, args.workers))
    detailed_results = []
    
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process_detail_batch, chunk, True, args.legacy_headless) for chunk in chunks]
        completed = 0
        
        for future in as_completed(futures):
            batch_results = future.result() or []
            detailed_results.extend(batch_results)
            completed += 1
            print(f"[STAGE 2] Completed batch {completed}/{len(chunks)} (+{len(batch_results)} publications)")
    
    # Merge results (detailed data takes precedence over listing data)
    final_data = {}
    for pub in publication_links:
        final_data[pub["link"]] = {"title": pub["title"], "link": pub["link"]}
    
    for detail in detailed_results:
        final_data[detail["link"]] = detail
    
    # Save final results
    final_publications = list(final_data.values())
    output_file = output_dir / "publications.json"
    output_file.write_text(json.dumps(final_publications, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"[COMPLETE] Saved {len(final_publications)} publications to {output_file}")

if __name__ == "__main__":
    main()