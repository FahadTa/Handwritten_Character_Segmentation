import re
import random
import string
from typing import List, Optional, Set
from pathlib import Path

import nltk
import wikipedia
from tqdm import tqdm


class TextSampler:
    """
    Text corpus sampler for synthetic dataset generation.
    Fetches text from Wikipedia or custom sources and prepares it for rendering.
    """
    
    def __init__(
        self,
        source: str = "wikipedia",
        character_set: Optional[str] = None,
        min_sentence_length: int = 5,
        max_sentence_length: int = 20,
        language: str = "en",
        seed: int = 42
    ):
        """
        Initialize text sampler.
        
        Args:
            source: Text source ('wikipedia', 'custom', or 'mixed')
            character_set: Allowed characters for filtering
            min_sentence_length: Minimum words per sentence
            max_sentence_length: Maximum words per sentence
            language: Language code for Wikipedia
            seed: Random seed for reproducibility
        """
        self.source = source
        self.character_set = set(character_set) if character_set else None
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.language = language
        self.seed = seed
        
        random.seed(seed)
        
        self._download_nltk_data()
        self.sentences = []
        
    def _download_nltk_data(self) -> None:
        """Download required NLTK data packages."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = text.strip()
        
        return text
    
    def _filter_by_characters(self, text: str) -> bool:
        """
        Check if text contains only allowed characters.
        
        Args:
            text: Text to check
            
        Returns:
            True if all characters are allowed, False otherwise
        """
        if self.character_set is None:
            return True
        
        text_chars = set(text)
        return text_chars.issubset(self.character_set)
    
    def _filter_sentence(self, sentence: str) -> bool:
        """
        Filter sentence based on length and character constraints.
        
        Args:
            sentence: Sentence to filter
            
        Returns:
            True if sentence passes filters, False otherwise
        """
        words = sentence.split()
        word_count = len(words)
        
        if word_count < self.min_sentence_length:
            return False
        
        if word_count > self.max_sentence_length:
            return False
        
        if not self._filter_by_characters(sentence):
            return False
        
        return True
    
    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract and filter sentences from text.
        
        Args:
            text: Text to extract sentences from
            
        Returns:
            List of filtered sentences
        """
        text = self._clean_text(text)
        
        raw_sentences = nltk.sent_tokenize(text)
        
        filtered_sentences = []
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if self._filter_sentence(sentence):
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def fetch_wikipedia_text(
        self,
        num_articles: int = 100,
        categories: Optional[List[str]] = None
    ) -> List[str]:
        """
        Fetch text from random Wikipedia articles.
        
        Args:
            num_articles: Number of articles to fetch
            categories: List of Wikipedia categories to sample from
            
        Returns:
            List of sentences
        """
        print(f"Fetching text from {num_articles} Wikipedia articles...")
        
        wikipedia.set_lang(self.language)
        
        sentences = []
        articles_fetched = 0
        attempts = 0
        max_attempts = num_articles * 3
        
        pbar = tqdm(total=num_articles, desc="Fetching articles")
        
        while articles_fetched < num_articles and attempts < max_attempts:
            attempts += 1
            
            try:
                if categories:
                    category = random.choice(categories)
                    pages = wikipedia.search(category, results=10)
                    if not pages:
                        continue
                    page_title = random.choice(pages)
                else:
                    page_title = wikipedia.random(pages=1)
                
                page = wikipedia.page(page_title, auto_suggest=False)
                content = page.content
                
                article_sentences = self._extract_sentences(content)
                
                if article_sentences:
                    sentences.extend(article_sentences)
                    articles_fetched += 1
                    pbar.update(1)
                
            except (
                wikipedia.exceptions.DisambiguationError,
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.WikipediaException
            ):
                continue
            except Exception as e:
                print(f"Warning: Error fetching article: {e}")
                continue
        
        pbar.close()
        
        print(f"Collected {len(sentences)} sentences from {articles_fetched} articles")
        
        return sentences
    
    def load_custom_text(self, file_path: str) -> List[str]:
        """
        Load text from custom file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of sentences
        """
        print(f"Loading text from {file_path}...")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        sentences = self._extract_sentences(text)
        
        print(f"Collected {len(sentences)} sentences from custom file")
        
        return sentences
    
    def build_corpus(
        self,
        num_wikipedia_articles: Optional[int] = None,
        custom_file_path: Optional[str] = None,
        wikipedia_categories: Optional[List[str]] = None
    ) -> None:
        """
        Build text corpus from configured sources.
        
        Args:
            num_wikipedia_articles: Number of Wikipedia articles to fetch
            custom_file_path: Path to custom text file
            wikipedia_categories: Wikipedia categories to sample from
        """
        all_sentences = []
        
        if self.source in ["wikipedia", "mixed"]:
            if num_wikipedia_articles is None:
                raise ValueError("num_wikipedia_articles must be specified for Wikipedia source")
            
            wiki_sentences = self.fetch_wikipedia_text(
                num_articles=num_wikipedia_articles,
                categories=wikipedia_categories
            )
            all_sentences.extend(wiki_sentences)
        
        if self.source in ["custom", "mixed"]:
            if custom_file_path is None:
                raise ValueError("custom_file_path must be specified for custom source")
            
            custom_sentences = self.load_custom_text(custom_file_path)
            all_sentences.extend(custom_sentences)
        
        self.sentences = all_sentences
        random.shuffle(self.sentences)
        
        print(f"Total corpus size: {len(self.sentences)} sentences")
    
    def sample_text(
        self,
        num_chars: Optional[int] = None,
        num_sentences: Optional[int] = None
    ) -> str:
        """
        Sample text from corpus.
        
        Args:
            num_chars: Target number of characters (approximate)
            num_sentences: Number of sentences to sample
            
        Returns:
            Sampled text string
        """
        if not self.sentences:
            raise ValueError("Corpus is empty. Call build_corpus() first.")
        
        if num_sentences is not None:
            sampled_sentences = random.sample(
                self.sentences,
                min(num_sentences, len(self.sentences))
            )
            return " ".join(sampled_sentences)
        
        if num_chars is not None:
            text = ""
            while len(text) < num_chars and self.sentences:
                sentence = random.choice(self.sentences)
                text += sentence + " "
            
            return text.strip()[:num_chars]
        
        return random.choice(self.sentences)
    
    def sample_sentences(self, num_sentences: int) -> List[str]:
        """
        Sample multiple individual sentences.
        
        Args:
            num_sentences: Number of sentences to sample
            
        Returns:
            List of sampled sentences
        """
        if not self.sentences:
            raise ValueError("Corpus is empty. Call build_corpus() first.")
        
        return random.sample(
            self.sentences,
            min(num_sentences, len(self.sentences))
        )
    
    def get_corpus_stats(self) -> dict:
        """
        Get statistics about the text corpus.
        
        Returns:
            Dictionary with corpus statistics
        """
        if not self.sentences:
            return {
                "num_sentences": 0,
                "total_chars": 0,
                "total_words": 0,
                "avg_sentence_length": 0,
                "unique_chars": set()
            }
        
        total_chars = sum(len(s) for s in self.sentences)
        total_words = sum(len(s.split()) for s in self.sentences)
        unique_chars = set("".join(self.sentences))
        
        return {
            "num_sentences": len(self.sentences),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_sentence_length": total_words / len(self.sentences),
            "unique_chars": unique_chars,
            "avg_chars_per_sentence": total_chars / len(self.sentences)
        }
    
    def save_corpus(self, output_path: str) -> None:
        """
        Save corpus to file for reuse.
        
        Args:
            output_path: Path to save corpus
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in self.sentences:
                f.write(sentence + "\n")
        
        print(f"Corpus saved to {output_path}")
    
    def load_corpus(self, corpus_path: str) -> None:
        """
        Load pre-built corpus from file.
        
        Args:
            corpus_path: Path to corpus file
        """
        corpus_path = Path(corpus_path)
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.sentences = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded corpus with {len(self.sentences)} sentences")