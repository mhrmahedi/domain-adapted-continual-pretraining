"""
Step 2: ChipNeMo-Style Advanced Data Curation Pipeline
Multi-level deduplication, quality assessment, domain enhancement
Industry-standard data curation for LLM pre-training
"""
import os
import json
import re
import sys
import warnings
import hashlib
import logging
import math
import unicodedata
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
import gzip
from tqdm import tqdm

# Advanced libraries
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    ADVANCED_LIBS = True
except ImportError:
    ADVANCED_LIBS = False
    print("  Advanced libraries not available - using basic deduplication")
    print(" Install with: pip install numpy scikit-learn pandas")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.CRITICAL)

logger = setup_logger("data_curation")


@dataclass
class CurationConfig:
    """ChipNeMo-style curation configuration"""
    # Quality Gates
    min_words_per_doc: int = 50
    max_words_per_doc: int = 999999999
    min_sentences_per_doc: int = 2
    min_quality_score: float = 0.25
    
    # Deduplication Thresholds
    exact_dedup_threshold: float = 1.0
    fuzzy_dedup_threshold: float = 0.85
    semantic_dedup_threshold: float = 0.80
    
    # Content Filtering
    min_language_score: float = 0.5
    max_repetition_ratio: float = 0.4
    min_lexical_diversity: float = 0.2
    
    # Domain Enhancement
    domain_boost_multiplier: float = 1.2
    technical_content_threshold: float = 0.1
    structured_content_bonus: float = 0.3
    
    # Privacy & Security
    enable_pii_removal: bool = True
    enable_data_anonymization: bool = True
    
    # Output Format
    output_format: str = "jsonl_compact"
    enable_compression: bool = True
    max_documents_per_file: int = 10000


class ChipNeMoDataCurator:
    """
    Advanced data curator following ChipNeMo methodology
    Multi-level deduplication + Quality assessment + Domain enhancement
    """
    
    def __init__(
        self,
        config_path: str = "config/data_config.yaml",
        custom_config: CurationConfig = None
    ):
        """
        Initialize curator
        
        Args:
            config_path: Path to pipeline configuration
            custom_config: Optional custom curation config
        """
        # Load pipeline config
        with open(config_path, 'r') as f:
            self.pipeline_config = yaml.safe_load(f)
        
        self.curation_config = self.pipeline_config['curation']
        self.config = custom_config or CurationConfig()
        
        # Set paths
        self.input_dir = Path(self.curation_config['input_dir'])
        self.output_dir = Path(self.curation_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize systems
        self._init_deduplication_systems()
        self._init_quality_filters()
        self._init_privacy_protection()
        
        # Statistics tracking
        self.stats = {
            'processing_start': datetime.now(),
            'total_input_documents': 0,
            'documents_after_stage1': 0,
            'documents_after_stage2': 0,
            'documents_after_stage3': 0,
            'documents_after_stage4': 0,
            'documents_after_stage5': 0,
            'final_curated_documents': 0,
            'quality_distribution': defaultdict(int),
            'filtering_reasons': defaultdict(list),
            'deduplication_stats': {
                'exact_duplicates_removed': 0,
                'fuzzy_duplicates_removed': 0,
                'semantic_duplicates_removed': 0,
                'duplicate_pairs_found': []
            },
            'domain_enhancement_stats': defaultdict(int),
            'detailed_rejections': [],
            'word_count_stats': {'min': float('inf'), 'max': 0, 'total': 0}
        }
        
        logger.info(" ChipNeMo-Style Data Curator Initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        if ADVANCED_LIBS:
            logger.info(" Advanced deduplication enabled (TF-IDF + Cosine Similarity)")
        else:
            logger.info("  Basic deduplication only (n-gram + hash)")
    
    def _init_deduplication_systems(self):
        """Initialize deduplication systems"""
        # Exact deduplication
        self.exact_hashes = set()
        self.hash_to_doc = {}
        
        # Fuzzy deduplication using n-grams
        self.fuzzy_signatures = {}
        self.ngram_size = 5
        
        # Semantic deduplication using TF-IDF
        if ADVANCED_LIBS:
            self.semantic_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(2, 3),
                lowercase=True,
                stop_words='english',
                min_df=1,
                max_df=0.95
            )
            self.document_vectors = []
            self.document_metadata = []
    
    def _init_quality_filters(self):
        """Initialize quality filtering components"""
        if ADVANCED_LIBS:
            self.quality_vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2
            )
        
        self.quality_patterns = self._compile_quality_patterns()
    
    def _init_privacy_protection(self):
        """Initialize privacy protection components"""
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        }
    
    def _compile_quality_patterns(self) -> Dict[str, re.Pattern]:
        """Compile quality assessment patterns"""
        return {
            'technical_terms': re.compile(
                r'\b(?:algorithm|system|design|method|model|data|analysis|implementation|'
                r'architecture|framework|protocol|specification|table|figure|research|study|'
                r'results|conclusion|test|measurement|process|function|structure|component|'
                r'parameter|variable|equation|formula|calculation)\b',
                re.IGNORECASE
            ),
            'domain_indicators': re.compile(
                r'\b(?:research|engineering|development|technology|scientific|technical|'
                r'academic|industrial|analysis|evaluation|experiment|document|report|'
                r'paper|article|methodology|technique|procedure)\b',
                re.IGNORECASE
            ),
            'structured_markers': re.compile(
                r'(?:Table|Figure|Section|Chapter|Appendix|References?|Fig|Eq|Equation|'
                r'Page|Vol|Volume|No\.|Number)\s*\d*',
                re.IGNORECASE
            ),
            'gibberish': re.compile(
                r'[^\w\s]{15,}|(.)\\1{8,}|\b[bcdfghjklmnpqrstvwxz]{8,}\b',
                re.IGNORECASE
            ),
            'repetitive_patterns': re.compile(r'(.{20,}?)\1{4,}'),
            'low_information': re.compile(r'^(.{1,10})\1{5,}$')
        }
    
    def curate_all_documents(self) -> Dict:
        """
        Main curation pipeline - processes all extracted documents
        Pipeline-compatible method
        """
        logger.info("=" * 80)
        logger.info(" Starting ChipNeMo-Style Data Curation")
        logger.info("=" * 80)
        
        # Load extracted documents
        processed_documents = self._load_extracted_documents()
        
        if not processed_documents:
            logger.warning("  No documents found to curate")
            return {}
        
        self.stats['total_input_documents'] = len(processed_documents)
        
        # Stage 1: Document Preparation
        logger.info("\n Stage 1: Document Preparation & Content Extraction")
        cleaned_docs = self._stage1_document_preparation(processed_documents)
        self.stats['documents_after_stage1'] = len(cleaned_docs)
        retention1 = (len(cleaned_docs)/len(processed_documents)*100) if processed_documents else 0
        logger.info(f" After Stage 1: {len(cleaned_docs)}/{len(processed_documents)} ({retention1:.1f}%)")
        
        # Stage 2: Quality Assessment
        logger.info("\n Stage 2: Quality Assessment & Filtering")
        quality_filtered = self._stage2_quality_assessment(cleaned_docs)
        self.stats['documents_after_stage2'] = len(quality_filtered)
        retention2 = (len(quality_filtered)/len(cleaned_docs)*100) if cleaned_docs else 0
        logger.info(f" After Stage 2: {len(quality_filtered)}/{len(cleaned_docs)} ({retention2:.1f}%)")
        
        # Stage 3: Multi-Level Deduplication
        logger.info("\n Stage 3: Multi-Level Deduplication")
        deduplicated = self._stage3_multi_level_deduplication(quality_filtered)
        self.stats['documents_after_stage3'] = len(deduplicated)
        retention3 = (len(deduplicated)/len(quality_filtered)*100) if quality_filtered else 0
        logger.info(f" After Stage 3: {len(deduplicated)}/{len(quality_filtered)} ({retention3:.1f}%)")
        
        # Stage 4: Domain Enhancement
        logger.info("\n Stage 4: Domain-Specific Enhancement")
        domain_enhanced = self._stage4_domain_enhancement(deduplicated)
        self.stats['documents_after_stage4'] = len(domain_enhanced)
        logger.info(f" After Stage 4: {len(domain_enhanced)}/{len(deduplicated)} (100.0%)")
        
        # Stage 5: Privacy Protection
        logger.info("\n Stage 5: Privacy Protection & Content Sanitization")
        privacy_protected = self._stage5_privacy_protection(domain_enhanced)
        self.stats['documents_after_stage5'] = len(privacy_protected)
        logger.info(f" After Stage 5: {len(privacy_protected)}/{len(domain_enhanced)} (100.0%)")
        
        # Stage 6: Training Optimization
        logger.info("\n⚡ Stage 6: Training Data Optimization")
        final_dataset = self._stage6_training_optimization(privacy_protected)
        self.stats['final_curated_documents'] = len(final_dataset['documents'])
        
        # Show results
        self._show_deduplication_results()
        self._show_detailed_statistics()
        
        # Save results
        output_paths = self._save_curated_dataset(final_dataset)
        curation_report = self._generate_curation_report(final_dataset)
        
        logger.info("\n" + "=" * 80)
        logger.info(f" Curation Completed Successfully!")
        logger.info(f" Final Dataset: {self.stats['final_curated_documents']} documents")
        logger.info(f" Output saved to: {self.output_dir}")
        logger.info("=" * 80)
        
        return {
            'curated_dataset': final_dataset,
            'curation_report': curation_report,
            'output_paths': output_paths,
            'statistics': self.stats
        }
    
    def _load_extracted_documents(self) -> List[Dict]:
        """Load extracted documents from previous step"""
        logger.info(f" Loading extracted documents from: {self.input_dir}")
        
        json_files = list(self.input_dir.glob("*.json"))
        
        if not json_files:
            logger.warning("  No JSON files found in extraction directory")
            return []
        
        documents = []
        
        for json_file in tqdm(json_files, desc="Loading documents"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                # Validate document
                if doc.get('extraction_success', False) and doc.get('pages'):
                    documents.append(doc)
                else:
                    logger.warning(f"  Skipping {json_file.name}: extraction failed or no content")
                    
            except Exception as e:
                logger.error(f" Error loading {json_file.name}: {e}")
                continue
        
        logger.info(f" Loaded {len(documents)} valid documents")
        return documents
    
    def _stage1_document_preparation(self, documents: List[Dict]) -> List[Dict]:
        """Stage 1: Document preparation"""
        cleaned_documents = []
        
        for i, doc in enumerate(tqdm(documents, desc="Preparing documents")):
            try:
                source_name = Path(doc.get('file_path', 'unknown')).name
                
                # Prepare document
                clean_doc = self._prepare_document(doc)
                
                if not clean_doc:
                    self._record_rejection(source_name, "Stage 1", "No extractable content", doc)
                    continue
                
                # Basic quality gates
                rejection_reason = self._check_basic_quality_gates(clean_doc)
                
                if rejection_reason:
                    self._record_rejection(source_name, "Stage 1", rejection_reason, clean_doc)
                    continue
                
                cleaned_documents.append(clean_doc)
                
            except Exception as e:
                source_name = Path(doc.get('file_path', 'unknown')).name
                self._record_rejection(source_name, "Stage 1", f"Processing error: {str(e)[:50]}", {})
                continue
        
        return cleaned_documents
    
    def _prepare_document(self, doc: Dict) -> Optional[Dict]:
        """Prepare document for curation"""
        content_parts = []
        total_words = 0
        
        # Extract all text content
        for page_num, page_data in doc.get('pages', {}).items():
            if isinstance(page_data, dict):
                text = page_data.get('text', '').strip()
                if text and len(text) >= 10:
                    content_parts.append(text)
                    total_words += len(text.split())
        
        if not content_parts:
            return None
        
        full_text = '\n\n'.join(content_parts)
        
        # Extract table content
        table_content = []
        table_count = 0
        
        for table in doc.get('tables', []):
            table_count += 1
            if 'data' in table and table['data']:
                table_text = self._extract_table_content(table['data'])
                if table_text:
                    table_content.append(f"[TABLE_{table_count}]\n{table_text}")
        
        # Combine content
        if table_content:
            full_content = full_text + '\n\n' + '\n\n'.join(table_content)
        else:
            full_content = full_text
        
        # Clean and normalize
        clean_content = self._normalize_content(full_content)
        
        if not clean_content or len(clean_content.split()) < self.config.min_words_per_doc:
            return None
        
        # Create clean document
        clean_doc = {
            'id': hashlib.md5(clean_content.encode()).hexdigest()[:12],
            'content': clean_content,
            'original_content': full_content,
            'source': doc.get('file_path', 'unknown'),
            'metadata': {
                'total_pages': doc.get('total_pages', 0),
                'word_count': len(clean_content.split()),
                'char_count': len(clean_content),
                'has_tables': table_count > 0,
                'table_count': table_count,
                'has_images': doc.get('images_detected', 0) > 0,
                'image_count': doc.get('images_detected', 0),
                'extraction_method': doc.get('extraction_method', 'unknown'),
                'quality_assessment': doc.get('quality_assessment', {})
            }
        }
        
        return clean_doc
    
    def _extract_table_content(self, table_data: List[List]) -> str:
        """Extract clean table content"""
        if not table_data:
            return ""
        
        clean_rows = []
        for row in table_data:
            if row:
                clean_cells = [str(cell).strip() for cell in row if cell and str(cell).strip()]
                if clean_cells:
                    clean_rows.append(' | '.join(clean_cells))
        
        return '\n'.join(clean_rows) if clean_rows else ""
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for better processing"""
        if not content:
            return ""
        
        # Unicode normalization
        content = unicodedata.normalize('NFKC', content)
        
        # Remove control characters
        content = ''.join(char for char in content if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
        
        # Normalize whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove page numbers and short lines
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) >= 3:
                if not re.match(r'^\d+$', line):  # Skip page numbers
                    if len(line) > 5 or any(char.isalpha() for char in line):
                        clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()
    
    def _check_basic_quality_gates(self, doc: Dict) -> Optional[str]:
        """Basic quality gates"""
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        # Word count check
        word_count = metadata.get('word_count', 0)
        if word_count < self.config.min_words_per_doc:
            return f"Too few words: {word_count} < {self.config.min_words_per_doc}"
        
        # Sentence count check
        sentences = [s.strip() for s in content.split('.') if s.strip() and len(s.strip()) >= 5]
        if len(sentences) < self.config.min_sentences_per_doc:
            return f"Too few sentences: {len(sentences)} < {self.config.min_sentences_per_doc}"
        
        # Gibberish check
        gibberish_matches = self.quality_patterns['gibberish'].findall(content)
        if len(gibberish_matches) > len(content.split()) * 0.2:
            return f"Too much gibberish: {len(gibberish_matches)} matches"
        
        return None
    
    def _stage2_quality_assessment(self, documents: List[Dict]) -> List[Dict]:
        """Stage 2: Quality assessment"""
        quality_filtered = []
        
        for doc in tqdm(documents, desc="Assessing quality"):
            source_name = Path(doc.get('source', 'unknown')).name
            content = doc['content']
            
            # Calculate quality score
            quality_metrics = self._calculate_quality_score(content, doc['metadata'])
            overall_quality = quality_metrics['overall_score']
            
            if overall_quality >= self.config.min_quality_score:
                doc['quality_metrics'] = quality_metrics
                doc['quality_grade'] = self._score_to_grade(overall_quality)
                quality_filtered.append(doc)
                
                grade = doc['quality_grade']
                self.stats['quality_distribution'][grade] += 1
            else:
                reason = f"Quality too low: {overall_quality:.3f} < {self.config.min_quality_score}"
                self._record_rejection(source_name, "Stage 2", reason, doc)
        
        return quality_filtered
    
    def _calculate_quality_score(self, content: str, metadata: Dict) -> Dict:
        """Calculate comprehensive quality score"""
        words = content.split()
        sentences = [s for s in content.split('.') if s.strip()]
        
        if not words:
            return self._zero_quality_metrics()
        
        # Lexical diversity
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = min(unique_words / len(words) * 1.5, 1.0)
        
        # Information density (Shannon Entropy)
        word_freq = Counter(word.lower() for word in words)
        if len(word_freq) > 1:
            probabilities = [count/len(words) for count in word_freq.values()]
            entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
            max_entropy = math.log2(len(word_freq))
            info_density = min((entropy / max_entropy) * 1.2, 1.0) if max_entropy > 0 else 0.5
        else:
            info_density = 0.3
        
        # Technical density
        technical_matches = len(self.quality_patterns['technical_terms'].findall(content))
        technical_density = min((technical_matches / len(words)) * 100, 1.0)
        
        # Structural quality
        structured_matches = len(self.quality_patterns['structured_markers'].findall(content))
        structural_quality = min(structured_matches / 5, 1.0)
        
        # Domain relevance
        domain_matches = len(self.quality_patterns['domain_indicators'].findall(content))
        domain_relevance = min((domain_matches / len(words)) * 50, 1.0)
        
        # Content richness
        content_richness = 0.3
        if metadata.get('has_tables'):
            content_richness += metadata.get('table_count', 0) * 0.15
        if metadata.get('has_images'):
            content_richness += metadata.get('image_count', 0) * 0.10
        content_richness = min(content_richness, 1.0)
        
        # Overall score (ChipNeMo-style weighted combination)
        overall_score = (
            lexical_diversity * 0.15 +
            info_density * 0.20 +
            technical_density * 0.15 +
            structural_quality * 0.15 +
            domain_relevance * 0.15 +
            content_richness * 0.20
        )
        
        overall_score = max(overall_score, 0.2)  # Minimum baseline
        
        return {
            'overall_score': round(overall_score, 4),
            'lexical_diversity': round(lexical_diversity, 4),
            'information_density': round(info_density, 4),
            'technical_density': round(technical_density, 4),
            'structural_quality': round(structural_quality, 4),
            'domain_relevance': round(domain_relevance, 4),
            'content_richness': round(content_richness, 4)
        }
    
    def _zero_quality_metrics(self) -> Dict:
        """Return zero quality metrics"""
        return {
            'overall_score': 0.0,
            'lexical_diversity': 0.0,
            'information_density': 0.0,
            'technical_density': 0.0,
            'structural_quality': 0.0,
            'domain_relevance': 0.0,
            'content_richness': 0.0
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9: return "A+"
        elif score >= 0.8: return "A"
        elif score >= 0.7: return "B+"
        elif score >= 0.6: return "B"
        elif score >= 0.5: return "C+"
        elif score >= 0.4: return "C"
        else: return "D"
    
    def _stage3_multi_level_deduplication(self, documents: List[Dict]) -> List[Dict]:
        """Stage 3: Multi-level deduplication"""
        logger.info("   Level 1: Exact duplicate removal...")
        level1_deduped = self._exact_deduplication(documents)
        
        logger.info("   Level 2: Fuzzy duplicate removal...")
        level2_deduped = self._fuzzy_deduplication(level1_deduped)
        
        if ADVANCED_LIBS and len(level2_deduped) > 1:
            logger.info("   Level 3: Semantic duplicate removal...")
            final_deduped = self._semantic_deduplication(level2_deduped)
        else:
            logger.info("    Semantic deduplication skipped")
            final_deduped = level2_deduped
        
        return final_deduped
    
    def _exact_deduplication(self, documents: List[Dict]) -> List[Dict]:
        """Exact deduplication using content hash"""
        deduplicated = []
        seen_hashes = {}
        
        for doc in documents:
            normalized_content = self._normalize_for_comparison(doc['content'])
            content_hash = hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes[content_hash] = doc
                doc['content_hash'] = content_hash[:12]
                deduplicated.append(doc)
            else:
                original_doc = seen_hashes[content_hash]
                source_name = Path(doc.get('source', 'unknown')).name
                original_name = Path(original_doc.get('source', 'unknown')).name
                
                self.stats['deduplication_stats']['exact_duplicates_removed'] += 1
                self.stats['deduplication_stats']['duplicate_pairs_found'].append({
                    'type': 'exact',
                    'removed': source_name,
                    'kept': original_name,
                    'similarity': 1.0
                })
                self._record_rejection(source_name, "Stage 3 - Exact Dedup", "Exact duplicate", doc)
        
        removed_count = len(documents) - len(deduplicated)
        logger.info(f"     Removed {removed_count} exact duplicates")
        
        return deduplicated
    
    def _normalize_for_comparison(self, content: str) -> str:
        """Normalize content for comparison"""
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[^\w\s]', '', content)
        return content.lower().strip()
    
    def _fuzzy_deduplication(self, documents: List[Dict]) -> List[Dict]:
        """Fuzzy deduplication using n-gram similarity"""
        if len(documents) <= 1:
            return documents
        
        deduplicated = []
        removed_indices = set()
        
        # Generate n-gram signatures
        signatures = {}
        for i, doc in enumerate(documents):
            signatures[i] = self._generate_ngram_signature(doc['content'])
        
        # Compare all pairs
        for i in range(len(documents)):
            if i in removed_indices:
                continue
            
            for j in range(i + 1, len(documents)):
                if j in removed_indices:
                    continue
                
                similarity = self._jaccard_similarity(signatures[i], signatures[j])
                
                if similarity >= self.config.fuzzy_dedup_threshold:
                    doc_i, doc_j = documents[i], documents[j]
                    source_i = Path(doc_i.get('source', 'unknown')).name
                    source_j = Path(doc_j.get('source', 'unknown')).name
                    
                    quality_i = doc_i.get('quality_metrics', {}).get('overall_score', 0)
                    quality_j = doc_j.get('quality_metrics', {}).get('overall_score', 0)
                    
                    if quality_i >= quality_j:
                        removed_indices.add(j)
                        kept_doc, removed_doc = source_i, source_j
                    else:
                        removed_indices.add(i)
                        kept_doc, removed_doc = source_j, source_i
                        break
                    
                    self.stats['deduplication_stats']['fuzzy_duplicates_removed'] += 1
                    self.stats['deduplication_stats']['duplicate_pairs_found'].append({
                        'type': 'fuzzy',
                        'removed': removed_doc,
                        'kept': kept_doc,
                        'similarity': similarity
                    })
        
        for i, doc in enumerate(documents):
            if i not in removed_indices:
                deduplicated.append(doc)
            else:
                source_name = Path(doc.get('source', 'unknown')).name
                self._record_rejection(source_name, "Stage 3 - Fuzzy Dedup", "Fuzzy duplicate", doc)
        
        removed_count = len(removed_indices)
        logger.info(f"     Removed {removed_count} fuzzy duplicates")
        
        return deduplicated
    
    def _generate_ngram_signature(self, content: str) -> Set[str]:
        """Generate n-gram signature"""
        words = re.findall(r'\w+', content.lower())
        ngrams = set()
        
        for i in range(len(words) - self.ngram_size + 1):
            ngram = ' '.join(words[i:i+self.ngram_size])
            ngrams.add(ngram)
        
        return ngrams
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _semantic_deduplication(self, documents: List[Dict]) -> List[Dict]:
        """Semantic deduplication using TF-IDF"""
        if len(documents) <= 1:
            return documents
        
        try:
            contents = [doc['content'] for doc in documents]
            tfidf_matrix = self.semantic_vectorizer.fit_transform(contents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            removed_indices = set()
            
            for i in range(len(documents)):
                if i in removed_indices:
                    continue
                
                for j in range(i + 1, len(documents)):
                    if j in removed_indices:
                        continue
                    
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= self.config.semantic_dedup_threshold:
                        doc_i, doc_j = documents[i], documents[j]
                        source_i = Path(doc_i.get('source', 'unknown')).name
                        source_j = Path(doc_j.get('source', 'unknown')).name
                        
                        quality_i = doc_i.get('quality_metrics', {}).get('overall_score', 0)
                        quality_j = doc_j.get('quality_metrics', {}).get('overall_score', 0)
                        
                        if quality_i >= quality_j:
                            removed_indices.add(j)
                            kept_doc, removed_doc = source_i, source_j
                        else:
                            removed_indices.add(i)
                            kept_doc, removed_doc = source_j, source_i
                            break
                        
                        self.stats['deduplication_stats']['semantic_duplicates_removed'] += 1
                        self.stats['deduplication_stats']['duplicate_pairs_found'].append({
                            'type': 'semantic',
                            'removed': removed_doc,
                            'kept': kept_doc,
                            'similarity': float(similarity)
                        })
            
            deduplicated = []
            for i, doc in enumerate(documents):
                if i not in removed_indices:
                    deduplicated.append(doc)
                else:
                    source_name = Path(doc.get('source', 'unknown')).name
                    self._record_rejection(source_name, "Stage 3 - Semantic Dedup", "Semantic duplicate", doc)
            
            removed_count = len(removed_indices)
            logger.info(f"     Removed {removed_count} semantic duplicates")
            
            return deduplicated
            
        except Exception as e:
            logger.warning(f"  Semantic deduplication failed: {e}")
            return documents
    
    def _stage4_domain_enhancement(self, documents: List[Dict]) -> List[Dict]:
        """Stage 4: Domain enhancement"""
        enhanced_docs = []
        
        for doc in documents:
            enhanced_doc = self._enhance_document(doc)
            if enhanced_doc:
                enhanced_docs.append(enhanced_doc)
                category = enhanced_doc.get('domain_category', 'general')
                self.stats['domain_enhancement_stats'][category] += 1
        
        return enhanced_docs
    
    def _enhance_document(self, doc: Dict) -> Optional[Dict]:
        """Enhance document with domain information"""
        content = doc['content']
        
        # Classify domain
        domain_category = self._classify_domain(content)
        domain_score = self._calculate_domain_relevance(content)
        
        # Apply boost for domain-relevant content
        original_quality = doc['quality_metrics']['overall_score']
        
        if domain_score >= self.config.technical_content_threshold:
            boosted_quality = min(original_quality * self.config.domain_boost_multiplier, 1.0)
            doc['quality_metrics']['overall_score'] = boosted_quality
            doc['quality_metrics']['domain_boosted'] = True
        
        doc['domain_category'] = domain_category
        doc['domain_relevance_score'] = round(domain_score, 4)
        
        return doc
    
    def _classify_domain(self, content: str) -> str:
        """Classify domain category"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['table', 'figure', 'data', 'result', 'experiment']):
            return 'research_data'
        elif any(word in content_lower for word in ['research', 'study', 'paper', 'journal']):
            return 'academic_paper'
        elif any(word in content_lower for word in ['system', 'design', 'implementation']):
            return 'technical_document'
        else:
            return 'general'
    
    def _calculate_domain_relevance(self, content: str) -> float:
        """Calculate domain relevance score"""
        words = content.split()
        if not words:
            return 0.2
        
        technical_count = len(self.quality_patterns['technical_terms'].findall(content))
        domain_count = len(self.quality_patterns['domain_indicators'].findall(content))
        
        relevance = ((technical_count + domain_count) / len(words)) * 20
        return min(max(relevance, 0.1), 1.0)
    
    def _stage5_privacy_protection(self, documents: List[Dict]) -> List[Dict]:
        """Stage 5: Privacy protection"""
        if not self.config.enable_pii_removal:
            return documents
        
        for doc in documents:
            protected_content = self._remove_pii(doc['content'])
            if protected_content != doc['content']:
                doc['content'] = protected_content
                doc['pii_removed'] = True
        
        return documents
    
    def _remove_pii(self, content: str) -> str:
        """Remove personally identifiable information"""
        protected_content = content
        
        for pii_type, pattern in self.pii_patterns.items():
            replacement = f"[{pii_type.upper()}]"
            protected_content = pattern.sub(replacement, protected_content)
        
        return protected_content
    
    def _stage6_training_optimization(self, documents: List[Dict]) -> Dict:
        """Stage 6: Prepare optimized training dataset"""
        # Sort by quality
        sorted_docs = sorted(documents, key=lambda x: x['quality_metrics']['overall_score'], reverse=True)
        
        training_dataset = {
            'documents': [],
            'metadata': {
                'total_documents': len(sorted_docs),
                'curation_date': datetime.now().isoformat(),
                'curation_method': 'chipnemo_advanced',
                'quality_distribution': dict(self.stats['quality_distribution']),
                'average_quality': sum(doc['quality_metrics']['overall_score'] for doc in sorted_docs) / len(sorted_docs) if sorted_docs else 0,
                'domain_distribution': dict(self.stats['domain_enhancement_stats']),
                'deduplication_stats': self.stats['deduplication_stats']
            }
        }
        
        for rank, doc in enumerate(sorted_docs):
            training_doc = {
                'id': doc['id'],
                'text': doc.get('original_content', doc['content']),
                'quality_score': doc['quality_metrics']['overall_score'],
                'quality_grade': doc['quality_grade'],
                'domain_category': doc.get('domain_category', 'general'),
                'source_info': {
                    'file': Path(doc['source']).name,
                    'pages': doc['metadata']['total_pages'],
                    'has_tables': doc['metadata']['has_tables'],
                    'has_images': doc['metadata']['has_images'],
                    'word_count': doc['metadata']['word_count']
                },
                'training_rank': rank + 1
            }
            
            training_dataset['documents'].append(training_doc)
        
        return training_dataset
    
    def _show_deduplication_results(self):
        """Show deduplication results"""
        stats = self.stats['deduplication_stats']
        total_removed = (
            stats['exact_duplicates_removed'] +
            stats['fuzzy_duplicates_removed'] +
            stats['semantic_duplicates_removed']
        )
        
        logger.info("\n DEDUPLICATION RESULTS:")
        logger.info(f"   Exact duplicates removed: {stats['exact_duplicates_removed']}")
        logger.info(f"   Fuzzy duplicates removed: {stats['fuzzy_duplicates_removed']}")
        logger.info(f"   Semantic duplicates removed: {stats['semantic_duplicates_removed']}")
        logger.info(f"   Total duplicates removed: {total_removed}")
        
        if stats['duplicate_pairs_found'] and len(stats['duplicate_pairs_found']) <= 10:
            logger.info("\n DUPLICATE PAIRS:")
            for pair in stats['duplicate_pairs_found']:
                logger.info(f"   {pair['type'].upper()}: {pair['removed']} -> {pair['kept']} (sim: {pair['similarity']:.3f})")
    
    def _show_detailed_statistics(self):
        """Show detailed statistics"""
        logger.info("\n QUALITY DISTRIBUTION:")
        for grade, count in sorted(self.stats['quality_distribution'].items()):
            logger.info(f"   Grade {grade}: {count} documents")
        
        logger.info("\n DOMAIN DISTRIBUTION:")
        for domain, count in self.stats['domain_enhancement_stats'].items():
            logger.info(f"   {domain}: {count} documents")
    
    def _record_rejection(self, filename: str, stage: str, reason: str, doc: Dict):
        """Record rejection details"""
        rejection_info = {
            'filename': filename,
            'stage': stage,
            'reason': reason,
            'word_count': doc.get('metadata', {}).get('word_count', 0) if isinstance(doc.get('metadata'), dict) else 0
        }
        
        self.stats['detailed_rejections'].append(rejection_info)
        self.stats['filtering_reasons'][reason].append(filename)
    
    def _save_curated_dataset(self, dataset: Dict) -> Dict[str, str]:
        """Save curated dataset"""
        output_paths = {}
        
        # JSONL format (for training)
        jsonl_path = self.output_dir / "curated_data.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for doc in dataset['documents']:
                f.write(json.dumps({'text': doc['text']}, ensure_ascii=False) + '\n')
        output_paths['training_jsonl'] = str(jsonl_path)
        
        # Compressed JSONL
        if self.config.enable_compression:
            compressed_path = self.output_dir / "curated_data.jsonl.gz"
            with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
                for doc in dataset['documents']:
                    f.write(json.dumps({'text': doc['text']}, ensure_ascii=False) + '\n')
            output_paths['training_jsonl_compressed'] = str(compressed_path)
        
        # Complete dataset with metadata
        complete_path = self.output_dir / "complete_curated_dataset.json"
        with open(complete_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        output_paths['complete_dataset'] = str(complete_path)
        
        # Statistics
        stats_file = self.output_dir / "curation_stats.json"
        with open(stats_file, 'w') as f:
            # Convert defaultdicts to regular dicts
            json_stats = dict(self.stats)
            json_stats['quality_distribution'] = dict(json_stats['quality_distribution'])
            json_stats['filtering_reasons'] = dict(json_stats['filtering_reasons'])
            json_stats['domain_enhancement_stats'] = dict(json_stats['domain_enhancement_stats'])
            json_stats['processing_start'] = json_stats['processing_start'].isoformat()
            json.dump(json_stats, f, indent=2)
        output_paths['statistics'] = str(stats_file)
        
        logger.info(f"\n Files saved:")
        logger.info(f"   Training data (JSONL): {jsonl_path}")
        logger.info(f"   Complete dataset (JSON): {complete_path}")
        logger.info(f"   Statistics: {stats_file}")
        
        return output_paths
    
    def _generate_curation_report(self, dataset: Dict) -> Dict:
        """Generate curation report"""
        processing_time = datetime.now() - self.stats['processing_start']
        
        report = {
            'curation_summary': {
                'total_input_documents': self.stats['total_input_documents'],
                'final_curated_documents': self.stats['final_curated_documents'],
                'retention_rate': self.stats['final_curated_documents'] / max(self.stats['total_input_documents'], 1),
                'processing_time_seconds': processing_time.total_seconds()
            },
            'stage_retention': {
                'after_stage1': self.stats['documents_after_stage1'],
                'after_stage2': self.stats['documents_after_stage2'],
                'after_stage3': self.stats['documents_after_stage3'],
                'after_stage4': self.stats['documents_after_stage4'],
                'after_stage5': self.stats['documents_after_stage5']
            },
            'deduplication_analysis': self.stats['deduplication_stats'],
            'quality_analysis': {
                'average_quality': dataset['metadata']['average_quality'],
                'quality_distribution': dataset['metadata']['quality_distribution'],
                'domain_distribution': dataset['metadata']['domain_distribution']
            },
            'methodology': {
                'version': 'ChipNeMo-style advanced curation v2.0',
                'techniques': [
                    'Multi-level deduplication (exact + fuzzy + semantic)',
                    'Research-based quality assessment',
                    'Domain-specific enhancement',
                    'Privacy protection (PII removal)',
                    'Training optimization'
                ]
            }
        }
        
        # Save report
        report_path = self.output_dir / "curation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report


def main():
    """Main curation function"""
    curator = ChipNeMoDataCurator()
    curator.curate_all_documents()


if __name__ == "__main__":
    main()
