"""
Step 1: Hybrid PDF Extractor with Quality Assessment
"""
import os
import json
import sys
import warnings
import logging
import re
import math
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict
from io import StringIO
from tqdm import tqdm

# Add project root to Python path for cross-platform compatibility
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import from src works correctly on both Windows and Linux
from src.utils.logger import setup_logger



# SUPPRESS ALL WARNINGS AND LOGS FOR CLEAN OUTPUT
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.CRITICAL)
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SuppressOutput:
    """Context manager to suppress stdout/stderr"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# Check library availability with suppressed output
libs_available = {}

with SuppressOutput():
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.documents.elements import Table, Image, Text, Title, NarrativeText
        libs_available['unstructured'] = True
    except ImportError:
        libs_available['unstructured'] = False

    try:
        import fitz  # PyMuPDF
        libs_available['pymupdf'] = True
    except ImportError:
        libs_available['pymupdf'] = False

    try:
        import pdfplumber
        libs_available['pdfplumber'] = True
    except ImportError:
        libs_available['pdfplumber'] = False

    try:
        from docx import Document as DocxDocument
        libs_available['docx'] = True
    except ImportError:
        libs_available['docx'] = False

    try:
        from bs4 import BeautifulSoup
        libs_available['beautifulsoup'] = True
    except ImportError:
        libs_available['beautifulsoup'] = False


logger = setup_logger("data_extraction")


class ResearchBasedQualityAssessment:
    """
    Research-based quality assessment using established metrics
    Based on academic literature in NLP and computational linguistics
    """
    
    def assess_content_quality(self, text: str, tables: List[Dict] = None) -> Dict:
        """
        Research-based quality assessment using established metrics
        
        Components:
        - Lexical Diversity (Type-Token Ratio)
        - Readability (Flesch-Kincaid)
        - Information Density (Shannon Entropy)
        - Structural Complexity
        - Content Richness
        
        Args:
            text: Text content to assess
            tables: List of detected tables
            
        Returns:
            Dictionary with quality metrics
        """
        if not text or not text.strip():
            return self._create_zero_quality_result()
        
        words = text.split()
        sentences = self._split_sentences(text)
        
        # Component 1: Lexical Diversity (Established NLP metric)
        lexical_diversity = self._calculate_lexical_diversity(words)
        
        # Component 2: Readability (Flesch-Kincaid, established 1975)
        readability = self._calculate_readability(text, words, sentences)
        
        # Component 3: Information Density (Shannon Entropy)
        information_density = self._calculate_information_density(words)
        
        # Component 4: Structural Complexity (Research-based)
        structural_complexity = self._calculate_structural_complexity(sentences)
        
        # Component 5: Content Richness (Tables, figures as indicators)
        content_richness = self._calculate_content_richness(text, tables or [])
        
        # Weighted combination (based on NLP literature)
        quality_score = (
            lexical_diversity * 0.25 +      # 25% - vocabulary richness
            readability * 0.25 +            # 25% - readability
            information_density * 0.20 +    # 20% - information content
            structural_complexity * 0.15 +  # 15% - sentence structure
            content_richness * 0.15         # 15% - structured content
        )
        
        return {
            'overall_quality': min(max(quality_score, 0.0), 1.0),
            'components': {
                'lexical_diversity': round(lexical_diversity, 3),
                'readability': round(readability, 3),
                'information_density': round(information_density, 3),
                'structural_complexity': round(structural_complexity, 3),
                'content_richness': round(content_richness, 3)
            },
            'methodology': 'research_based',
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
    
    def _calculate_lexical_diversity(self, words: List[str]) -> float:
        """Type-Token Ratio - established lexical diversity metric"""
        if len(words) == 0:
            return 0.0
        
        unique_words = len(set(word.lower() for word in words))
        total_words = len(words)
        
        # TTR with scaling (research-based approach)
        return min(unique_words / total_words, 1.0)
    
    def _calculate_readability(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Flesch Reading Ease - established readability metric (1975)"""
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        # Count syllables (simplified)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula
        if len(sentences) > 0 and total_syllables > 0:
            flesch_score = (206.835 - 
                          (1.015 * len(words) / len(sentences)) - 
                          (84.6 * total_syllables / len(words)))
            
            # Normalize to 0-1 scale (0-100 Flesch scale to 0-1)
            return max(0.0, min(flesch_score / 100.0, 1.0))
        
        return 0.5  # Default for edge cases
    
    def _calculate_information_density(self, words: List[str]) -> float:
        """Shannon Entropy - information theory (1948)"""
        if len(words) == 0:
            return 0.0
        
        # Calculate word frequency distribution
        word_counts = Counter(word.lower() for word in words)
        total_words = len(words)
        
        # Shannon entropy calculation
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize (theoretical max entropy for this vocabulary size)
        max_entropy = math.log2(len(word_counts)) if len(word_counts) > 1 else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_structural_complexity(self, sentences: List[str]) -> float:
        """Sentence structure complexity metrics"""
        if len(sentences) == 0:
            return 0.0
        
        # Average sentence length (words per sentence)
        total_words = sum(len(sentence.split()) for sentence in sentences)
        avg_sentence_length = total_words / len(sentences)
        
        # Normalize sentence length complexity (research shows 15-20 words optimal)
        length_score = min(avg_sentence_length / 20.0, 1.0)
        
        # Punctuation complexity (indicates complex sentences)
        complex_punct = sum(sentence.count(',') + sentence.count(';') + sentence.count(':') 
                          for sentence in sentences)
        punct_score = min(complex_punct / len(sentences) / 3.0, 1.0)  # Normalize by 3 punctuations
        
        return (length_score + punct_score) / 2.0
    
    def _calculate_content_richness(self, text: str, tables: List[Dict]) -> float:
        """Content richness based on structured elements"""
        richness_score = 0.0
        
        # Tables indicate structured information
        if tables:
            richness_score += min(len(tables) * 0.2, 0.6)  # Up to 60% from tables
        
        # Technical terms (numbers, acronyms, technical language)
        numbers = len(re.findall(r'\b\d+\.?\d*\b', text))
        acronyms = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        words = text.split()
        if words:
            technical_density = (numbers + acronyms) / len(words)
            richness_score += min(technical_density * 2.0, 0.4)  # Up to 40% from technical content
        
        return min(richness_score, 1.0)
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting (linguistic approximation)"""
        word = word.lower().strip('.,!?;:')
        if len(word) == 0:
            return 0
        
        vowels = 'aeiouy'
        syllables = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                syllables += 1
            prev_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(syllables, 1)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_zero_quality_result(self) -> Dict:
        """Return zero quality for empty content"""
        return {
            'overall_quality': 0.0,
            'components': {
                'lexical_diversity': 0.0,
                'readability': 0.0,
                'information_density': 0.0,
                'structural_complexity': 0.0,
                'content_richness': 0.0
            },
            'methodology': 'research_based',
            'word_count': 0,
            'sentence_count': 0
        }


class EnhancedSuperiorHybridExtractor:
    """
    Enhanced extractor with quality assessment integration
    Industry-standard multi-library hybrid extraction
    """
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Initialize extractor with configuration
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.extraction_config = self.config['extraction']
        self.input_dir = Path(self.extraction_config['input_dir'])
        self.output_dir = Path(self.extraction_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality_assessor = ResearchBasedQualityAssessment()
        
        # Statistics tracking
        self.extraction_stats = {
            'total_documents': 0,
            'successful_extractions': 0,
            'total_tables': 0,
            'total_images': 0,
            'quality_scores': [],
            'component_scores': defaultdict(list)
        }
        
        logger.info("Hybrid Extractor Initialized")
        logger.info(f"Available Libraries: {[lib for lib, avail in libs_available.items() if avail]}")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def extract_comprehensive_content(self, pdf_path: str) -> Dict:
        """
        Extract PDF content with multi-library hybrid approach
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with comprehensive extraction results
        """
        if not Path(pdf_path).exists():
            return self._create_failed_content(pdf_path, "File not found")
        
        filename = Path(pdf_path).name
        
        content = {
            'file_path': pdf_path,
            'extraction_method': 'enhanced_superior_hybrid',
            'extraction_success': True,
            'pages': {},
            'tables': [],
            'images': [],
            'total_pages': 0,
            'privacy_guarantee': 'LOCAL_PROCESSING_ONLY'
        }
        
        try:
            # METHOD 1: PyMuPDF First (Most Reliable for structure)
            if libs_available['pymupdf']:
                content = self._extract_with_pymupdf_enhanced(pdf_path)
            
            # METHOD 2: Enhance with pdfplumber tables
            if libs_available['pdfplumber'] and content.get('extraction_success'):
                self._enhance_with_pdfplumber_tables(content, pdf_path)
            
            # METHOD 3: Unstructured as final enhancement (with full suppression)
            if libs_available['unstructured'] and content.get('extraction_success'):
                self._enhance_with_unstructured_quiet(content, pdf_path)
            
            # METHOD 4: QUALITY ASSESSMENT
            self._assess_document_quality(content)
            
            # Finalize
            self._finalize_content(content)
            self._save_content_locally(content)
            
            # Enhanced summary output with quality
            self._print_enhanced_summary(content, filename)
            
            return content
            
        except Exception as e:
            logger.error(f"❌ {filename}: Failed - {str(e)}")
            return self._create_failed_content(pdf_path, str(e))
    
    def _extract_with_pymupdf_enhanced(self, pdf_path: str) -> Dict:
        """
        Enhanced PyMuPDF extraction with comprehensive table/image detection
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extraction results dictionary
        """
        try:
            doc = fitz.open(pdf_path)
            
            content = {
                'file_path': pdf_path,
                'extraction_method': 'enhanced_pymupdf',
                'extraction_success': True,
                'pages': {},
                'tables': [],
                'images': [],
                'total_pages': len(doc),
                'metadata': doc.metadata or {},
                'privacy_guarantee': 'LOCAL_PROCESSING_ONLY'
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = self._extract_clean_text(page)
                
                # ENHANCED IMAGE DETECTION
                images = self._extract_images_comprehensive(page, page_num)
                
                # ENHANCED TABLE DETECTION - Multiple methods
                tables = []
                
                # Method 1: Native PyMuPDF tables
                tables.extend(self._extract_pymupdf_tables(page, page_num))
                
                # Method 2: Text pattern tables
                tables.extend(self._extract_text_pattern_tables(page, page_num))
                
                # Method 3: Grid-based detection
                tables.extend(self._extract_grid_tables(page, page_num))
                
                # Remove duplicates
                tables = self._deduplicate_tables(tables)
                
                # Store page data
                content['pages'][page_num] = {
                    'text': text,
                    'word_count': len(text.split()) if text else 0,
                    'tables': tables,
                    'table_count': len(tables),
                    'images': images,
                    'image_count': len(images),
                    'has_tables': len(tables) > 0,
                    'has_images': len(images) > 0
                }
                
                # Add to main collections
                content['tables'].extend(tables)
                content['images'].extend(images)
            
            doc.close()
            return content
            
        except Exception as e:
            return self._create_failed_content(pdf_path, str(e))
    
    def _extract_images_comprehensive(self, page, page_num: int) -> List[Dict]:
        """Comprehensive image detection using multiple methods"""
        images = []
        
        try:
            # Method 1: Standard images
            page_images = page.get_images(full=True)
            for img_idx, img in enumerate(page_images):
                images.append({
                    'image_id': len(images),
                    'page': page_num + 1,
                    'method': 'pymupdf_standard',
                    'xref': img[0],
                    'width': img[2] if len(img) > 2 else None,
                    'height': img[3] if len(img) > 3 else None,
                })
            
            # Method 2: Block-level image detection
            try:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block.get("type") == 1:  # Image block
                        bbox = block.get("bbox", [0, 0, 0, 0])
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        
                        if width > 10 and height > 10:  # Filter tiny images
                            # Check if this is a duplicate
                            is_duplicate = any(
                                abs(img.get('width', 0) - width) < 5 and 
                                abs(img.get('height', 0) - height) < 5 
                                for img in images
                            )
                            
                            if not is_duplicate:
                                images.append({
                                    'image_id': len(images),
                                    'page': page_num + 1,
                                    'method': 'pymupdf_blocks',
                                    'width': int(width),
                                    'height': int(height),
                                    'bbox': bbox
                                })
            except:
                pass
                
        except Exception:
            pass
        
        return images
    
    def _extract_pymupdf_tables(self, page, page_num: int) -> List[Dict]:
        """PyMuPDF native table detection with lower thresholds"""
        tables = []
        
        try:
            page_tables = page.find_tables()
            for table_idx, table in enumerate(page_tables):
                try:
                    table_data = table.extract()
                    if self._is_valid_table_permissive(table_data):
                        tables.append({
                            'table_id': len(tables),
                            'page': page_num + 1,
                            'method': 'pymupdf_native',
                            'data': table_data,
                            'rows': len(table_data),
                            'cols': len(table_data[0]) if table_data else 0,
                            'bbox': table.bbox,
                            'confidence': 'high'
                        })
                except:
                    continue
        except:
            pass
        
        return tables
    
    def _extract_text_pattern_tables(self, page, page_num: int) -> List[Dict]:
        """Extract tables from text patterns"""
        tables = []
        
        try:
            text_dict = page.get_text("dict")
            blocks = text_dict.get("blocks", [])
            
            for block in blocks:
                if "lines" in block:
                    lines = []
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span.get("text", "")
                        if line_text.strip():
                            lines.append(line_text)
                    
                    # Look for tabular patterns
                    table_data = self._extract_table_from_text_enhanced(lines)
                    if table_data and len(table_data) >= 2:
                        tables.append({
                            'table_id': len(tables),
                            'page': page_num + 1,
                            'method': 'text_pattern',
                            'data': table_data,
                            'rows': len(table_data),
                            'cols': len(table_data[0]) if table_data else 0,
                            'confidence': 'medium'
                        })
        except:
            pass
        
        return tables
    
    def _extract_grid_tables(self, page, page_num: int) -> List[Dict]:
        """Detect tables based on visual grid patterns"""
        tables = []
        
        try:
            drawings = page.get_drawings()
            
            h_lines = []
            v_lines = []
            
            for drawing in drawings:
                for item in drawing.get("items", []):
                    if item[0] == "l":  # Line
                        x1, y1, x2, y2 = item[1:5]
                        if abs(y1 - y2) < 2:  # Horizontal line
                            h_lines.append((x1, y1, x2, y2))
                        elif abs(x1 - x2) < 2:  # Vertical line
                            v_lines.append((x1, y1, x2, y2))
            
            # If we have grid lines, there might be a table
            if len(h_lines) >= 3 and len(v_lines) >= 3:
                bbox = self._calculate_grid_bbox(h_lines, v_lines)
                if bbox:
                    tables.append({
                        'table_id': len(tables),
                        'page': page_num + 1,
                        'method': 'grid_detection',
                        'data': [["Grid table detected"]],  # Placeholder
                        'rows': 1,
                        'cols': 1,
                        'bbox': bbox,
                        'confidence': 'low'
                    })
        except:
            pass
        
        return tables
    
    def _calculate_grid_bbox(self, h_lines, v_lines):
        """Calculate bounding box from grid lines"""
        try:
            if not h_lines or not v_lines:
                return None
            
            x_coords = [x for line in v_lines for x in [line[0], line[2]]]
            y_coords = [y for line in h_lines for y in [line[1], line[3]]]
            
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        except:
            return None
    
    def _extract_table_from_text_enhanced(self, lines: List[str]) -> List[List[str]]:
        """Enhanced table extraction from text lines"""
        if len(lines) < 2:
            return []
        
        table_rows = []
        
        for line in lines:
            potential_cells = None
            
            # Multiple separator strategies
            separators = [
                ('\t', 2),           # Tab separated
                ('|', 2),            # Pipe separated  
                ('  ', 3),           # Double space
                ('   ', 2),          # Triple space
            ]
            
            for sep, min_parts in separators:
                parts = line.split(sep)
                
                if len(parts) >= min_parts:
                    potential_cells = [cell.strip() for cell in parts if cell.strip()]
                    if len(potential_cells) >= 2:
                        break
            
            if potential_cells and len(potential_cells) >= 2:
                table_rows.append(potential_cells)
        
        # Validate table structure
        if len(table_rows) >= 2:
            col_counts = [len(row) for row in table_rows]
            avg_cols = sum(col_counts) / len(col_counts)
            
            filtered_rows = []
            for row in table_rows:
                if abs(len(row) - avg_cols) <= 1:  # Allow 1 column difference
                    filtered_rows.append(row)
            
            if len(filtered_rows) >= 2:
                return filtered_rows
        
        return []
    
    def _enhance_with_pdfplumber_tables(self, content: Dict, pdf_path: str):
        """Enhance with pdfplumber's superior table detection"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    strategies = [
                        {},  # Default
                        {
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                        },
                        {
                            "vertical_strategy": "text", 
                            "horizontal_strategy": "text",
                        },
                        {
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                            "edge_min_length": 3,
                        }
                    ]
                    
                    for strategy in strategies:
                        try:
                            tables = page.extract_tables(table_settings=strategy)
                            if tables:
                                for table_data in tables:
                                    if self._is_valid_table_permissive(table_data):
                                        # Check for duplicates
                                        if not self._is_duplicate_table_content(content['tables'], table_data, page_num + 1):
                                            new_table = {
                                                'table_id': len(content['tables']),
                                                'page': page_num + 1,
                                                'method': 'pdfplumber_enhanced',
                                                'data': table_data,
                                                'rows': len(table_data),
                                                'cols': len(table_data[0]) if table_data else 0,
                                                'confidence': 'high'
                                            }
                                            content['tables'].append(new_table)
                                            content['pages'][page_num]['tables'].append(new_table)
                                            content['pages'][page_num]['table_count'] += 1
                                            break  # Found table with this strategy
                        except:
                            continue
                            
        except Exception:
            pass
    
    def _enhance_with_unstructured_quiet(self, content: Dict, pdf_path: str):
        """Enhance with unstructured (fully suppressed)"""
        try:
            with SuppressOutput():
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy="fast",
                    infer_table_structure=True,
                )
                
                for element in elements:
                    try:
                        if isinstance(element, Table) or 'Table' in type(element).__name__:
                            page_num = 0
                            if hasattr(element, 'metadata') and element.metadata:
                                if hasattr(element.metadata, 'page_number'):
                                    page_num = element.metadata.page_number - 1
                            
                            element_text = str(element)
                            if element_text and len(element_text) > 50:  # Substantial content
                                existing_tables_on_page = [
                                    t for t in content['tables'] 
                                    if t['page'] == page_num + 1
                                ]
                                
                                if len(existing_tables_on_page) == 0:  # No tables found yet
                                    new_table = {
                                        'table_id': len(content['tables']),
                                        'page': page_num + 1,
                                        'method': 'unstructured_ai',
                                        'content': element_text,
                                        'confidence': 'medium'
                                    }
                                    content['tables'].append(new_table)
                                    if page_num in content['pages']:
                                        content['pages'][page_num]['tables'].append(new_table)
                                        content['pages'][page_num]['table_count'] += 1
                    except:
                        continue
                        
        except:
            pass  # Fail silently
    
    def _assess_document_quality(self, content: Dict):
        """Assess document quality using research-based metrics"""
        # Collect all text content
        all_text_parts = []
        
        for page_data in content.get('pages', {}).values():
            text = page_data.get('text', '')
            if text and text.strip():
                all_text_parts.append(text.strip())
        
        # Combine all text
        full_text = '\n\n'.join(all_text_parts)
        
        if not full_text.strip():
            # No text content
            content['quality_assessment'] = self.quality_assessor._create_zero_quality_result()
            return
        
        # Assess quality
        quality_result = self.quality_assessor.assess_content_quality(
            full_text, 
            content.get('tables', [])
        )
        
        content['quality_assessment'] = quality_result
        
        # Update statistics
        self.extraction_stats['quality_scores'].append(quality_result['overall_quality'])
        for component, score in quality_result['components'].items():
            self.extraction_stats['component_scores'][component].append(score)
    
    def _is_valid_table_permissive(self, table_data) -> bool:
        """More permissive table validation"""
        if not table_data or len(table_data) < 2:
            return False
        
        row_lengths = [len(row) for row in table_data if row]
        if not row_lengths or max(row_lengths) < 2:
            return False
        
        # Very permissive content check
        non_empty_cells = 0
        total_cells = sum(row_lengths[:3])  # Check first 3 rows only
        
        for row in table_data[:3]:
            for cell in row:
                if cell and str(cell).strip() and str(cell).strip().lower() not in ['', 'none', 'null']:
                    non_empty_cells += 1
        
        content_ratio = non_empty_cells / total_cells if total_cells > 0 else 0
        return content_ratio >= 0.15  # Only 15% content required
    
    def _is_duplicate_table_content(self, existing_tables: List[Dict], new_data, page_num: int) -> bool:
        """Check for duplicate table content"""
        page_tables = [t for t in existing_tables if t['page'] == page_num]
        
        new_rows = len(new_data)
        new_cols = len(new_data[0]) if new_data else 0
        
        for table in page_tables:
            if (table.get('rows') == new_rows and table.get('cols') == new_cols):
                return True
        
        return False
    
    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables"""
        unique_tables = []
        seen = set()
        
        for table in tables:
            key = (table['page'], table['rows'], table['cols'])
            if key not in seen:
                seen.add(key)
                unique_tables.append(table)
        
        return unique_tables
    
    def _extract_clean_text(self, page) -> str:
        """Extract clean text"""
        try:
            text = page.get_text()
            # Basic cleaning
            text = text.replace('\x00', '').replace('\ufffd', '')
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines)
        except:
            return ""
    
    def _finalize_content(self, content: Dict):
        """Finalize content with statistics"""
        content['images_detected'] = len(content.get('images', []))
        self._calculate_stats(content)
        
        # Update global statistics
        self.extraction_stats['total_documents'] += 1
        if content.get('extraction_success', False):
            self.extraction_stats['successful_extractions'] += 1
        self.extraction_stats['total_tables'] += len(content.get('tables', []))
        self.extraction_stats['total_images'] += len(content.get('images', []))
    
    def _calculate_stats(self, content: Dict):
        """Calculate basic stats"""
        total_words = sum(page.get('word_count', 0) for page in content['pages'].values())
        total_tables = len(content.get('tables', []))
        total_images = len(content.get('images', []))
        
        content['document_stats'] = {
            'total_words': total_words,
            'total_tables': total_tables,
            'total_images': total_images,
            'total_pages': content['total_pages']
        }
    
    def _print_enhanced_summary(self, content: Dict, filename: str):
        """Enhanced summary with quality metrics"""
        stats = content.get('document_stats', {})
        quality = content.get('quality_assessment', {})
        pages = content.get('total_pages', 0)
        tables = stats.get('total_tables', 0)
        images = stats.get('total_images', 0)
        
        # Build content summary
        summary_parts = [f"{pages}p"]
        
        if tables > 0:
            table_pages = sorted(list(set([t['page'] for t in content['tables']])))
            if len(table_pages) <= 5:
                table_info = f"📊{tables}({','.join(map(str, table_pages))})"
            else:
                table_info = f"📊{tables}(P{table_pages[0]}-{table_pages[-1]})"
            summary_parts.append(table_info)
        
        if images > 0:
            image_pages = sorted(list(set([i['page'] for i in content['images']])))
            if len(image_pages) <= 5:
                image_info = f"🖼️{images}({','.join(map(str, image_pages))})"
            else:
                image_info = f"🖼️{images}(P{image_pages[0]}-{image_pages[-1]})"
            summary_parts.append(image_info)
        
        if tables == 0 and images == 0:
            summary_parts.append("text-only")
        
        # Add quality information
        overall_quality = quality.get('overall_quality', 0)
        word_count = quality.get('word_count', 0)
        
        quality_grade = self._get_quality_grade(overall_quality)
        summary_parts.append(f"Q:{quality_grade}({overall_quality:.3f})")
        summary_parts.append(f"{word_count}w")
        
        logger.info(f"📄 {filename}: {' | '.join(summary_parts)}")
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        elif score >= 0.3:
            return "D"
        else:
            return "F"
    
    def _create_failed_content(self, pdf_path: str, error: str) -> Dict:
        """Create failed content placeholder"""
        return {
            'file_path': pdf_path,
            'extraction_success': False,
            'error': error,
            'pages': {},
            'tables': [],
            'images': [],
            'total_pages': 0,
            'quality_assessment': self.quality_assessor._create_zero_quality_result()
        }
    
    def _save_content_locally(self, content: Dict):
        """Save extracted content to JSON"""
        try:
            input_path = Path(content.get('file_path', 'unknown'))
            filename = input_path.stem + ".json"
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save content: {e}")
    
    def extract_all_documents(self) -> List[Dict]:
        """
        Extract all documents from input directory
        Pipeline-compatible main extraction method
        """
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("⚠️  No PDF files found in input directory")
            return []
        
        logger.info(f"Processing {len(pdf_files)} PDFs with quality assessment...")
        logger.info("Format: filename: pages | tables(pages) | images(pages) | Quality:Grade(score) | words")
        logger.info("─" * 80)
        
        all_content = []
        
        for i, pdf_file in enumerate(tqdm(pdf_files, desc="Extracting documents")):
            try:
                logger.info(f"[{i+1:2d}/{len(pdf_files)}] Processing...")
                content = self.extract_comprehensive_content(str(pdf_file))
                
                if content.get('extraction_success', False):
                    all_content.append(content)
                    
            except Exception as e:
                logger.error(f"{pdf_file.name}: Failed - {str(e)}")
        
        # Print final statistics
        self._print_final_statistics()
        
        return all_content
    
    def _print_final_statistics(self):
        """Print comprehensive final statistics"""
        stats = self.extraction_stats
        
        logger.info("─" * 80)
        logger.info("   FINAL EXTRACTION & QUALITY STATISTICS")
        logger.info(f"  Documents processed: {stats['successful_extractions']}/{stats['total_documents']}")
        logger.info(f"  Total tables found: {stats['total_tables']}")
        logger.info(f"  Total images found: {stats['total_images']}")
        
        if stats['quality_scores']:
            avg_quality = sum(stats['quality_scores']) / len(stats['quality_scores'])
            max_quality = max(stats['quality_scores'])
            min_quality = min(stats['quality_scores'])
            
            logger.info(f"      QUALITY METRICS:")
            logger.info(f"      Overall Average: {avg_quality:.3f} ({self._get_quality_grade(avg_quality)})")
            logger.info(f"      Highest Quality: {max_quality:.3f} ({self._get_quality_grade(max_quality)})")
            logger.info(f"      Lowest Quality:  {min_quality:.3f} ({self._get_quality_grade(min_quality)})")
            
            # Component averages
            logger.info(f"    COMPONENT AVERAGES:")
            for component, scores in stats['component_scores'].items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    comp_name = {
                        'lexical_diversity': 'Lexical Diversity',
                        'readability': 'Readability',
                        'information_density': 'Information Density',
                        'structural_complexity': 'Structural Complexity',
                        'content_richness': 'Content Richness'
                    }.get(component, component.title())
                    logger.info(f"      {comp_name}: {avg_score:.3f}")
        
        logger.info(f"    Data saved to: {self.output_dir}")
        logger.info("─" * 80)
        
        # Save statistics to JSON
        stats_file = self.output_dir / "extraction_stats.json"
        with open(stats_file, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            json_stats = dict(stats)
            json_stats['component_scores'] = dict(json_stats['component_scores'])
            json.dump(json_stats, f, indent=2)
        logger.info(f"📊 Statistics saved to: {stats_file}")


def main():
    """Main extraction function for standalone execution"""
    logger.info("   Hybrid PDF Extractor with Quality Assessment")
    logger.info("   Features: Tables, Images + Research-Based Quality Metrics")
    logger.info("   Clean output with comprehensive quality analysis")
    logger.info("=" * 80)
    
    try:
        extractor = EnhancedSuperiorHybridExtractor()
        results = extractor.extract_all_documents()
        return len(results) > 0
    except Exception as e:
        logger.error(f" Failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
