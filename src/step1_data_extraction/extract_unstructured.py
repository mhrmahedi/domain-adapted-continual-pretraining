"""#This version has some error in Linux version, need to fix some issues
Step 1: Document Extraction Using Unstructured Library Only
Optimized for text extraction with table structure inference
"""

import os
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import setup_logger


# Import Unstructured
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text

logger = setup_logger("unstructured_extractor")


class UnstructuredExtractor:
    """
    Simple, fast extractor using only Unstructured library
    Optimized for text quality with table structure preservation
    """
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize extractor with configuration"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.extraction_config = self.config['extraction']
        self.input_dir = Path(self.extraction_config['input_dir'])
        self.output_dir = Path(self.extraction_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_elements': 0
        }
        
        logger.info(" Unstructured Extractor Initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def extract_from_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """
        Extract text from PDF using Unstructured
        Uses 'fast' strategy for speed without external dependencies
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with extracted content or None if failed
        """
        try:
            # Extract elements from PDF
            # strategy="fast" - fastest, no OCR or heavy dependencies
            # strategy="hi_res" - better quality but slower (use if you need OCR)
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="fast",                    # Fast extraction without OCR
                infer_table_structure=True,         # Detect and preserve tables
                extract_images_in_pdf=False         # Skip images for speed
            )
            
            # Convert elements to text
            text = "\n\n".join([str(el) for el in elements])
            
            # Extract tables separately
            tables = []
            for el in elements:
                if hasattr(el, 'metadata') and el.metadata.text_as_html:
                    tables.append({
                        'content': el.metadata.text_as_html,
                        'type': 'table'
                    })
            
            self.stats['successful'] += 1
            self.stats['total_elements'] += len(elements)
            
            return {
                'filename': pdf_path.name,
                'text': text,
                'tables': tables,
                'element_count': len(elements),
                'source_type': 'pdf',
                'extractor': 'unstructured'
            }
            
        except Exception as e:
            logger.error(f"Failed to extract {pdf_path.name}: {e}")
            self.stats['failed'] += 1
            return None
    
    def extract_from_docx(self, docx_path: Path) -> Optional[Dict]:
        """Extract text from DOCX using Unstructured"""
        try:
            elements = partition_docx(filename=str(docx_path))
            text = "\n\n".join([str(el) for el in elements])
            
            self.stats['successful'] += 1
            self.stats['total_elements'] += len(elements)
            
            return {
                'filename': docx_path.name,
                'text': text,
                'tables': [],
                'element_count': len(elements),
                'source_type': 'docx',
                'extractor': 'unstructured'
            }
            
        except Exception as e:
            logger.error(f"Failed to extract {docx_path.name}: {e}")
            self.stats['failed'] += 1
            return None
    
    def extract_from_html(self, html_path: Path) -> Optional[Dict]:
        """Extract text from HTML using Unstructured"""
        try:
            elements = partition_html(filename=str(html_path))
            text = "\n\n".join([str(el) for el in elements])
            
            self.stats['successful'] += 1
            self.stats['total_elements'] += len(elements)
            
            return {
                'filename': html_path.name,
                'text': text,
                'tables': [],
                'element_count': len(elements),
                'source_type': 'html',
                'extractor': 'unstructured'
            }
            
        except Exception as e:
            logger.error(f"Failed to extract {html_path.name}: {e}")
            self.stats['failed'] += 1
            return None
    
    def extract_from_txt(self, txt_path: Path) -> Optional[Dict]:
        """Extract text from TXT using Unstructured"""
        try:
            elements = partition_text(filename=str(txt_path))
            text = "\n\n".join([str(el) for el in elements])
            
            self.stats['successful'] += 1
            self.stats['total_elements'] += len(elements)
            
            return {
                'filename': txt_path.name,
                'text': text,
                'tables': [],
                'element_count': len(elements),
                'source_type': 'txt',
                'extractor': 'unstructured'
            }
            
        except Exception as e:
            logger.error(f"Failed to extract {txt_path.name}: {e}")
            self.stats['failed'] += 1
            return None
    
    def extract_all_documents(self) -> List[Dict]:
        """
        Extract from all supported document types in input directory
        
        Returns:
            List of extraction results
        """
        all_results = []
        
        # Gather all files
        pdf_files = list(self.input_dir.glob("*.pdf"))
        docx_files = list(self.input_dir.glob("*.docx"))
        html_files = list(self.input_dir.glob("*.html"))
        txt_files = list(self.input_dir.glob("*.txt"))
        
        all_files = pdf_files + docx_files + html_files + txt_files
        self.stats['total_files'] = len(all_files)
        
        if not all_files:
            logger.warning(f" No documents found in {self.input_dir}")
            return all_results
        
        logger.info(f"\n   Found {len(all_files)} documents:")
        logger.info(f"   - PDFs: {len(pdf_files)}")
        logger.info(f"   - DOCX: {len(docx_files)}")
        logger.info(f"   - HTML: {len(html_files)}")
        logger.info(f"   - TXT: {len(txt_files)}")
        
        # Process each file type with progress bar
        for pdf_file in tqdm(pdf_files, desc=" Processing PDFs"):
            result = self.extract_from_pdf(pdf_file)
            if result:
                all_results.append(result)
        
        for docx_file in tqdm(docx_files, desc=" Processing DOCX"):
            result = self.extract_from_docx(docx_file)
            if result:
                all_results.append(result)
        
        for html_file in tqdm(html_files, desc=" Processing HTML"):
            result = self.extract_from_html(html_file)
            if result:
                all_results.append(result)
        
        for txt_file in tqdm(txt_files, desc=" Processing TXT"):
            result = self.extract_from_txt(txt_file)
            if result:
                all_results.append(result)
        
        # Save results
        self._save_results(all_results)
        self._print_statistics()
        
        return all_results
    
    def _save_results(self, results: List[Dict]):
        """Save extraction results to JSONL file"""
        if not results:
            logger.warning("No results to save")
            return
        
        output_file = self.output_dir / "extracted_data.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"\n Saved {len(results)} documents to {output_file}")
    
    def _print_statistics(self):
        """Print extraction statistics"""
        logger.info("\n" + "="*80)
        logger.info(" EXTRACTION STATISTICS")
        logger.info("="*80)
        logger.info(f"Total files found:       {self.stats['total_files']}")
        logger.info(f"Successfully extracted:  {self.stats['successful']}")
        logger.info(f"Failed extractions:      {self.stats['failed']}")
        logger.info(f"Total elements extracted: {self.stats['total_elements']}")
        
        if self.stats['successful'] > 0:
            avg_elements = self.stats['total_elements'] / self.stats['successful']
            logger.info(f"Average elements/doc:    {avg_elements:.1f}")
        
        success_rate = (self.stats['successful'] / self.stats['total_files'] * 100) if self.stats['total_files'] > 0 else 0
        logger.info(f"Success rate:            {success_rate:.1f}%")
        logger.info("="*80)


def main():
    """Main extraction function for standalone execution"""
    logger.info(" Unstructured-Only Document Extractor")
    logger.info(" Fast text extraction with table structure preservation")
    logger.info("=" * 80)
    
    try:
        extractor = UnstructuredExtractor()
        results = extractor.extract_all_documents()
        
        if results:
            logger.info(f"\n Extraction complete! Processed {len(results)} documents")
            return True
        else:
            logger.warning("\n No documents were successfully extracted")
            return False
            
    except Exception as e:
        logger.error(f" Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
