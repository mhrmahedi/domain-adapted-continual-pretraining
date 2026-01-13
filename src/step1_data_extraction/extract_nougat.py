"""
#This version has some error in Linux version, need to fix some issues
Step 1: PDF Extraction Using Meta's Nougat Model (Local, No API)
Nougat: Neural Optical Understanding for Academic Documents
Perfect for scientific papers with equations, tables, and complex layouts
"""

import os
import json
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch

# Add project root to path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import setup_logger
logger = setup_logger("nougat_extractor")

class NougatExtractor:
    """
    Local Nougat-based PDF extractor (no API calls)
    Runs Meta's Nougat model entirely on your machine
    Excellent for academic/scientific documents with math and tables
    """
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize Nougat extractor"""
        # Load configuration
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.extraction_config = self.config.get('extraction', {})
        self.input_dir = Path(self.extraction_config.get('input_dir', 'data/raw'))
        self.output_dir = Path(self.extraction_config.get('output_dir', 'data/extracted'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nougat-specific output directory
        self.nougat_output_dir = self.output_dir / "nougat_markdown"
        self.nougat_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if Nougat is installed
        self._check_nougat_installation()
        
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0
        }
        
        logger.info("🥐 Nougat Extractor Initialized (Local, No API)")
        logger.info(f"Device: {self.device}")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _check_nougat_installation(self):
        """Check if Nougat is installed"""
        try:
            result = subprocess.run(
                ['nougat', '--version'],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0:
                logger.info("✅ Nougat is installed")
            else:
                logger.warning("⚠️  Nougat may not be installed correctly")
        except FileNotFoundError:
            logger.error("❌ Nougat is not installed!")
            logger.error("Install with: pip install nougat-ocr")
            raise RuntimeError("Nougat not installed. Run: pip install nougat-ocr")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Nougat check timed out")
        except Exception as e:
            logger.warning(f"Could not verify Nougat installation: {e}")
    
    def extract_from_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """
        Extract text from PDF using Nougat (local inference)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content or None if failed
        """
        try:
            # Output file will be created by Nougat
            output_stem = pdf_path.stem
            output_file = self.nougat_output_dir / f"{output_stem}.mmd"
            
            # Build Nougat command (runs locally, no API)
            nougat_cmd = [
                'nougat',
                str(pdf_path),  # Input PDF
                '-o', str(self.nougat_output_dir),  # Output directory
                '--markdown',  # Output in markdown format
            ]
            
            # Add CUDA flag if available
            if self.device == "cuda":
                nougat_cmd.extend(['--batchsize', '4'])  # Batch size for GPU
            else:
                nougat_cmd.extend(['--batchsize', '1'])  # Smaller batch for CPU
            
            logger.info(f"Processing {pdf_path.name} with Nougat...")
            
            # Run Nougat (local inference, no internet required after model download)
            result = subprocess.run(
                nougat_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per PDF
                check=False,
                env=os.environ.copy()  # Pass environment variables
            )
            
            if result.returncode != 0:
                logger.error(f"Nougat failed for {pdf_path.name}: {result.stderr}")
                self.stats['failed'] += 1
                return None
            
            # Read the generated markdown file
            if not output_file.exists():
                logger.error(f"Output file not created: {output_file}")
                self.stats['failed'] += 1
                return None
            
            with open(output_file, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            
            # Parse tables and equations from markdown
            tables = self._extract_tables_from_markdown(markdown_text)
            equations = self._extract_equations_from_markdown(markdown_text)
            
            self.stats['successful'] += 1
            
            return {
                'filename': pdf_path.name,
                'text': markdown_text,
                'tables': tables,
                'equations': equations,
                'equation_count': len(equations),
                'table_count': len(tables),
                'markdown_file': str(output_file),
                'source_type': 'pdf',
                'extractor': 'nougat',
                'device': self.device
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout processing {pdf_path.name}")
            self.stats['failed'] += 1
            return None
        except Exception as e:
            logger.error(f"Failed to extract {pdf_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.stats['failed'] += 1
            return None
    
    def _extract_tables_from_markdown(self, markdown_text: str) -> List[Dict]:
        """Extract tables from Nougat's markdown output"""
        tables = []
        # Nougat outputs tables in markdown format
        # Pattern: | col1 | col2 | ... |
        lines = markdown_text.split('\n')
        current_table = []
        in_table = False
        
        for line in lines:
            if line.strip().startswith('|') and line.strip().endswith('|'):
                current_table.append(line.strip())
                in_table = True
            elif in_table and current_table:
                # Table ended
                tables.append({
                    'content': '\n'.join(current_table),
                    'row_count': len(current_table) - 1  # Minus header separator
                })
                current_table = []
                in_table = False
        
        # Add last table if exists
        if current_table:
            tables.append({
                'content': '\n'.join(current_table),
                'row_count': len(current_table) - 1
            })
        
        return tables
    
    def _extract_equations_from_markdown(self, markdown_text: str) -> List[Dict]:
        """Extract LaTeX equations from Nougat's markdown output"""
        equations = []
        # Nougat preserves LaTeX equations in $...$ or $$...$$
        import re
        
        # Extract inline equations $...$
        inline_eqs = re.findall(r'\$([^\$]+)\$', markdown_text)
        for eq in inline_eqs:
            equations.append({
                'type': 'inline',
                'latex': eq.strip()
            })
        
        # Extract display equations $$...$$
        display_eqs = re.findall(r'\$\$([^\$]+)\$\$', markdown_text)
        for eq in display_eqs:
            equations.append({
                'type': 'display',
                'latex': eq.strip()
            })
        
        return equations
    
    def extract_all_pdfs(self) -> List[Dict]:
        """
        Extract from all PDFs in input directory using Nougat
        
        Returns:
            List of extraction results
        """
        pdf_files = list(self.input_dir.glob("*.pdf"))
        self.stats['total_files'] = len(pdf_files)
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {self.input_dir}")
            return []
        
        logger.info(f"\n📁 Found {len(pdf_files)} PDF files to process with Nougat")
        logger.info("⚠️  Note: First run will download Nougat model (~1.5GB)")
        logger.info("📡 All processing is LOCAL - no data sent to external APIs\n")
        
        all_results = []
        
        # Process each PDF with progress bar
        for pdf_file in tqdm(pdf_files, desc="🥐 Processing PDFs with Nougat"):
            result = self.extract_from_pdf(pdf_file)
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
        
        output_file = self.output_dir / "nougat_extracted_data.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"\n💾 Saved {len(results)} documents to {output_file}")
        logger.info(f"📄 Markdown files saved to {self.nougat_output_dir}")
    
    def _print_statistics(self):
        """Print extraction statistics"""
        logger.info("\n" + "="*80)
        logger.info("📊 NOUGAT EXTRACTION STATISTICS")
        logger.info("="*80)
        logger.info(f"Total PDFs found:        {self.stats['total_files']}")
        logger.info(f"Successfully extracted:  {self.stats['successful']}")
        logger.info(f"Failed extractions:      {self.stats['failed']}")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_files']) * 100
            logger.info(f"Success rate:            {success_rate:.1f}%")
        
        logger.info(f"Device used:             {self.device}")
        logger.info(f"Processing mode:         LOCAL (no API calls)")
        logger.info("="*80)


def main():
    """Main extraction function for standalone execution"""
    logger.info("🥐 Nougat PDF Extractor (Meta AI)")
    logger.info("🎯 Perfect for academic/scientific documents")
    logger.info("🔒 100% Local - No API calls, no data leaves your machine")
    logger.info("=" * 80)
    
    try:
        extractor = NougatExtractor()
        results = extractor.extract_all_pdfs()
        
        if results:
            logger.info(f"\n✅ Extraction complete! Processed {len(results)} PDFs")
            logger.info("📄 Output format: Markdown with preserved LaTeX equations")
            return True
        else:
            logger.warning("\n⚠️  No documents were successfully extracted")
            return False
            
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
