#!/usr/bin/env python3
"""
Generate .pyi stub files from Jupyter notebooks.
Extracts function definitions and creates type hint files.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Tuple


def extract_functions_from_notebook(notebook_path: str) -> List[Tuple[str, str]]:
    """
    Extract function definitions from a Jupyter notebook.
    
    Returns a list of (function_name, function_signature) tuples.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    functions = []
    
    # Extract from code cells
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            
            # Find all function definitions
            # Pattern: def function_name(args):
            pattern = r'^def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*([^:]+))?:'
            
            for match in re.finditer(pattern, source, re.MULTILINE):
                func_name = match.group(1)
                params = match.group(2)
                return_type = match.group(3)
                
                if 'test_' in func_name:
                    continue

                # Build signature
                if return_type:
                    return_type = return_type.strip()
                    signature = f"def {func_name}({params}) -> {return_type}: ..."
                else:
                    signature = f"def {func_name}({params}): ..."
                
                functions.append((func_name, signature))
    
    return functions


def generate_pyi_file(notebook_path: str, output_dir: str = None) -> str:
    """
    Generate a .pyi file from a notebook.
    
    Returns the path to the generated .pyi file.
    """
    if output_dir is None:
        output_dir = os.path.dirname(notebook_path)
    
    notebook_name = Path(notebook_path).stem
    pyi_path = os.path.join(output_dir, f"{notebook_name}.pyi")
    
    functions = extract_functions_from_notebook(notebook_path)
    
    # Generate .pyi content
    lines = [
        f"# Stub file for {notebook_name}.ipynb",
        "# Auto-generated from notebook code cells",
        "",
    ]
    
    # Add imports that are commonly needed
    lines.append("from typing import Any, Callable, Iterable, List, Optional, Tuple, Union")
    lines.append("")
    
    # Add function signatures
    for func_name, signature in functions:
        lines.append(signature)
    
    content = "\n".join(lines)
    
    # Write .pyi file
    with open(pyi_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return pyi_path


def process_notebook(notebook_path: str) -> List[str]:
    notebook_path = Path(notebook_path)
    print(f"Processing {notebook_path.name}...")
    
    try:
        pyi_path = generate_pyi_file(str(notebook_path))
        print(f"  ✓ Generated {Path(pyi_path).name}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    

if __name__ == "__main__":
    import sys
    
    generated = process_notebook(sys.argv[1])
    