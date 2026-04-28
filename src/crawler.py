from pathlib import Path
from src.config import CODE_EXTENSIONS,SKIP_DIRS
import os

def crawl_repo(repo_path: str)->list[Path]:
    """
    Walk a repo folder and return all code files,
    skipping irrelevant directories"""

    repo = Path(repo_path)
    if not repo.exists():
        raise ValueError(f"Repo path {repo_path} does not exist")
    if not repo.is_dir():
        raise ValueError(f"Repo path {repo_path} is not a directory")
    found_files = []

    for root,dirs,files in os.walk(repo):
        # Remove skip dirs IN PLACE so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in files:
            filepath  = Path(root)/filename
            if filepath.suffix in CODE_EXTENSIONS:
                found_files.append(filepath)
        
    return found_files

def summarize_crawl(files: list[Path])->None:
    """Print summary of what was found"""
    from collections import Counter
    ext__counts = Counter(f.suffix for f in files)
    print(f"\nFound {len(files)} code files:")
    for ext,count in ext__counts.most_common():
        print(f"  {ext:10}: {count} files")
