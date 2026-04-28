from pathlib import Path
from dataclasses import dataclass

@dataclass
class Chunk:
    """Represents one meaningful piece of code."""
    chunk_id: str
    file_path: str
    language: str
    chunk_type: str         #'function','class','module'
    name: str
    code: str
    start_line: int
    end_line: int

#lines that start with these mean a chunk is beginning
# boundaries per language
BOUNDARIES = {
    "py":   ("def ", "async def ", "class "),
    "js":   ("function ", "const ", "class ", "async function "),
    "ts":   ("function ", "const ", "class ", "async function ", "interface "),
    "java": ("public ", "private ", "protected ", "class ", "void ", "static "),
    "cpp":  ("void ", "int ", "bool ", "class ", "struct ", "auto "),
    "c":    ("void ", "int ", "bool ", "struct ", "static "),
    "go":   ("func ", "type ", "var "),
    "rb":   ("def ", "class ", "module "),
}

# fallback for unknown languages
DEFAULT_BOUNDARIES = ("def ", "class ", "function ")

def chunk_file(path: Path)->list[Chunk]:
    """
    Split one code file into chunks by function/class boundaries.
    Falls back to whole file as one chunk if no boundaries found.
    """

    try:
        code = path.read_text(encoding="utf-8",errors="replace")
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return []
    
    lines = code.splitlines()
    language = path.suffix.lstrip(".")

    current_name = f"module:{path.stem}"
    current_type = "module"
    current_start = 0
    current_lines = []
    language = path.suffix.lstrip(".")
    lang_boundaries = BOUNDARIES.get(language, DEFAULT_BOUNDARIES)

    chunks = []
    for i,line in enumerate(lines):
        stripped = line.strip()
        is_boundary = any(stripped.startswith(b) for b in lang_boundaries)

        if is_boundary and current_lines:
            _flush(chunks,path,language,current_name,
                   current_type,current_lines,current_start,i-1)
            
            #Reset for new chunk
            current_start = i
            current_lines = []
            current_type = "class" if stripped.startswith("class") else "function"
            current_name = stripped.split("(")[0].split(" ")[-1].rstrip(":")
        else:
            current_lines.append(line)

    if current_lines:
        _flush(chunks,path,language,current_name,
               current_type,current_lines,current_start,len(lines)-1)
    return chunks

def _flush(chunks,path,language,name,chunk_type,lines,start,end):
    """
    Save accumulated lines as a Chunk — but only if it has
    meaningful content (not just blank lines or tiny snippets).
    """

    code = "\n".join(lines).strip()
    if len(code)<30:
        return  #skip tiny chunks

    chunks.append(Chunk(
        chunk_id=f"{path}:{start}",
         file_path  = str(path),
        language   = language,
        chunk_type = chunk_type,
        name       = name,
        code       = code,
        start_line = start,
        end_line   = end,
    ))

def chunk_files(files: list[Path])->list[Chunk]:
    """Chunk all files and return one combined list of chunks."""
    all_chunks = []
    for f in files:
        chunks = chunk_file(f)
        all_chunks.extend(chunks)
        print(f"  {f.name:30}->{len(chunks)} chunks")
    return all_chunks