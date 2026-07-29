"""Adaptive chunking: different strategy per file type.

Python      -> AST-based (function/class boundaries)
Markdown    -> heading-based
YAML/JSON   -> whole-file (usually small/structured; naive splitting breaks parsing)
everything else -> fixed-size line window with overlap
"""
import ast
from dataclasses import dataclass, field
from typing import List, Optional

FIXED_WINDOW_LINES = 60
FIXED_WINDOW_OVERLAP = 10

# tree-sitter grammar name + the node types that count as a "definition" for
# each language we support beyond Python's native ast module.
TREE_SITTER_LANGUAGES = {
    "java": ("java", ["class_declaration", "interface_declaration", "method_declaration"]),
    "go": ("go", ["function_declaration", "method_declaration", "type_declaration"]),
    "rust": ("rust", ["function_item", "impl_item", "struct_item", "enum_item", "trait_item"]),
    "typescript": ("typescript", ["function_declaration", "class_declaration", "method_definition", "interface_declaration"]),
    "javascript": ("javascript", ["function_declaration", "class_declaration", "method_definition"]),
    "cpp": ("cpp", ["function_definition", "class_specifier", "struct_specifier"]),
    "c": ("c", ["function_definition", "struct_specifier"]),
}


@dataclass
class Chunk:
    repo: str
    file: str
    language: str
    content: str
    start_line: int
    end_line: int
    symbol: Optional[str] = None       # function/class/heading name
    kind: str = "generic"              # function | class | heading | generic
    imports: List[str] = field(default_factory=list)
    commit_sha: Optional[str] = None   # last commit that touched this file
    author: Optional[str] = None       # last-commit author
    branch: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.file}:{self.start_line}-{self.end_line}"


def _python_imports(tree: ast.Module) -> List[str]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def chunk_python(repo: str, file: str, content: str) -> List[Chunk]:
    """AST-based: one chunk per top-level function/class, whole-file fallback on parse error."""
    lines = content.splitlines()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return chunk_generic(repo, file, "python", content)

    imports = _python_imports(tree)
    chunks: List[Chunk] = []
    top_level_nodes = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    if not top_level_nodes:
        return chunk_generic(repo, file, "python", content)

    for node in top_level_nodes:
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        # include decorators
        if node.decorator_list:
            start = min(d.lineno for d in node.decorator_list)
        snippet = "\n".join(lines[start - 1:end])
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        chunks.append(Chunk(
            repo=repo, file=file, language="python", content=snippet,
            start_line=start, end_line=end, symbol=node.name, kind=kind,
            imports=imports,
        ))

    # module-level code (imports, constants, docstring) not covered above
    covered = set()
    for n in top_level_nodes:
        covered.update(range(n.lineno, getattr(n, "end_lineno", n.lineno) + 1))
    remaining_lines = [i + 1 for i in range(len(lines)) if (i + 1) not in covered]
    if remaining_lines:
        module_snippet = "\n".join(
            lines[i - 1] for i in remaining_lines if lines[i - 1].strip()
        )
        if module_snippet.strip():
            chunks.insert(0, Chunk(
                repo=repo, file=file, language="python", content=module_snippet,
                start_line=1, end_line=remaining_lines[-1], symbol="<module>",
                kind="module", imports=imports,
            ))
    return chunks


def chunk_markdown(repo: str, file: str, content: str) -> List[Chunk]:
    """Heading-based split."""
    lines = content.splitlines()
    chunks: List[Chunk] = []
    current_heading = "<intro>"
    buf: List[str] = []
    start = 1

    def flush(end_line: int):
        if buf and "\n".join(buf).strip():
            chunks.append(Chunk(
                repo=repo, file=file, language="markdown",
                content="\n".join(buf), start_line=start, end_line=end_line,
                symbol=current_heading, kind="heading",
            ))

    for i, line in enumerate(lines, start=1):
        if line.startswith("#"):
            flush(i - 1)
            current_heading = line.lstrip("#").strip() or "<section>"
            buf = [line]
            start = i
        else:
            buf.append(line)
    flush(len(lines))
    return chunks or chunk_generic(repo, file, "markdown", content)


def chunk_tree_sitter(repo: str, file: str, language: str, content: str) -> List[Chunk]:
    """AST-based chunking for Java/Go/Rust/TypeScript/JavaScript/C/C++ using
    tree-sitter-languages (prebuilt grammars, no compiler toolchain needed).
    Falls back to generic windowing if the grammar isn't available or parsing fails.
    """
    entry = TREE_SITTER_LANGUAGES.get(language)
    if not entry:
        return chunk_generic(repo, file, language, content)
    grammar_name, def_node_types = entry

    try:
        import tree_sitter_languages as tsl
        parser = tsl.get_parser(grammar_name)
    except Exception:
        return chunk_generic(repo, file, language, content)

    try:
        tree = parser.parse(content.encode("utf-8"))
    except Exception:
        return chunk_generic(repo, file, language, content)

    lines = content.splitlines()
    chunks: List[Chunk] = []

    def node_name(node) -> Optional[str]:
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "field_identifier"):
                return content.encode("utf-8")[child.start_byte:child.end_byte].decode("utf-8", "ignore")
        return None

    def walk(node, depth=0):
        if node.type in def_node_types and depth <= 3:  # avoid over-nesting into every inner block
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            snippet = "\n".join(lines[start_line - 1:end_line])
            if snippet.strip():
                chunks.append(Chunk(
                    repo=repo, file=file, language=language, content=snippet,
                    start_line=start_line, end_line=end_line,
                    symbol=node_name(node), kind=node.type,
                ))
            return  # don't descend further into an already-captured definition
        for child in node.children:
            walk(child, depth + 1)

    walk(tree.root_node)

    if not chunks:
        return chunk_generic(repo, file, language, content)
    return chunks


def chunk_generic(repo: str, file: str, language: str, content: str) -> List[Chunk]:
    """Fixed-size sliding window with overlap; used as fallback and for config/text files."""
    lines = content.splitlines()
    if not lines:
        return []
    chunks: List[Chunk] = []
    i = 0
    step = FIXED_WINDOW_LINES - FIXED_WINDOW_OVERLAP
    while i < len(lines):
        window = lines[i:i + FIXED_WINDOW_LINES]
        if not any(l.strip() for l in window):
            i += step
            continue
        chunks.append(Chunk(
            repo=repo, file=file, language=language,
            content="\n".join(window), start_line=i + 1,
            end_line=min(i + FIXED_WINDOW_LINES, len(lines)), kind="generic",
        ))
        i += step
    return chunks


def chunk_file(repo: str, file: str, language: str, content: str) -> List[Chunk]:
    """Adaptive dispatch by language/file type."""
    if language == "python":
        return chunk_python(repo, file, content)
    if language == "markdown":
        return chunk_markdown(repo, file, content)
    if language in TREE_SITTER_LANGUAGES:
        return chunk_tree_sitter(repo, file, language, content)
    # YAML/JSON/others: whole-file if small, else generic windowing
    if language in ("yaml", "json") and len(content.splitlines()) <= FIXED_WINDOW_LINES:
        lines = content.splitlines()
        return [Chunk(
            repo=repo, file=file, language=language, content=content,
            start_line=1, end_line=len(lines), kind="whole_file",
        )]
    return chunk_generic(repo, file, language, content)
