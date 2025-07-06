"""
Parser module for document and code parsing
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from .models import DocumentSection, CodeSymbol


class BaseParser(ABC):
    """Base class for all parsers"""

    def __init__(self):
        self.supported_extensions = []

    @abstractmethod
    def parse(self, file_path: Path, content: str) -> Union[List[DocumentSection], List[CodeSymbol]]:
        """Parse file content and return structured data"""
        pass

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file"""
        return file_path.suffix.lower() in self.supported_extensions


class MarkdownParser(BaseParser):
    """Parser for Markdown documents"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.md', '.markdown', '.mdown', '.mkd']

    def parse(self, file_path: Path, content: str) -> List[DocumentSection]:
        """Parse markdown content into sections"""
        lines = content.split('\n')
        sections = []
        current_section = None
        section_counter = 0

        # Stack to track nested sections
        section_stack = []

        for line_num, line in enumerate(lines, 1):
            # Check for markdown headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Close current section if exists
                if current_section:
                    current_section['end_line'] = line_num - 1
                    current_section['content'] = '\n'.join(lines[current_section['start_line']-1:current_section['end_line']])
                    sections.append(self._create_section(current_section, file_path))

                # Update section stack based on level
                while section_stack and section_stack[-1]['level'] >= level:
                    section_stack.pop()

                # Create new section
                section_id = f"{file_path.stem}_{line_num}_{level}"
                parent_id = section_stack[-1]['id'] if section_stack else None

                current_section = {
                    'id': section_id,
                    'title': title,
                    'level': level,
                    'start_line': line_num,
                    'end_line': len(lines),  # Will be updated when section ends
                    'parent_id': parent_id,
                    'children': []
                }

                # Add to parent's children
                if parent_id:
                    for section in sections:
                        if section.id == parent_id:
                            section.children.append(section_id)
                            break

                # Add to stack
                section_stack.append(current_section)
                section_counter += 1

        # Close final section
        if current_section:
            current_section['end_line'] = len(lines)
            current_section['content'] = '\n'.join(lines[current_section['start_line']-1:current_section['end_line']])
            sections.append(self._create_section(current_section, file_path))

        return sections

    def _create_section(self, section_data: Dict[str, Any], file_path: Path) -> DocumentSection:
        """Create DocumentSection from parsed data"""
        return DocumentSection(
            id=section_data['id'],
            title=section_data['title'],
            level=section_data['level'],
            start_line=section_data['start_line'],
            end_line=section_data['end_line'],
            content=section_data['content'],
            parent_id=section_data['parent_id'],
            children=section_data['children'],
            file_path=str(file_path)
        )


class RestructuredTextParser(BaseParser):
    """Parser for RestructuredText documents"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.rst', '.rest']

    def parse(self, file_path: Path, content: str) -> List[DocumentSection]:
        """Parse RST content into sections"""
        lines = content.split('\n')
        sections = []
        current_section = None
        section_counter = 0

        # Common RST header patterns
        header_patterns = [
            r'^=+$',  # = for titles
            r'^-+$',  # - for sections
            r'^\^+$', # ^ for subsections
            r'^"+$',  # " for subsubsections
        ]

        for line_num, line in enumerate(lines, 1):
            # Check if this line could be a header underline
            if line_num < len(lines) and any(re.match(pattern, line.strip()) for pattern in header_patterns):
                # Check if previous line exists and could be a title
                if line_num > 1:
                    prev_line = lines[line_num - 2].strip()
                    if prev_line and len(prev_line) <= len(line.strip()):
                        # This is likely a header
                        title = prev_line
                        level = self._get_rst_level(line.strip())

                        # Close current section
                        if current_section:
                            current_section['end_line'] = line_num - 2
                            current_section['content'] = '\n'.join(lines[current_section['start_line']-1:current_section['end_line']])
                            sections.append(self._create_section(current_section, file_path))

                        # Create new section
                        section_id = f"{file_path.stem}_{line_num}_{level}"
                        current_section = {
                            'id': section_id,
                            'title': title,
                            'level': level,
                            'start_line': line_num - 1,
                            'end_line': len(lines),
                            'parent_id': None,
                            'children': []
                        }
                        section_counter += 1

        # Close final section
        if current_section:
            current_section['end_line'] = len(lines)
            current_section['content'] = '\n'.join(lines[current_section['start_line']-1:current_section['end_line']])
            sections.append(self._create_section(current_section, file_path))

        return sections

    def _get_rst_level(self, underline: str) -> int:
        """Determine RST header level based on underline character"""
        if underline.startswith('='):
            return 1
        elif underline.startswith('-'):
            return 2
        elif underline.startswith('^'):
            return 3
        elif underline.startswith('"'):
            return 4
        else:
            return 5

    def _create_section(self, section_data: Dict[str, Any], file_path: Path) -> DocumentSection:
        """Create DocumentSection from parsed data"""
        return DocumentSection(
            id=section_data['id'],
            title=section_data['title'],
            level=section_data['level'],
            start_line=section_data['start_line'],
            end_line=section_data['end_line'],
            content=section_data['content'],
            parent_id=section_data['parent_id'],
            children=section_data['children'],
            file_path=str(file_path)
        )


class PythonParser(BaseParser):
    """Parser for Python code"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.py', '.pyw', '.pyi']

    def parse(self, file_path: Path, content: str) -> List[CodeSymbol]:
        """Parse Python code and extract symbols"""
        symbols = []

        try:
            # Parse using AST
            tree = ast.parse(content)

            for node in ast.walk(tree):
                symbol = self._extract_symbol(node, file_path, content)
                if symbol:
                    symbols.append(symbol)

        except SyntaxError as e:
            # Fallback to regex parsing for invalid syntax
            symbols = self._regex_parse_python(content, file_path)

        return symbols

    def _extract_symbol(self, node: ast.AST, file_path: Path, content: str) -> Optional[CodeSymbol]:
        """Extract symbol from AST node"""
        if isinstance(node, ast.FunctionDef):
            return CodeSymbol(
                name=node.name,
                type='function',
                file_path=str(file_path),
                line_number=node.lineno,
                column=node.col_offset,
                scope=self._get_scope(node),
                signature=self._get_function_signature(node),
                docstring=ast.get_docstring(node)
            )

        elif isinstance(node, ast.ClassDef):
            return CodeSymbol(
                name=node.name,
                type='class',
                file_path=str(file_path),
                line_number=node.lineno,
                column=node.col_offset,
                scope=self._get_scope(node),
                signature=f"class {node.name}:",
                docstring=ast.get_docstring(node)
            )

        elif isinstance(node, ast.AsyncFunctionDef):
            return CodeSymbol(
                name=node.name,
                type='async_function',
                file_path=str(file_path),
                line_number=node.lineno,
                column=node.col_offset,
                scope=self._get_scope(node),
                signature=self._get_function_signature(node, is_async=True),
                docstring=ast.get_docstring(node)
            )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                return CodeSymbol(
                    name=alias.name,
                    type='import',
                    file_path=str(file_path),
                    line_number=node.lineno,
                    column=node.col_offset,
                    scope='global',
                    signature=f"import {alias.name}",
                    docstring=None
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                return CodeSymbol(
                    name=alias.name,
                    type='import',
                    file_path=str(file_path),
                    line_number=node.lineno,
                    column=node.col_offset,
                    scope='global',
                    signature=f"from {module} import {alias.name}",
                    docstring=None
                )

        return None

    def _get_scope(self, node: ast.AST) -> str:
        """Determine the scope of a symbol"""
        # This is a simplified version - in reality, you'd track the scope stack
        return 'global'

    def _get_function_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool = False) -> str:
        """Generate function signature"""
        args = []

        # Regular arguments
        for arg in node.args.args:
            args.append(arg.arg)

        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            args.append(arg.arg)

        # Varargs
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # Kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        signature = f"{'async ' if is_async else ''}def {node.name}({', '.join(args)}):"
        return signature

    def _regex_parse_python(self, content: str, file_path: Path) -> List[CodeSymbol]:
        """Fallback regex parsing for Python"""
        symbols = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Function definitions
            func_match = re.match(r'^\s*(async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*:', line)
            if func_match:
                is_async = func_match.group(1) is not None
                func_name = func_match.group(2)
                params = func_match.group(3)

                symbols.append(CodeSymbol(
                    name=func_name,
                    type='async_function' if is_async else 'function',
                    file_path=str(file_path),
                    line_number=line_num,
                    column=0,
                    scope='global',
                    signature=f"{'async ' if is_async else ''}def {func_name}({params}):",
                    docstring=None
                ))

            # Class definitions
            class_match = re.match(r'^\s*class\s+(\w+)(\([^)]*\))?\s*:', line)
            if class_match:
                class_name = class_match.group(1)
                inheritance = class_match.group(2) or ''

                symbols.append(CodeSymbol(
                    name=class_name,
                    type='class',
                    file_path=str(file_path),
                    line_number=line_num,
                    column=0,
                    scope='global',
                    signature=f"class {class_name}{inheritance}:",
                    docstring=None
                ))

        return symbols


class JavaScriptParser(BaseParser):
    """Parser for JavaScript/TypeScript code"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs']

    def parse(self, file_path: Path, content: str) -> List[CodeSymbol]:
        """Parse JavaScript/TypeScript code and extract symbols"""
        symbols = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Function declarations
            func_match = re.match(r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', line)
            if func_match:
                func_name = func_match.group(1)
                params = func_match.group(2)

                symbols.append(CodeSymbol(
                    name=func_name,
                    type='function',
                    file_path=str(file_path),
                    line_number=line_num,
                    column=0,
                    scope='global',
                    signature=f"function {func_name}({params})",
                    docstring=None
                ))

            # Arrow functions
            arrow_match = re.match(r'^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>', line)
            if arrow_match:
                func_name = arrow_match.group(1)

                symbols.append(CodeSymbol(
                    name=func_name,
                    type='function',
                    file_path=str(file_path),
                    line_number=line_num,
                    column=0,
                    scope='global',
                    signature=f"const {func_name} = () => {{}}",
                    docstring=None
                ))

            # Class declarations
            class_match = re.match(r'^\s*(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?', line)
            if class_match:
                class_name = class_match.group(1)

                symbols.append(CodeSymbol(
                    name=class_name,
                    type='class',
                    file_path=str(file_path),
                    line_number=line_num,
                    column=0,
                    scope='global',
                    signature=f"class {class_name}",
                    docstring=None
                ))

        return symbols


class JavaParser(BaseParser):
    """Parser for Java code"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.java']

    def parse(self, file_path: Path, content: str) -> List[CodeSymbol]:
        """Parse Java code and extract symbols"""
        symbols = []
        lines = content.split('\n')
        
        # Track current package and class for scope information
        current_package = None
        current_class = None
        brace_depth = 0
        
        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Skip empty lines and comments
            if not stripped_line or stripped_line.startswith('//') or stripped_line.startswith('/*'):
                continue
                
            # Track brace depth for scope
            brace_depth += line.count('{') - line.count('}')
            
            # Package declaration
            package_match = re.match(r'^\s*package\s+([\w.]+)\s*;', line)
            if package_match:
                current_package = package_match.group(1)
                
            # Import statements
            import_match = re.match(r'^\s*import\s+(?:static\s+)?([\w.*]+)\s*;', line)
            if import_match:
                import_name = import_match.group(1)
                symbols.append(CodeSymbol(
                    name=import_name.split('.')[-1],
                    type='import',
                    file_path=str(file_path),
                    line_number=line_num,
                    column=0,
                    scope='global',
                    signature=f"import {import_name}",
                    docstring=None
                ))
                
            # Class/Interface/Enum declarations
            class_match = re.match(r'^\s*(?:public\s+|private\s+|protected\s+|abstract\s+|final\s+|static\s+)*\s*(class|interface|enum)\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?', line)
            if class_match:
                class_type = class_match.group(1)
                class_name = class_match.group(2)
                current_class = class_name
                
                # Generate scope
                scope = current_package if current_package else 'default'
                
                symbols.append(CodeSymbol(
                    name=class_name,
                    type=class_type,
                    file_path=str(file_path),
                    line_number=line_num,
                    column=0,
                    scope=scope,
                    signature=f"{class_type} {class_name}",
                    docstring=None
                ))
                
            # Method declarations
            method_match = re.match(r'^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+|abstract\s+|synchronized\s+)*\s*(?:(\w+(?:<[^>]+>)?)\s+)?(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\s*[{;]', line)
            if method_match and current_class:
                return_type = method_match.group(1) or 'void'
                method_name = method_match.group(2)
                params = method_match.group(3)
                
                # Skip constructors (method name same as class name)
                if method_name == current_class:
                    method_type = 'constructor'
                    signature = f"{method_name}({params})"
                else:
                    method_type = 'method'
                    signature = f"{return_type} {method_name}({params})"
                
                # Generate scope
                scope = f"{current_package}.{current_class}" if current_package else current_class
                
                symbols.append(CodeSymbol(
                    name=method_name,
                    type=method_type,
                    file_path=str(file_path),
                    line_number=line_num,
                    column=0,
                    scope=scope,
                    signature=signature,
                    docstring=None
                ))
                
            # Field declarations
            field_match = re.match(r'^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+|volatile\s+|transient\s+)*\s*(\w+(?:<[^>]+>)?)\s+(\w+)(?:\s*=\s*[^;]+)?\s*;', line)
            if field_match and current_class and brace_depth > 0:
                field_type = field_match.group(1)
                field_name = field_match.group(2)
                
                # Skip common non-field patterns
                if field_type in ['return', 'throw', 'if', 'for', 'while', 'try', 'catch', 'finally', 'switch', 'case', 'default']:
                    continue
                
                # Generate scope
                scope = f"{current_package}.{current_class}" if current_package else current_class
                
                symbols.append(CodeSymbol(
                    name=field_name,
                    type='field',
                    file_path=str(file_path),
                    line_number=line_num,
                    column=0,
                    scope=scope,
                    signature=f"{field_type} {field_name}",
                    docstring=None
                ))
                
            # Reset current class when we exit its scope
            if brace_depth == 0 and current_class:
                current_class = None

        return symbols


class ParserFactory:
    """Factory for creating appropriate parsers"""

    def __init__(self):
        self.parsers = [
            MarkdownParser(),
            RestructuredTextParser(),
            PythonParser(),
            JavaScriptParser(),
            JavaParser(),
        ]

    def get_parser(self, file_path: Path) -> Optional[BaseParser]:
        """Get appropriate parser for file"""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def parse_file(self, file_path: Path, content: str) -> Union[List[DocumentSection], List[CodeSymbol], None]:
        """Parse file using appropriate parser"""
        parser = self.get_parser(file_path)
        if parser:
            return parser.parse(file_path, content)
        return None


class ContentPreprocessor:
    """Utility class for content preprocessing"""

    @staticmethod
    def clean_markdown(content: str) -> str:
        """Clean markdown content for better indexing"""
        # Remove image references
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)

        # Remove links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)

        # Remove code fences
        content = re.sub(r'```[\s\S]*?```', '', content)
        content = re.sub(r'`[^`]+`', '', content)

        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)

        # Clean up whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)

        return content.strip()

    @staticmethod
    def extract_code_comments(content: str, language: str) -> List[str]:
        """Extract comments from code"""
        comments = []
        lines = content.split('\n')

        if language in ['python', 'shell', 'ruby']:
            # Single line comments with #
            for line in lines:
                match = re.search(r'#\s*(.+)', line)
                if match:
                    comments.append(match.group(1).strip())

        elif language in ['javascript', 'typescript', 'java', 'c', 'cpp']:
            # Single line comments with //
            for line in lines:
                match = re.search(r'//\s*(.+)', line)
                if match:
                    comments.append(match.group(1).strip())

            # Multi-line comments with /* */
            multiline_pattern = r'/\*[\s\S]*?\*/'
            for match in re.finditer(multiline_pattern, content):
                comment = match.group(0)
                # Clean up comment
                comment = re.sub(r'/\*|\*/', '', comment)
                comment = re.sub(r'^\s*\*\s*', '', comment, flags=re.MULTILINE)
                comments.append(comment.strip())

        return [c for c in comments if c]  # Filter empty comments

    @staticmethod
    def normalize_whitespace(content: str) -> str:
        """Normalize whitespace in content"""
        # Replace multiple spaces with single space
        content = re.sub(r' +', ' ', content)

        # Replace multiple newlines with double newline
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

        # Remove trailing whitespace from lines
        content = re.sub(r' +\n', '\n', content)

        return content.strip()
