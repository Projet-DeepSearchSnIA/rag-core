import re
from typing import List, Set
from .document_schemas import ContentBlock


class TextPreprocessor:
    """nettoyage et normalisation du texte extrait"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.remove_headers_footers = self.config.get('remove_headers_footers', True)
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)
        self.merge_hyphenated = self.config.get('merge_hyphenated_words', True)
        self.extract_urls = self.config.get('extract_urls', True)
        self.extract_emails = self.config.get('extract_emails', True)

        self.repeated_patterns: Set[str] = set()

    def preprocess_blocks(self, blocks: List[ContentBlock]) -> List[ContentBlock]:
        if self.remove_headers_footers:
            self._detect_repeated_patterns(blocks)

        cleaned_blocks = []

        for block in blocks:
            if not block.content or not block.content.strip():
                continue

            cleaned_content = self._clean_content(block.content, block.type)

            if not cleaned_content.strip():
                continue

            if self.remove_headers_footers and self._is_repeated_pattern(cleaned_content):
                continue

            block.content = cleaned_content

            if block.metadata is None:
                block.metadata = {}

            if self.extract_urls:
                urls = self._extract_urls(cleaned_content)
                if urls:
                    block.metadata['urls'] = urls

            if self.extract_emails:
                emails = self._extract_emails(cleaned_content)
                if emails:
                    block.metadata['emails'] = emails

            cleaned_blocks.append(block)

        return cleaned_blocks

    def _clean_content(self, content: str, content_type: str) -> str:
        # les formules LaTeX ne sont pas touchées
        if content_type == "formula":
            return content.strip()

        if self.normalize_whitespace:
            content = self._normalize_whitespace(content)

        if content_type in ["text", "title"]:
            if self.merge_hyphenated:
                content = self._merge_hyphenated_words(content)
            content = self._clean_special_chars(content)

        elif content_type == "table":
            content = self._clean_table(content)

        return content.strip()

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text

    def _merge_hyphenated_words(self, text: str) -> str:
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text

    def _clean_special_chars(self, text: str) -> str:
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = text.replace('‘', "'").replace('’', "'")
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('–', '-').replace('—', '-')
        return text

    def _clean_table(self, table_text: str) -> str:
        lines = table_text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)

    def _detect_repeated_patterns(self, blocks: List[ContentBlock]):
        content_counts = {}

        for block in blocks:
            content = block.content.strip()
            if len(content) < 100 and len(content) > 5:
                content_counts[content] = content_counts.get(content, 0) + 1

        for content, count in content_counts.items():
            if count > 3:
                self.repeated_patterns.add(content)

    def _is_repeated_pattern(self, content: str) -> bool:
        return content.strip() in self.repeated_patterns

    def _extract_urls(self, text: str) -> List[str]:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return list(set(re.findall(url_pattern, text)))

    def _extract_emails(self, text: str) -> List[str]:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return list(set(re.findall(email_pattern, text)))

    def merge_consecutive_text_blocks(
        self,
        blocks: List[ContentBlock],
        max_length: int = 1000
    ) -> List[ContentBlock]:
        if not blocks:
            return []

        merged = []
        current_text = []
        current_page = blocks[0].page_number

        for block in blocks:
            if (block.type != "text" or
                    block.page_number != current_page or
                    (current_text and sum(len(t) for t in current_text) > max_length)):

                if current_text:
                    merged.append(ContentBlock(
                        type="text",
                        content=' '.join(current_text),
                        page_number=current_page
                    ))
                    current_text = []

                if block.type != "text":
                    merged.append(block)
                else:
                    current_text = [block.content]
                    current_page = block.page_number

            elif block.type == "text":
                current_text.append(block.content)
                current_page = block.page_number

        if current_text:
            merged.append(ContentBlock(
                type="text",
                content=' '.join(current_text),
                page_number=current_page
            ))

        return merged
