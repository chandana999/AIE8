import os
import hashlib
from typing import List
from datetime import datetime
import PyPDF2


class DocumentMetadata:
    """Metadata class for tracking document information"""
    
    def __init__(self, filename: str, file_type: str, size: int, 
                 created_date: datetime = None, modified_date: datetime = None, 
                 page_count: int = None):
        self.filename = filename
        self.file_type = file_type
        self.size = size
        self.created_date = created_date
        self.modified_date = modified_date
        self.page_count = page_count
        self.content_hash = None
        self.chunk_index = None
    
    def set_content_hash(self, content: str):
        """Set content hash for the document"""
        self.content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary"""
        return {
            'filename': self.filename,
            'file_type': self.file_type,
            'size': self.size,
            'created_date': self.created_date,
            'modified_date': self.modified_date,
            'page_count': self.page_count,
            'content_hash': self.content_hash,
            'chunk_index': self.chunk_index
        }


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.metadata = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and (self.path.endswith(".txt") or self.path.endswith(".pdf")):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt/.pdf file."
            )

    def load_file(self):
        if self.path.endswith(".pdf"):
            self.load_pdf()
        else:
            with open(self.path, "r", encoding=self.encoding) as f:
                content = f.read()
                self.documents.append(content)
                
                # Create metadata
                file_stats = os.stat(self.path)
                metadata = DocumentMetadata(
                    filename=os.path.basename(self.path),
                    file_type="txt",
                    size=file_stats.st_size,
                    created_date=datetime.fromtimestamp(file_stats.st_ctime),
                    modified_date=datetime.fromtimestamp(file_stats.st_mtime),
                    page_count=1
                )
                metadata.set_content_hash(content)
                self.metadata.append(metadata)
    
    def load_pdf(self):
        """Load text from PDF file"""
        try:
            with open(self.path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                self.documents.append(text)
                
                # Create metadata for PDF
                file_stats = os.stat(self.path)
                metadata = DocumentMetadata(
                    filename=os.path.basename(self.path),
                    file_type="pdf",
                    size=file_stats.st_size,
                    created_date=datetime.fromtimestamp(file_stats.st_ctime),
                    modified_date=datetime.fromtimestamp(file_stats.st_mtime),
                    page_count=len(pdf_reader.pages)
                )
                metadata.set_content_hash(text)
                self.metadata.append(metadata)
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    with open(file_path, "r", encoding=self.encoding) as f:
                        content = f.read()
                        self.documents.append(content)
                        
                        # Create metadata for txt file
                        file_stats = os.stat(file_path)
                        metadata = DocumentMetadata(
                            filename=file,
                            file_type="txt",
                            size=file_stats.st_size,
                            created_date=datetime.fromtimestamp(file_stats.st_ctime),
                            modified_date=datetime.fromtimestamp(file_stats.st_mtime),
                            page_count=1
                        )
                        metadata.set_content_hash(content)
                        self.metadata.append(metadata)
                        
                elif file.endswith(".pdf"):
                    try:
                        with open(file_path, "rb") as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text() + "\n"
                            self.documents.append(text)
                            
                            # Create metadata for PDF file
                            file_stats = os.stat(file_path)
                            metadata = DocumentMetadata(
                                filename=file,
                                file_type="pdf",
                                size=file_stats.st_size,
                                created_date=datetime.fromtimestamp(file_stats.st_ctime),
                                modified_date=datetime.fromtimestamp(file_stats.st_mtime),
                                page_count=len(pdf_reader.pages)
                            )
                            metadata.set_content_hash(text)
                            self.metadata.append(metadata)
                    except Exception as e:
                        print(f"Warning: Could not read PDF {file}: {e}")

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
