from pathlib import Path
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader,DocxReader, ImageReader,PptxReader
def get_file_extractors()->dict:
    extractors = {
        ".pdf":PyMuPDFReader(),
        ".docx":DocxReader(),
        ".doc":DocxReader(),
        ".jpg":ImageReader(),
        ".png":ImageReader(),
        ".jpeg":ImageReader(),
        ".pptx":PptxReader(),
    }
    return extractors
def load_documents_from_path(path: str| Path):
    extractors = get_file_extractors()
    document =SimpleDirectoryReader(path,file_extractor=extractors)
    result=document.load_data()
    return result

