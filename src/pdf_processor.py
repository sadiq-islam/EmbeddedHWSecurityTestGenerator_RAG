"""Local Docling conversion. Token-aware preparation happens in VectorStore."""
from hashlib import sha256
from io import BytesIO


class PDFProcessor:
    def __init__(self, pdf_bytes: bytes, max_pages: int = 100,
                 source_name: str = "uploaded_spec.pdf"):
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        self.pdf_bytes = pdf_bytes
        self.max_pages = max_pages
        self.source_name = source_name
        self.document_id = sha256(pdf_bytes).hexdigest()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def extract_document(self):
        import pymupdf
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.datamodel.base_models import InputFormat, DocumentStream
        from docling.chunking import HierarchicalChunker

        with pymupdf.open(stream=self.pdf_bytes, filetype="pdf") as pdf:
            total_pages = pdf.page_count
            if total_pages > self.max_pages:
                raise ValueError(f"PDF has {total_pages} pages; page limit is {self.max_pages}")
        options = PdfPipelineOptions()
        options.do_ocr = True
        options.ocr_options = RapidOcrOptions()
        options.do_table_structure = True
        # Page images are not consumed by this text retrieval implementation.
        options.generate_page_images = False
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        result = converter.convert(DocumentStream(name=self.source_name,
                                                  stream=BytesIO(self.pdf_bytes)))
        chunks = list(HierarchicalChunker().chunk(result.document))
        return result.document, chunks, total_pages
