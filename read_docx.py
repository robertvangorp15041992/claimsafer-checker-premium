from docx import Document

def get_doc_content(path):
    doc = Document(path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text
