import pypdf
import docx


def extract_text(file):

    text = ""

    if file.name.endswith(".pdf"):

        reader = pypdf.PdfReader(file)

        for page in reader.pages:
            text += page.extract_text()

    elif file.name.endswith(".docx"):

        doc = docx.Document(file)

        for para in doc.paragraphs:
            text += para.text

    elif file.name.endswith(".txt"):

        text = file.read().decode()

    return text