import re

def extract_questions(text):
    """
    Extract questions directly from text if they already exist.
    """
    pattern = r'[^.?!]*\?'
    questions = re.findall(pattern, text)

    cleaned = []

    for q in questions:
        q = q.strip()
        if len(q) > 10:
            cleaned.append(q)

    return cleaned