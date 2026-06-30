import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_question(context, difficulty):

    prompt = f"""
You are an expert technical interviewer.

Based on the following material, generate ONE interview question.

Difficulty level: {difficulty}

Rules:
- Beginner → basic conceptual question
- Intermediate → conceptual + explanation
- Advanced → deeper technical or scenario-based question

Material:
{context}

Return only the question.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content.strip()