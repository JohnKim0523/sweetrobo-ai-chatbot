import json
import time
from openai import OpenAI
from dotenv import dotenv_values

# === Load API key ===
config = dotenv_values(".env")
client = OpenAI(api_key=config.get("OPENAI_API_KEY"))

# === GPT Reformatter ===
def reformat_answer(question, answer):
    prompt = f"""
You are formatting customer support answers.

Format the response in a structured way using this template:

- Start with a one-line summary of the problem.
- Add a "Why is this happening?" section if the original answer contains possible causes.
- Add a "What to check:" or "What to do:" section if there are troubleshooting or fix steps.
- Use plain text (no markdown).
- Use short bullet points, NOT full sentences.
- Omit sections that don't apply.

Question:
{question}

Original Answer:
{answer}

Structured Answer:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error for question: {question}\n{e}")
        return answer  # fallback to original if GPT fails

# === Load Original Dataset ===
with open("qa_JSON.json", encoding="utf-8") as f:
    data = json.load(f)

# === Process & Reformat ===
for i, item in enumerate(data):
    print(f"🔄 Processing item {i + 1}/{len(data)}...")
    q = item.get("q", "")
    a = item.get("a", "")
    if q and a:
        item["a"] = reformat_answer(q, a)
        time.sleep(1.2)  # slow down to avoid rate limits

# === Save Cleaned File ===
with open("cleaned_qa_JSON.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("\n✅ All done! Reformatted dataset saved as cleaned_qa_JSON.json")
