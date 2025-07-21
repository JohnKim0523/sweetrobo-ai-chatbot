import os
from dotenv import load_dotenv
from pinecone import Pinecone

# === Load .env ===
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")
if not pinecone_key:
    raise ValueError("❌ Missing PINECONE_API_KEY")

# === Init Pinecone client and index ===
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("sweetrobo-ai")
namespace = "sweetrobo-v2"

# === Test hybrid query with metadata filter ===
query_text = "machine won't turn on"

try:
    print(f"\n🔍 Querying: {query_text}")
    res = index.query(
        namespace=namespace,
        text=query_text,  # ← hybrid search trigger
        top_k=5,
        include_metadata=True,
        filter={
            "tags": {"$in": ["cotton"]}  # optional: match only "cotton" tagged vectors
        }
    )

    if not res.matches:
        print("❌ No matches found.")
    else:
        print(f"✅ Found {len(res.matches)} match(es):\n")
        for i, match in enumerate(res.matches, start=1):
            score = round(match.get("score", 0), 3)
            q = match.metadata.get("q", "[No Q]")
            a = match.metadata.get("a", "[No A]")
            print(f"🔹 Match {i} (Score: {score})\nQ: {q}\nA: {a}\n")

except Exception as e:
    print(f"❌ Query error: {e}")
