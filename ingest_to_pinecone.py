import json
import os
import uuid
from dotenv import dotenv_values
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import re

# === Load API keys ===
config = dotenv_values(".env")
openai_key = config.get("OPENAI_API_KEY")
pinecone_key = config.get("PINECONE_API_KEY")

if not openai_key or not pinecone_key:
    print("❌ Missing keys in .env")
    exit(1)

# === Init clients ===
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)

index_name = "sweetrobo-ai"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# === Wipe existing vectors (safe delete) ===
print("\U0001f9f9 Deleting existing vectors in the index...")
try:
    index.delete(delete_all=True)
except Exception as e:
    print(f"⚠️ Could not delete existing vectors: {e}")

# === Load and pair Q&A data ===
with open("qa_JSON.json", "r") as f:
    qa_data = json.load(f)

paired_data = []

for entry in qa_data:
    q = entry.get("q", "").strip()
    a = entry.get("a", "").strip()
    if not q or not a:
        continue

    full_text = f"User: {q}\nAssistant: {a}"
    uid = str(uuid.uuid4())

    meta = entry.get("metadata", {})
    usefulness = entry.get("usefulness", {})

    tags = entry.get("tags", [])
    error_codes = []
    for tag in tags:
        match = re.match(r"(?:error_)?(\d{4,})", tag.lower())
        if match:
            try:
                error_codes.append(str(int(match.group(1))))  # Pinecone expects strings or list of strings
            except:
                continue

    metadata = {
        "threadId": entry.get("thread_id"),
        "machine_type": str(entry.get("machine_type", "Unknown")).strip().upper(),
        "tags": tags,
        "domain": meta.get("domain", "unknown"),
        "confidence": float(meta.get("confidence", 0.0)),
        "media_reference": meta.get("media_reference", False),
        "escalation": meta.get("escalation", False),
        "usefulness_score": usefulness.get("score", 0),
        "usefulness_reason": usefulness.get("reason", ""),
        "source": "qa_JSON",
        "q": q,
        "a": a,
        "error_codes": error_codes
    }

    paired_data.append((uid, full_text, metadata))

# === Batch embed and upsert ===
def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

total = 0
skipped = 0

for batch_num, chunk in enumerate(chunked(paired_data, 100), start=1):
    texts = [item[1] for item in chunk]

    try:
        response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    except Exception as e:
        print(f"❌ Failed embedding batch {batch_num}: {e}")
        continue

    vectors = []
    for emb in response.data:
        i = emb.index
        uid, _, meta = chunk[i]
        embedding = emb.embedding

        if embedding is None:
            print(f"❌ NULL embedding for UID {uid}")
            skipped += 1
            continue
        elif not isinstance(embedding, list):
            print(f"❌ Non-list embedding for UID {uid}")
            skipped += 1
            continue
        elif len(embedding) != 1536:
            print(f"❌ Wrong length for UID {uid} — got {len(embedding)})")
            skipped += 1
            continue

        vectors.append({
            "id": uid,
            "values": embedding,
            "metadata": meta
        })

    if vectors:
        index.upsert(vectors=vectors)
        print(f"\U0001f968 Batch {batch_num}: Upserted {len(vectors)} vectors.")
        total += len(vectors)

print(f"\n✅ Total ingested: {total}")
if skipped:
    print(f"⚠️ Skipped: {skipped} entries — see logs above.")