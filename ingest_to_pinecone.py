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
    print("ERROR: Missing keys in .env")
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
print("Deleting existing vectors in the index...")
try:
    index.delete(delete_all=True, namespace="sweetrobo-v2")
except Exception as e:
    print(f"WARNING: Could not delete existing vectors: {e}")

# === Load and pair Q&A data ===
with open("cleaned_qa_JSON.json", encoding="utf-8") as f:
    qa_data = json.load(f)

print(f"\n{'='*80}")
print("EMBEDDING STRATEGY: Question + Answer Combined")
print(f"{'='*80}")
print("This allows the chatbot to find answers based on CONTENT, not just question similarity.")
print("Example: If answer mentions 'device test interface', it will be found when searching for that.\n")

paired_data = []

for entry in qa_data:
    q = entry.get("q", "").strip()
    a = entry.get("a", "").strip()
    if not q or not a:
        continue

    # IMPORTANT: Embedding Q+A together so search can find answers based on content
    full_text = f"Question: {q}\n\nAnswer: {a}"
    uid = str(uuid.uuid4())

    meta = entry.get("metadata", {})
    usefulness = entry.get("usefulness", {})

    tags = entry.get("tags", [])

    # NEW: Use error_codes from metadata if it exists (from smart extraction)
    # Otherwise, fall back to extracting from tags
    error_codes_from_metadata = meta.get("error_codes", [])
    if error_codes_from_metadata:
        # Use the smart-extracted error codes
        error_codes = [str(code) for code in error_codes_from_metadata]
    else:
        # Fallback: extract from tags (old method)
        error_codes = []
        for tag in tags:
            match = re.match(r"(?:error_)?(\d{4,})", tag.lower())
            if match:
                try:
                    error_codes.append(str(int(match.group(1))))
                except:
                    continue

    # NEW: Get entry_type from metadata (direct_qa or knowledge_base)
    entry_type = meta.get("entry_type", "direct_qa")

    metadata = {
        "thread_id": entry.get("thread_id"),
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
        "error_codes": error_codes,
        "entry_type": entry_type  # NEW: Include entry_type for routing
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
        print(f"ERROR: Failed embedding batch {batch_num}: {e}")
        continue

    vectors = []
    for emb in response.data:
        i = emb.index
        uid, _, meta = chunk[i]
        embedding = emb.embedding

        if embedding is None:
            print(f"ERROR: NULL embedding for UID {uid}")
            skipped += 1
            continue
        elif not isinstance(embedding, list):
            print(f"ERROR: Non-list embedding for UID {uid}")
            skipped += 1
            continue
        elif len(embedding) != 1536:
            print(f"ERROR: Wrong length for UID {uid} - got {len(embedding)})")
            skipped += 1
            continue

        vectors.append({
            "id": uid,
            "values": embedding,
            "metadata": meta
        })

    if vectors:
        index.upsert(namespace="sweetrobo-v2", vectors=vectors)
        print(f"Batch {batch_num}: Upserted {len(vectors)} vectors.")
        total += len(vectors)

print(f"\n{'='*80}")
print("UPLOAD COMPLETE!")
print(f"{'='*80}")
print(f"Total vectors ingested: {total}")
if skipped:
    print(f"WARNING: Skipped: {skipped} entries - see logs above.")

# Count entries with error codes and entry types
entries_with_error_codes = sum(1 for _, _, meta in paired_data if meta.get("error_codes"))
direct_qa_count = sum(1 for _, _, meta in paired_data if meta.get("entry_type") == "direct_qa")
knowledge_base_count = sum(1 for _, _, meta in paired_data if meta.get("entry_type") == "knowledge_base")

print(f"\nDataset Statistics:")
print(f"  - Entries with error codes: {entries_with_error_codes}")
print(f"  - Direct Q&A entries: {direct_qa_count}")
print(f"  - Knowledge base entries: {knowledge_base_count}")

print(f"\n{'='*80}")
print("EMBEDDING STRATEGY ACTIVE")
print(f"{'='*80}")
print("[OK] Each vector contains BOTH question AND answer")
print("[OK] Search will find entries based on answer content, not just questions")
print("[OK] Example: 'device test interface' in answer will be found even if question doesn't mention it")
print(f"{'='*80}\n")