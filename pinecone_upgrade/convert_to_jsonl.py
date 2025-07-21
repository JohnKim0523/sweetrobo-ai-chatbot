import json

with open("qa_JSON.json", "r", encoding="utf-8") as infile, open("qa_pairs.jsonl", "w", encoding="utf-8") as outfile:
    data = json.load(infile)
    for item in data:
        json.dump(item, outfile)
        outfile.write("\n")
