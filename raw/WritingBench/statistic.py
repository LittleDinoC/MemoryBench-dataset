import jsonlines
import json

with open("/nfs/wangchangyue1/MemoryBench/raw/WritingBench/benchmark_all.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    
label = {}

for d in data:
    if d["domain1"] not in label:
        label[d["domain1"]] = []
    if not d["domain2"] in label[d["domain1"]]:
        label[d["domain1"]].append(d["domain2"])
    
print(json.dumps(label, indent=2, ensure_ascii=False))