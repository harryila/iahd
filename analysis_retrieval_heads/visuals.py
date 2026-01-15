import json, pandas as pd, matplotlib.pyplot as plt
from glob import glob

records = []

for path in glob("*_section5.json"):
    with open(path) as f:
        data = json.load(f)

    for r in data["results"]:
        records.append({
            "model": data["model"],
            "context": data["context"],
            "correct": r["correct"],
            "gt": r["ground_truth"],
            "pred": r["normalized_answer"],
        })

df = pd.DataFrame(records)

# ---------------- Accuracy Plot ----------------
acc = df.groupby(["model", "context"])["correct"].mean().reset_index()

plt.figure()
for model in acc.model.unique():
    sub = acc[acc.model == model]
    plt.plot(sub.context, sub.correct, marker="o", label=model)

plt.ylabel("Accuracy")
plt.xlabel("Context Length")
plt.title("Section 5 Accuracy (Numeric Retrieval)")
plt.legend()
plt.show()

# ---------------- UNKNOWN rate ----------------
unknowns = df[df.gt == "UNKNOWN"]
print("\nUNKNOWN prediction counts:")
print(unknowns.groupby(["model", "context"]).size())

# ---------------- Numeric error analysis ----------------
numeric = df[df.gt != "UNKNOWN"]
errors = numeric[numeric.pred != numeric.gt]
print("\nNumeric mismatch counts:")
print(errors.groupby(["model", "context"]).size())
