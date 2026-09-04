import csv

with open("data/grader_label_sheet_v2.csv", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

for row in rows:
    if row["question_id"] in ["q003", "q004", "q005", "q006", "q007", "q008", "q009", "q010"]:
        print("=" * 80)
        print(row["question_id"], "| rank", row["rank"], "|", row["company"])
        print("QUESTION:", row["question"])
        print("CHUNK:")
        print(row["chunk_text"][:500])
        print()