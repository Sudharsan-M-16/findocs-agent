import csv

with open("data/grader_label_sheet_v2.csv", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

with open("data/candidates_for_labeling.txt", "w", encoding="utf-8") as file:
    for i, row in enumerate(rows, 1):
        file.write(f"{i}. [{row['company']}] {row['question_id']} | RANK {row['rank']} | SECTION {row['section']}\n")
        file.write(f"QUESTION: {row['question']}\n")
        file.write("CHUNK:\n")
        file.write(row["chunk_text"])
        file.write("\n")
        file.write("-" * 100)
        file.write("\n\n")

print("Wrote data/candidates_for_labeling.txt")