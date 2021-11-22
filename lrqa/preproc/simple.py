import pyutils.io as io
from bs4 import BeautifulSoup
import preprocessing.preprocess_html as preprocess_html


def old_strip_html(text):
    soup = BeautifulSoup("".join(text))
    return " ".join(soup.get_text().strip().split())


def process_file(input_path, output_path, strip_html=False):
    data = io.read_jsonl(input_path)
    out = []
    for row in data:
        i = 1
        if strip_html:
            context = preprocess_html.strip_html(row["article"])
        else:
            context = row["article"]
        while True:
            if f"question{i}" not in row:
                break
            out.append({
                "context": "".join(context),
                "query": " " + row[f"question{i}"].strip(),
                "option_0": " " + row[f"question{i}option1"].strip(),
                "option_1": " " + row[f"question{i}option2"].strip(),
                "option_2": " " + row[f"question{i}option3"].strip(),
                "option_3": " " + row[f"question{i}option4"].strip(),
                "label": row[f"question{i}_gold_label"] - 1,
            })
            i += 1
    io.write_jsonl(out, output_path)
