from rouge_score import rouge_scorer
import spacy
import pyutils.io as io
import pyutils.display as display
from bs4 import BeautifulSoup
import numpy as np


class SimpleScorer:
    def __init__(self, metrics=(("rouge1", "r"),), use_stemmer=True):
        self.metrics = metrics
        self.scorer = rouge_scorer.RougeScorer(
            [metric[0] for metric in self.metrics],
            use_stemmer=use_stemmer,
        )

    def score(self, reference: str, target: str):
        scores = self.scorer.score(reference, target)
        sub_scores = []
        for metric, which_score in self.metrics:
            score = scores[metric]
            if which_score == "p":
                score_value = score.precision
            elif which_score == "r":
                score_value = score.recall
            elif which_score == "f":
                score_value = score.fmeasure
            else:
                raise KeyError(which_score)
            sub_scores.append(score_value)
        return np.mean(sub_scores)


def get_sent_data(raw_text: str):
    """Given a passage, return sentences and word counts."""
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    soup = BeautifulSoup("".join(raw_text))
    context = " ".join(soup.get_text().strip().split())
    sent_data = []
    for sent_obj in nlp(context).sents:
        sent_data.append({
            "text": str(sent_obj).strip(),
            "word_count": len(sent_obj),
        })
    return sent_data


def get_top_sentences(query: str, sent_data: list, max_word_count: int, scorer: SimpleScorer):
    scores = []
    for sent_idx, sent_dict in enumerate(sent_data):
        scores.append((sent_idx, scorer.score(query, sent_dict["text"])))

    # Sort by rouge score, in descending order
    sorted_scores = sorted(scores, key=lambda _: _[1], reverse=True)

    # Choose highest scoring sentences
    chosen_sent_indices = []
    total_word_count = 0
    for sent_idx, score in sorted_scores:
        sent_word_count = sent_data[sent_idx]["word_count"]
        if total_word_count + sent_word_count > max_word_count:
            break
        chosen_sent_indices.append(sent_idx)
        total_word_count += sent_word_count

    # Re-condense article
    shortened_article = " ".join(sent_data[sent_idx]["text"] for sent_idx in sorted(chosen_sent_indices))
    return shortened_article


def process_file(input_path, output_path, scorer: SimpleScorer, query_type="question", max_word_count=300,
                 verbose=False):
    data = io.read_jsonl(input_path)
    out = []
    for row in display.maybe_tqdm(data, verbose=verbose):
        sent_data = get_sent_data(row["article"])
        i = 1
        while True:
            if f"question{i}" not in row:
                break
            if query_type == "question":
                query = row[f"question{i}"].strip()
            elif query_type == "oracle_answer":
                query = row[f"question{i}option{row[f'question{i}_gold_label']}"].strip()
            elif query_type == "oracle_question_answer":
                query = (
                    row[f"question{i}"].strip()
                    + " " + row[f"question{i}option{row['question{i}_gold_label']}"].strip()
                )
            else:
                raise KeyError(query_type)
            shortened_article = get_top_sentences(
                query=query,
                sent_data=sent_data,
                max_word_count=max_word_count,
                scorer=scorer,
            )
            out.append({
                "context": shortened_article,
                "query": " " + row[f"question{i}"].strip(),
                "option_0": " " + row[f"question{i}option1"].strip(),
                "option_1": " " + row[f"question{i}option2"].strip(),
                "option_2": " " + row[f"question{i}option3"].strip(),
                "option_3": " " + row[f"question{i}option4"].strip(),
                "label": row[f"question{i}_gold_label"] - 1,
            })
            i += 1
    io.write_jsonl(out, output_path)
