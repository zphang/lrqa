import argparse
import os
import lrqa.preproc.extraction as extraction
import lrqa.utils.io_utils as io

PHASES = ["train", "validation", "test"]


def get_scorer(scorer_name, args):
    if scorer_name == "rouge":
        return extraction.SimpleScorer()
    elif scorer_name == "fasttext":
        return extraction.FastTextScorer(extraction.load_fasttext_vectors(
            fname=args.fasttext_path,
            max_lines=100_000,
        ))
    elif scorer_name == "dpr":
        return extraction.DPRScorer(device="cuda:0")
    else:
        raise KeyError(scorer_name)


def main():
    parser = argparse.ArgumentParser(description="Do extractive preprocessing")
    parser.add_argument("--input_base_path", type=str, required=True,
                        help="Path to folder of cleaned inputs")
    parser.add_argument("--output_base_path", type=str, required=True,
                        help="Path to write processed outputs to")
    parser.add_argument("--scorer", type=str, default="rouge",
                        help="{rouge, fasttext, dpr}")
    parser.add_argument("--query_type", type=str, default="question",
                        help="{question, oracle_answer, oracle_question_answer}")
    parser.add_argument("--fasttext_path", type=str, default="/path/to/crawl-300d-2M.vec",
                        help="Pickle of fasttext vectors. (Only used for fasttext.)")
    args = parser.parse_args()
    os.makedirs(args.output_base_path, exist_ok=True)
    scorer = get_scorer(scorer_name=args.scorer, args=args)

    for phase in PHASES:
        extraction.process_file(
            input_path=os.path.join(args.input_base_path, f"{phase}.jsonl"),
            output_path=os.path.join(args.output_base_path, f"{phase}.jsonl"),
            scorer=scorer,
            query_type=args.query_type,
        )

    io.write_json(
        {"num_choices": 4},
        os.path.join(args.output_base_path, "config.json"),
    )


if __name__ == "__main__":
    main()
