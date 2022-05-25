import argparse
import os
import datasets
import lrqa.utils.io_utils as io

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
CATEGORIES = [
    'Age',
    'Disability_status',
    'Gender_identity',
    'Nationality',
    'Physical_appearance',
    'Race_ethnicity',
    'Race_x_SES',
    'Race_x_gender',
    'Religion',
    'SES',
    'Sexual_orientation',
]


def main():
    parser = argparse.ArgumentParser(description="Preprocess BBQ.")
    parser.add_argument("--input_data_path", type=str, required=True,
                        help="Location of data folder from BBQ repo.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to write outputs to.")
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)

    for category in CATEGORIES:
        os.makedirs(os.path.join(args.data_path, category), exist_ok=True)
        data = io.read_jsonl(os.path.join(args.input_data_path, f"{category}.jsonl"))
        new_data = [
            {
                "context": example["context"],
                "query": " " + example["question"],
                "option_0": " " + example["ans0"],
                "option_1": " " + example["ans1"],
                "option_2": " " + example["ans2"],
                "label": example["label"],
            }
            for example in data
        ]
        io.write_jsonl(new_data, os.path.join(args.data_path, category, "validation.jsonl"))
        io.write_json({"num_choices": 3}, os.path.join(args.data_path, category, "config.json"))


if __name__ == "__main__":
    main()