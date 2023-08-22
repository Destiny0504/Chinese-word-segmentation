import argparse

import MCCWS.preprocess
import MCCWS.util


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_gold_path",
        default="",
        help="Give the path of test gold file which you want to preprocess.",
        type=str,
    )
    parser.add_argument(
        "--new_path",
        default="",
        help="Give the path where to store the preprocessed test file.",
        type=str,
    )

    return parser.parse_args()


def main(args):
    MCCWS.util.set_seed(42)

    dataset = {"[SET]": [args.original_gold_path]}
    preprocessed_dataset = MCCWS.preprocess.preprocess(datasets=dataset)

    with open(f"{args.original_gold_path}", "w") as f:
        for data in preprocessed_dataset.original_data:
            f.write(data + "\n")
    with open(args.new_path, "w") as f:
        for data in preprocessed_dataset.data:
            f.write(data + "\n")


if __name__ == "__main__":
    main(args=get_args())
