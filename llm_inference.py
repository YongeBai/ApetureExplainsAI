from argparse import ArgumentParser

import torch

from chains import get_retriever, qa_chain, table_of_context_chain
from utils import extract_info, write_script

torch.cuda.empty_cache()

print(torch.cuda.get_device_name(0))


def main():
    parser = ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    file_path = args.file_path

    retriever = get_retriever(file_path)
    title, abstract = extract_info(file_path)
    table_of_context = table_of_context_chain.run(
        {"paper_name": title, "abstract": abstract}
    )

    print("Written script")
    write_script(qa_chain, retriever, table_of_context, title)


if __name__ == "__main__":
    main()
