import os
import textwrap

from langchain.chains import LLMChain
from langchain.schema.retriever import BaseRetriever
from PyPDF2 import PdfReader
from typing import Tuple


def wrap_text(text: str, width: int = 100) -> str:
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = "\n".join(wrapped_lines)
    return wrapped_text


def process_response(response: str):
    response = wrap_text(response["result"])
    print(response)


def extract_info(file_path: str) -> Tuple[str, str]:
    paper_name_ext = os.path.basename(file_path)
    paper_name = os.path.splitext(paper_name_ext)[0]
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    start_index = text.find("Abstract") + len("Abstract")
    end_index = (
        text.find("Introduction") - 2
    )  # to account for the 1 Introduction or 1. Introduction format

    return paper_name, text[start_index:end_index]


PATH_TO_SCRIPTS = "/notebooks/scripts"


def write_to_script(chapter: str, text: str, paper_name: str):
    script_path = os.path.join(PATH_TO_SCRIPTS, f"{paper_name}.txt")

    with open(script_path, "a") as f:
        f.write(f"{chapter}.\n{text}\n\n")
        f.close()


def write_script(
    qa_chain: LLMChain, retriever: BaseRetriever, table_of_context: str, paper_name: str
):
    chapters = table_of_context.split(",")
    for chapter in chapters:
        docs = extract_document_text(retriever, chapter, paper_name)
        text = qa_chain.run(
            {"paper_name": paper_name, "chapter": chapter, "context": docs}
        )
        write_to_script(chapter, text, paper_name)


def extract_document_text(retriever: BaseRetriever, chapter: str, paper_name) -> str:
    retriever_prompt = f"""Write a chapter on the {chapter} for the research paper {paper_name}, 
    ensure a comprehensive and diverse exploration of the {chapter}, incorporating a wide range of examples and data sources"""

    documents = retriever.get_relevant_documents(retriever_prompt)

    docs = ""
    for doc in documents:
        docs += "\n\n" + doc.page_content
    return docs
