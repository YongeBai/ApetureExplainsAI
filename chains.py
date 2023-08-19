import os

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema.retriever import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, TextIteratorStreamer, pipeline

torch.cuda.empty_cache()

# Loading Model
model_name_or_path = "TheBloke/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GPTQ"
model_basename = "wizard-vicuna-13b-uncensored-superhot-8k-GPTQ-4bit-128g.no-act.order"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    device_map="balanced",
    use_triton=False,
    quantize_config=None,
)

model.seqlen = 4096

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=4096,
)

llm = HuggingFacePipeline(pipeline=pipe)

# Making table_of_context_chain
template = """
Use the following context to create a table of context for an 8 minute educational Youtube video about the research paper {paper_name}

{abstract}

here are some examples for other papers, return something in the exact same format and nothing else
ex.
Attention Is All You Need
Introduction,Summary,Background,Model Architecture,Results and Applications,Conclusion

BERT 
Introduction,Summary,Model Overview,Application Overview,Ethics Statement,Conclusion

FlashAttention
Introduction,Summary,Background,I-O Awareness,Block-Sparse FlashAttention,Experiments,Conclusion

{paper_name}
"""
prompt = PromptTemplate(input_variables=["abstract", "paper_name"], template=template)
table_of_context_chain = LLMChain(llm=llm, prompt=prompt)

# Making QA chain
template = """
You are GlaDOS, the main antagonist from the Portal series. You are creating an eductional youtube video about {paper_name}
for a Youtube channel called Apeture Explains AI
Use the context below to write the {chapter} chapter for the script

{context}

{chapter}

"""
prompt = PromptTemplate(
    input_variables=["paper_name", "chapter", "context"], template=template
)

llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = LLMChain(llm=llm, prompt=prompt)


def get_retriever(file_path: str) -> BaseRetriever:
    # Loading and splitting docs
    doc_1 = PyPDFLoader(file_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50)
    text_1 = text_splitter.split_documents(doc_1)

    # Embeddings and loading Vector DB
    model_name = "thenlper/gte-large"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    persist_dir = "db"
    vectordb = Chroma.from_documents(
        documents=text_1, embedding=embeddings, persist_directory=persist_dir
    )
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4, "fetch_k": 15}, search_type="mmr"
    )
    return retriever
