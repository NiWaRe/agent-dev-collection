import os, re, random
from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Dict, Optional
from pydantic import PrivateAttr

import weave
import asyncio
from weave import Table
from weave.trace import serializer 
from weave.trace.custom_objs import MemTraceFilesArtifact

import faiss
import numpy as np
import pandas as pd

from litellm import (
    embedding, 
    aembedding, 
    completion,
    acompletion,
)

from langchain_community.document_loaders import WebBaseLoader, OnlinePDFLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Weave Serializers are used to define how non-primitive data types are stored on Weave
# Weave Objects allow custom objects to be displayed other than just their __str__ on Weave

#########
# MODEL #
#########
# TODO: replace Any type for _model with something more sensible
# TODO: validate whether extra handling of gpt is necessary
# TODO: check with developers of litellm (one wrote me) why they re-init the model on every predict
# TODO: implement local models or HF logic (check out logic from lc project and Thomas' PR)
class ChatModel(weave.Model):
    """
    We define an extra ChatModel class to be able store and version more parameters than just the model name.
    Especially, relevant if we consider fine-tuning (locally or aaS) because of specific parameters.
    """
    chat_model: str
    cm_temperature: float
    cm_max_new_tokens: int
    cm_quantize: bool
    inference_batch_size: int
    device: str
    _model: Any = PrivateAttr()

    def model_post_init(self, __context):
        # either use LiteLLM or local models
        pass

    @weave.op()
    async def predict(self, query: List[str]) -> dict:
        completion_args = {
            "model": self.chat_model,
            "messages": query,
            "temperature": self.cm_temperature,
            "max_tokens": self.cm_max_new_tokens,
        }
        response = await acompletion(**completion_args)

        # NOTE: make sure that copied values are returned and not references
        return dict(response.choices[0].message)
    
    
# TODO: check whether this will be recognized as "predict" for weave.Model
# TODO: make sure that return type will be list of list of floats

# TODO: check whether this will be recognized as "predict" for weave.Model
# TODO: make sure that return type will be list of list of floats
# TODO: implement the difference of embedding model being a local model, a string to HF or a string to LiteLLM
class EmbeddingModel(weave.Model):
    """
    We define an extra EmbeddingModel class to be able store and version more parameters than just the model name.
    Especially, relevant if we consider fine-tuning (locally or aaS) because of specific parameters.
    """
    embedding_model: str
    device: str
    embedding_model_norm_embed: bool
    _model: Any = PrivateAttr()

    def model_post_init(self, __context):
        # either use LiteLLM or local models
        # self._model = HuggingFaceEmbeddings(
        #     model_name=self.embedding_model,
        #     model_kwargs={"device": self.device},
        #     encode_kwargs={"normalize_embeddings": self.embedding_model_norm_embed},
        # )
        pass

    @weave.op()
    def embedd(self, docs: List[str]) -> List[float]:        
        doc_embeddings = embedding(
            model=self.embedding_model,
            input=docs,
            #logger_fn=lambda x:print(f"LITELLM CALL: {x}"),
        )
        if len(docs) == 1:
            return doc_embeddings["data"][0]["embedding"]
        else:
            return [doc_embedding["embedding"] for doc_embedding in doc_embeddings["data"]]

    @weave.op()
    async def aembedd(self, docs: List[str]) -> List[float]:        
        doc_embeddings = await aembedding(
            model=self.embedding_model,
            input=docs,
            #logger_fn=lambda x:print(f"LITELLM CALL: {x}"),
        )
        if len(docs) == 1:
            return doc_embeddings["data"][0]["embedding"]
        else:
            return [doc_embedding["embedding"] for doc_embedding in doc_embeddings["data"]]
    
# TODO: check if this necessary and how to make more general next to OpenAI Models
class PromptTemplate(weave.Object):
    system_prompt: str
    human_prompt: str

    @weave.op()
    def format_prompt(
        self,
        system_prompt_args: Optional[Dict[str, str]] = {},
        human_prompt_args: Optional[Dict[str, str]] = {},
    ):
        "A formatting function for OpenAI models"
        system_prompt_formatted = self.system_prompt.format(**system_prompt_args)
        human_prompt_formatted = self.human_prompt.format(**human_prompt_args)
        messages = [
            {"role": "system", "content": system_prompt_formatted},
            {"role": "user", "content": human_prompt_formatted},
        ]
        return messages
    
###############
# VECTORSTORE #
###############

def save_instance(obj: faiss.IndexFlatIP, artifact: MemTraceFilesArtifact, name: str) -> None:
    """
    Allow faiss index stores to be saved in Weave.
    """
    with artifact.writeable_file_path(f"{name}.faissindex") as write_path:
        faiss.write_index(obj, write_path)

def load_instance(artifact: MemTraceFilesArtifact, name: str) -> faiss.IndexFlatIP:
    """
    Allow faiss index stores to be loaded from Weave.
    """
    return faiss.read_index(artifact.path(f"{name}.faissindex"))

# TODO: check Anish's implementation for more mature class 
# (multi-process embeddings for search too, pre-computing, different distance functions - also how to extend index)
class VectorStore(weave.Object):
    """
    VectorStore object that holds index model, docs reference, and embedding model as str.
    It should be used to both init and search the index.
    Modified from hooman chatbot: https://github.com/wandb/hooman/blob/main/faiss_vectorstore.py
    Inspired by Anish's demo: https://github.com/ash0ts/snowflake-arctic-weave-demo
    """
    docs: weave.Dataset
    embedding_model: EmbeddingModel 
    key: str = "page_content" 
    limit: int = -1
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunked_docs: Optional[List[Dict]] = None
    index: Optional[faiss.IndexFlat] = None
    
    # def model_post_init(self, __context):
    #     serializer.register_serializer(faiss.Index, save_instance, load_instance)
    #     #asyncio.run(self._embed_all_docs_async())
    #     self.index = self._embed_all_docs()
        
    # TODO: changes to this function will not be versioned!
    # TODO: how to pass in kwargs for weave.Object (like name)
    # we only pass in arguments to the contructor that don't need further processing in __init__
    def __init__(self, docs: weave.Dataset, embedding_model: EmbeddingModel, key: str = "page_content", 
                 limit: int = -1, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, 
                 chunked_docs: Optional[List[Dict]] = None, index: Optional[faiss.IndexFlat] = None):
        
        super().__init__(docs=docs, embedding_model=embedding_model, key=key, limit=limit, chunk_size=chunk_size,
                         chunk_overlap=chunk_overlap, chunked_docs=chunked_docs, index=index)
        
        # set fixed attributes
        self.docs = docs
        self.embedding_model = embedding_model
        self.key = key
        self.limit = limit
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunked_docs = chunked_docs

        # Register the serializer for FAISS index
        serializer.register_serializer(faiss.IndexFlatIP, save_instance, load_instance)

        # Embed the docs if index doesn't already exist (will update self.index, self.chunked_docs)
        if not index:
            #asyncio.run(self._embed_all_docs_async())
            index = self._embed_all_docs()

        # kwargs are only args meant for weave.Object (like name, description, etc.)
        super().__init__(docs=docs, embedding_model=embedding_model, key=key, limit=limit, chunk_size=chunk_size,
                         chunk_overlap=chunk_overlap, chunked_docs=self.chunked_docs, index=index)
    
    @weave.op()
    async def _chunk_docs_async(self, ref_col: str, non_chunked_docs: Table) -> List[str]:
        """
        Chunk the documents in the dataset. Also convert Weave types into primitive types while doing so.
        """
        if not (self.chunk_size and self.chunk_overlap):
            raise ValueError("No chunk_size or chunk_overlap provided. Please provide both.")
        
        chunked_docs = []
        for doc in non_chunked_docs:
            ref_field_list = doc[ref_col].split()
            for i in range(0, len(ref_field_list), self.chunk_size - self.chunk_overlap):
                new_sub_doc = {k: v for k, v in doc.items()}
                new_sub_doc[ref_col] = " ".join(ref_field_list[i:i + self.chunk_size])
                chunked_docs.append(new_sub_doc)
        return chunked_docs

    @weave.op()
    def _chunk_docs(self, ref_col: str, non_chunked_docs: Table) -> List[str]:
        """
        Chunk the documents in the dataset. Also convert Weave types into primitive types while doing so.
        """
        if not (self.chunk_size and self.chunk_overlap):
            raise ValueError("No chunk_size or chunk_overlap provided. Please provide both.")
        
        chunked_docs = []
        for doc in tqdm(non_chunked_docs, desc="Chunking documents"):
            ref_field_list = doc[ref_col].split()
            for i in range(0, len(ref_field_list), self.chunk_size - self.chunk_overlap):
                new_sub_doc = {k: v for k, v in doc.items()}
                new_sub_doc[ref_col] = " ".join(ref_field_list[i:i + self.chunk_size])
                chunked_docs.append(new_sub_doc)
        return chunked_docs

    @weave.op()
    async def _embed_all_docs_async(self) -> None:
        """
        Embed all documents in the dataset, chunk them, and create the FAISS index.
        """
        if not self.docs:
            raise ValueError("No documents found in the dataset.")

        # Chunk and embed documents concurrently    
        self._chunked_docs = await self._chunk_docs_async(ref_col=self.key, non_chunked_docs=self.docs.rows[:self.limit])
        embedding_tasks = [self.embedding_model.aembedd(docs=[doc[self.key]]) for doc in self._chunked_docs]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        
        ## Create FAISS index and add the embeddings
        self.index = faiss.IndexFlatIP(len(embeddings_list[0]))
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        self.index.add(embeddings_matrix)

    @weave.op()
    def _embed_all_docs(self) -> faiss.IndexFlatIP:
        """
        Embed all documents in the dataset, chunk them, and create the FAISS index.
        """
        if not self.docs:
            raise ValueError("No documents found in the dataset.")

        # Chunk and embed documents concurrently    
        self.chunked_docs = self._chunk_docs(ref_col=self.key, non_chunked_docs=self.docs.rows[:self.limit])
        embeddings_list = []
        for doc in tqdm(self.chunked_docs, desc="Embedding documents"):
            embeddings_list.append(
                self.embedding_model.embedd(docs=[doc[self.key]])
            )
        
        ## Create FAISS index and add the embeddings
        index = faiss.IndexFlatIP(len(embeddings_list[0]))
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        index.add(embeddings_matrix)
        return index

    @weave.op()
    def search(self, query: str, k: int) -> List[Dict]:
        """
        Search for the appropriate document chunks using faiss.IndexFlat.search returning max k vectors.
        Return a list of dicts with at least the keys "content" and "url" (used in eval).
        """
        if not self.index:
            raise ValueError("No index has been created. Please call create first.")

        embedded_query = self.embedding_model.embedd(docs=[query])
        query_vector = np.array(embedded_query, dtype=np.float32)

        if query_vector.ndim == 1:
            query_vector = query_vector[np.newaxis, :]
        if query_vector.shape[1] != self.index.d:
            raise ValueError(f"Query vector shape {query_vector.shape} does not match index shape {self.index.d}")
        
        # scores, indices = await asyncio.to_thread(self.index.search, query_vector, k)
        scores, indices = self.index.search(query_vector, k)

        # TODO: wrap in list to make sure that value is returned and not reference
        return list([self.chunked_docs[int(i)] for i in indices[0]])
    
###########
# GENERAL #
###########
# TODO: check Anish's RAG example (pre- and post-processing functions, why @dataclass PromptTemplate)
class RagModel(weave.Model):
    chat_model: ChatModel
    vector_store: VectorStore
    rag_prompt_user: str
    rag_prompt_system: str
    raw_data_artifact: str
    retrieval_chain_type: str
    inference_batch_size: int
    prompt: Optional[PromptTemplate] = None

    def __init__(self, chat_model: ChatModel, vector_store: VectorStore, rag_prompt_user: str,
                 rag_prompt_system: str, raw_data_artifact: str, retrieval_chain_type: str,
                 inference_batch_size: int, prompt: PromptTemplate = None):
        super().__init__(chat_model=chat_model, vector_store=vector_store, rag_prompt_user=rag_prompt_user, 
                         rag_prompt_system=rag_prompt_system, raw_data_artifact=raw_data_artifact, 
                         retrieval_chain_type=retrieval_chain_type, inference_batch_size=inference_batch_size, 
                         prompt=prompt)
        self.chat_model = chat_model
        self.vector_store = vector_store
        self.rag_prompt_user = rag_prompt_user
        self.rag_prompt_system = rag_prompt_system
        self.raw_data_artifact = raw_data_artifact
        self.retrieval_chain_type = retrieval_chain_type
        self.inference_batch_size = inference_batch_size
        self.prompt = prompt
        
        prompt = PromptTemplate(
            system_prompt=self.rag_prompt_system,
            human_prompt=self.rag_prompt_user,
        )

        super().__init__(chat_model=chat_model, vector_store=vector_store, rag_prompt_user=rag_prompt_user, 
                         rag_prompt_system=rag_prompt_system, raw_data_artifact=raw_data_artifact, 
                         retrieval_chain_type=retrieval_chain_type, inference_batch_size=inference_batch_size, 
                         prompt=prompt)

    @weave.op()
    async def predict(self, query: str, n_documents: int = 2) -> dict:
        # vectorstore search
        context_documents = self.vector_store.search(query=query, k=n_documents)

        # prompt formatting
        context = "\n\n".join(
            [f"Context {i+1}:\n{doc}" for i,
                doc in enumerate(context_documents)]
        )
        human_prompt_args = {
            "question": query,
            "context": context,
        }
        messages = self.prompt.format_prompt(
            human_prompt_args=human_prompt_args
        )

        # chat model inference 
        answer = await self.chat_model.predict(messages)
        return {"result": answer, "source_documents": context_documents}
    
    
# TODO: replace langchain download and extraction with own functions
@weave.op()
def download_source_docs(
        source_list_path: str,
        raw_data_artifact: str,
        **kwargs,
    ) -> None:
    """Download sources and save them as table artifact to Weave"""

    # Read the sources list
    sources_list_df = pd.read_csv(Path(__file__).parent/source_list_path)
    sources_list = sources_list_df.to_dict(orient="records")

    # Initialize a list to store all downloaded sources
    downloaded_sources = []

    # Define loader mapping for different types of sources
    loader_mapping = {
        "pdf": OnlinePDFLoader,
        "web": WebBaseLoader
    }
    for source in tqdm(sources_list, desc="Downloading sources"):
        # Select the appropriate loader and download the source
        loader_cls = loader_mapping.get(source["type"], WebBaseLoader)
        extracted_raw = loader_cls(source["url"]).load()

        for extracted in extracted_raw:
            extracted_dict = extracted.dict()
            extracted_dict["metadata"] = str(extracted_dict["metadata"])  # Convert metadata to string for Weave compatibility
            extracted_dict.update(source)  # Add the original source info to extracted data
            downloaded_sources.append(extracted_dict)

    # Convert the list of dictionaries to a DataFrame
    sources_df = pd.DataFrame(downloaded_sources, columns=sources_list_df.columns.tolist() + ["page_content", "metadata"])

    # Create and publish the dataset to Weave
    dataset = weave.Dataset(
        name=raw_data_artifact,
        rows=sources_df.to_dict(orient="records")
    ) 
    weave.publish(dataset)

# TODO: replace langchain chunking with own functions
@weave.op()
async def gen_data(
        gen_model: ChatModel,
        prompt_template: PromptTemplate,
        raw_data_artifact: str,
        dataset_artifact: str,
        questions_per_chunk: int,
        max_chunks_considered: int,
        source_chunk_size: int,
        source_chunk_overlap: int,
        **kwargs,
    ) -> None:
    """Generate question-answer-source pairs for the provided sources and upload to Weave.
       Inspired by llamaindex.evaluation.DatasetGenerator that generates questions per document.
       We will assume a document to be the entirety of a given source. In contrary to LlamaIndex
       we will not first generate questions and the responses in a separate step but we will generate
       both questions and answers at the same time and use custom parsing to extract the pairs."""
    
    # weave: get sources and split into chunks (with :latest version)
    source_df = pd.DataFrame(weave.ref(raw_data_artifact).get().rows)
    source_docs = DataFrameLoader(source_df, page_content_column="page_content").load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=source_chunk_size,
        chunk_overlap=source_chunk_overlap
    )
    all_splits = text_splitter.split_documents(source_docs)

    # Sample uniformly from all splits
    sampled_docs = random.sample(all_splits, min(max_chunks_considered, len(all_splits)))

    # Generate questions and answers concurrently per sampled_doc
    queries, answers, sources = [], [], []

    async def generate_qa_pairs(doc):
        """Generate questions and answers for a given document."""
        messages = prompt_template.format_prompt(
            human_prompt_args={
                "questions_per_chunk": questions_per_chunk,
                "source_str": doc.page_content,
            }
        )
        output = await gen_model.predict(messages)

        doc_queries = re.findall(r"QUESTION: (.*)\nANSWER:", output['content'])
        doc_answers = re.findall(r"ANSWER: (.*)", output['content'])
        doc_sources = [doc.metadata['url']] * questions_per_chunk

        return doc_queries, doc_answers, doc_sources

    results = await asyncio.gather(*[generate_qa_pairs(doc) for doc in sampled_docs])

    # Aggregate results
    for doc_queries, doc_answers, doc_sources in results:
        queries.extend(doc_queries)
        answers.extend(doc_answers)
        sources.extend(doc_sources)

    # Create and publish the dataset
    weave.publish(weave.Dataset(
        name=dataset_artifact,
        rows=[{"query": query, "answer": answer, "main_source": source}
              for query, answer, source in zip(queries, answers, sources)]
    ))
