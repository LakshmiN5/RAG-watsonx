
import pymilvus as milvus
from pymilvus import MilvusClient, DataType
import sys
import abc
import os
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain_milvus import Milvus

os.environ["GRPC_DNS_RESOLVER"] = ""

class MilvusStrategy:
    def connect(self):
        raise NotImplementedError("Must implement connect() method")


class RemoteMilvusStrategy(MilvusStrategy):
    def connect(self):
        print("Connecting to remote Milvus instance.")
        client = MilvusClient(
            uri=SERVER_ADDR,
            secure=False,
            server_pem_path=SERVER_PEM_PATH,
        )
        print("Connected to remote Milvus instance.")
        return client

class InMemoryMilvusStrategy(MilvusStrategy):
    def connect(self):
        print("Connecting to in-memory Milvus instance.")
        client = MilvusClient("milvus_demo.db")
        print("Connected to in-memory Milvus instance.")
        return client

def adapt(number_of_files, total_file_size, data_size_in_kbs):
    """
    Decides whether to use a remote Milvus connection or an in-memory connection based on the number of files and their total size.

    
    Parameters:
    number_of_files (int): The number of files to be indexed.
    total_file_size (float): The total size of the files to be indexed, in megabytes.

    Returns:
    client: A Milvus client object.
    """
    strategy = InMemoryMilvusStrategy()
    if(number_of_files> 10 or 
        total_file_size > 10 or 
        data_size_in_kbs > .25)  :
        strategy = RemoteMilvusStrategy()
    

    client = strategy.connect()
    return client


#TODO: Provide the Hosted Milvus Server Address. Format: https://username:password@hostname:portnumber
SERVER_ADDR = ""                

# TODO: Provide the Hosted Milvus Server Permission Path
SERVER_PEM_PATH = ""

class RAGHandler(metaclass=abc.ABCMeta):
    def __init__(self, milvusClient):
        self.client = milvusClient

    @abc.abstractmethod
    def create_index(self, context):
        pass

    @abc.abstractmethod
    def search(self, query_text):
        pass

class BasicRAGHandler(RAGHandler):

    def __init__(self, milvusClient):
        super().__init__(milvusClient)

   
    def create_index(self, split_docs):
        # Generate embeddings for each split document
        #embeddings = [self.transformer.encode(doc).tolist() for doc in split_docs]
        #print("Created embeddings")
        embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {
            'input_text': True
        }
        }
        print("Creating embeddings")
        watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=os.getenv("PROJECT_ID"),
        params=embed_params,
        )
        embeddings = [watsonx_embedding.embed_query(doc.page_content) for doc in split_docs]
        print("Created embeddings")

        
        # Schema Definition
        schema = self.client.create_schema()
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="context", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embeddings", datatype=DataType.FLOAT_VECTOR, dim=768)

        # Define Index
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embeddings",
            index_type="FLAT", 
            metric_type="L2",
        )
        # Drop Collection
        self.client.drop_collection("Agentic_RAG_Collection")

        #Create Collection
        self.client.create_collection(
            collection_name="Agentic_RAG_Collection",
            schema=schema,
            index_params=index_params
        )
        print("Created Collection")

        # Prepare the data for insertion
        data_to_insert = [
            {
                "context": doc.page_content,
                "embeddings": emb
            }
            for doc, emb in zip(split_docs, embeddings)
        ]

        # Insert the data
        res = self.client.insert(
            collection_name="Agentic_RAG_Collection",
            data=data_to_insert
        )
        print("Inserted Data")

        # Load Collection
        self.client.load_collection(
            collection_name="Agentic_RAG_Collection"
        )

        print("Inserted data into the collection")
        
        vector_store_loaded = Milvus(
                        embeddings,
                        connection_args={"uri": SERVER_ADDR,"server_pem_path":SERVER_PEM_PATH,},
                        collection_name="langchain_example",
                    )
        
        #create retriever
        retriever = vector_store_loaded.as_retriever()
        return retriever





