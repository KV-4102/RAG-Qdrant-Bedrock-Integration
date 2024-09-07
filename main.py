import qdrant_client as qc
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM
from qdrant_client.http.models import *
from dotenv import load_dotenv, dotenv_values
import json
import boto3
import torch
import random

class RAG_QDRANT_BEDROCK:
    def __init__(self):
        self.name = 'RAG_QDRANT_BEDROCK'
        self.description = 'Retrieve and Generate with Qdrant and Bedrock'
        load_dotenv()
        values_env = dotenv_values(".env")
        URL = values_env['URL']
        COLLECTION_NAME = values_env['COLLECTION_NAME']
        DIMENSION = int(values_env['DIMENSION'])
        model_name = "anferico/bert-for-patents"
        self.URL = URL
        self.COLLECTION_NAME = COLLECTION_NAME
        self.DIMENSION = DIMENSION
        self.model_name = model_name

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = BertForMaskedLM.from_pretrained(model_name, config=config).to("cpu")

    def get_qdrant_client(self):
        client = qc.QdrantClient(url=self.URL)
        return client

    def retrieve(self, query: str) -> str:
        client = self.get_qdrant_client()

        # Tokenize the query
        encoded_input = self.tokenizer(query, return_tensors="pt", padding=False, truncation=True, max_length=512).to(
            "cpu")

        with torch.no_grad():
            outputs = self.model(**encoded_input)

        # Get the pooled output (contextualized representation of the entire paragraph)
        last_hidden_state = outputs['hidden_states'][-1]
        response = torch.mean(last_hidden_state, axis=1).cpu()
        embedding_array = response.squeeze().detach().numpy()
        xq = embedding_array

        search_result = client.search(collection_name=self.COLLECTION_NAME,
                                      query_vector=xq,
                                      query_filter=None,
                                      limit=20)

        contexts = []
        for result in search_result:
            contexts.append(result.payload['snippet'])

        # Shuffle the contexts
        #random.shuffle(contexts)

        # Join the shuffled contexts into a single string
        shuffled_context = "\n---\n".join(contexts)

        return shuffled_context

    def generate_completion(self, context: str, query: str) -> str:
        prompt_data = f"""Human:Rank all of these 20 candidate sentences in context in order of similarity based on the user input query:
        'context is:'{context}\n\n 'query is:'{query}
        Assistant:"""

        body = json.dumps({"prompt": prompt_data, })
        modelId = "meta.llama2-13b-chat-v1"
        accept = "application/json"
        contentType = "application/json"

        bedrock_runtime_client = boto3.client('bedrock-runtime',
                                              aws_access_key_id="",
                                              aws_secret_access_key="",
                                              region_name='')

        response = bedrock_runtime_client.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())

        response_text = response_body["generation"]
        return response_text

    def query(self, query: str) -> str:
        try:
            context = self.retrieve(query)
            completion = self.generate_completion(context, query)
            #print("query", query)
            #print("context", context)
            #print("completion", completion)
            return completion
        except Exception as e:
            # Print the exception
            print(f"An error occurred: {e}")
            return "Error: Something went wrong."
