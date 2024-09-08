**Retrieval Augmented Generation System using Qdrant and LLAMA2 on Bedrock**

This project implements a retrieval augmented generation system utilizing Qdrant for retrieval and LLAMA2 on Bedrock for text generation. It uses a Large Sentence BERT model for text embeddings.


**Installation**

Clone the repository:

git clone <repository_url>

cd <repository_directory>

Install dependencies:
pip install -r requirements.txt

Set up environment variables:
Create a .env file in the root directory and define the following variables:

URL="<your_qdrant_server_url"

DIMENSION=<Qdrant_collection_dimension>

COLLECTION_NAME="<Qdrant_Collection_name>"

METRIC_NAME="<distance_metric>"


**Usage**

Run the app3.py script:
python app3.py
Enter the input query when prompted.

The system will retrieve relevant contexts using Qdrant and generate completions using LLAMA2 on Bedrock.


**Files**

app3.py: Main script to interact with the retrieval augmented generation system.

main.py: Contains the implementation of the RAG_QDRANT_BEDROCK class.


**Dependencies**

qdrant_client: Python client library for Qdrant.

boto3: AWS SDK for Python.
