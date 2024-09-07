from main import RAG_QDRANT_BEDROCK

# Create an instance of the RAG_QDRANT_BEDROCK class outside the callback
rqb = RAG_QDRANT_BEDROCK()

# Get user input from the console
user_input =''
# Use the existing instance of RAG_QDRANT_BEDROCK
output = rqb.query(user_input)

# Print the output
print(output)
