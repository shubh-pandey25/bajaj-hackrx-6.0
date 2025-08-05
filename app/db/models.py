import pinecone

# Initialize the Pinecone client
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='YOUR_PINECONE_ENVIRONMENT')

# Create a Pinecone index
index_name = 'policy-qa'
pinecone.Index(index_name).create()

# Define a function to add vectors to the Pinecone index
def add_vectors_to_pinecone(vectors, metadata):
    pinecone.Index(index_name).upsert(vectors=vectors, metadata=metadata)

# Define a function to query the Pinecone index
def query_pinecone(query_vector, top_k=5):
    return pinecone.Index(index_name).query(vectors=query_vector, top_k=top_k)