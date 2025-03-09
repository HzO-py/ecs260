import openai
from llama_index.core import SimpleDirectoryReader,GPTVectorStoreIndex

def generate_response(query):
    retrieved_info = query_engine.query(query)
    prompt = f"Using the following information: {retrieved_info}, please answer the question: {query}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Example usage
query = "How do I reset my device?"
print(generate_response(query))

# documents = ["user_manual_1.txt", "user_manual_2.txt"] # This is your custom data
openai.api_key = "key"
reader = SimpleDirectoryReader("./dataset/example")

index = GPTVectorStoreIndex.from_documents(reader.load_data())
# Initialize the query engine
query_engine = index.as_query_engine(similarity_top_k=3)
# Example query
query = "How do I reset my device?"
response = query_engine.query(query)
print("test" + response)

