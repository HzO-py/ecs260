import os
import time
import openai
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

openai.api_key = ""
client = openai

dataset_folder = "../dataset"
rag_folder = os.path.join(dataset_folder, "RAG_processed")
index_path = os.path.join(dataset_folder, "indexes")
os.makedirs(index_path, exist_ok=True)
embed_model = OpenAIEmbedding(client=client)

def process_file(file_number, max_retries=3):
    filename = f"{file_number}.txt"
    file_path = os.path.join(rag_folder, filename)
    index_save_path = os.path.join(index_path, f"index_{file_number}.json")

    if os.path.exists(index_save_path):
        print(f"Index {index_save_path} exists, skip")
        return

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist, skip")
        return

    retries = 0
    delay = 5  
    while retries <= max_retries:
        try:
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            index = GPTVectorStoreIndex.from_documents(documents, embed_model=embed_model)
            
            index.storage_context.persist(index_save_path)

            print(f"{file_path} index is saved to {index_save_path}")
            break  
        except Exception as e:
            error_str = str(e)
            if "rate_limit_exceeded" in error_str:
                retries += 1
                print(f"{file_path} waits again for {delay} seconds ...")
                time.sleep(delay)
            else:
                print(f"Processing {file_path} fail: {e}")
                break

def get_sorted_files(rag_folder, size_limit=10 * 1024 * 1024):
    files = []

    for file in os.listdir(rag_folder):
        file_path = os.path.join(rag_folder, file)
        if os.path.isfile(file_path) and file.endswith(".txt"):
            file_size = os.path.getsize(file_path)
            files.append((file, file_size))

    files.sort(key=lambda x: x[1])

    selected_files = []
    for file, size in files:
        if size > size_limit:
            break
        selected_files.append(int(file.split('.')[0]))

    print(f"Select {len(selected_files)} files")
    return selected_files

if __name__ == '__main__':
    sorted_files = get_sorted_files(rag_folder)
    print(sorted_files)

    for file_name in sorted_files:
        process_file(file_name)