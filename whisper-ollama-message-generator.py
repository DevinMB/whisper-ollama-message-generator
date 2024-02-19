from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import json
import time
import os
from dotenv import load_dotenv
import schedule
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.telegram import text_to_docs


load_dotenv()

source_topic_name = os.getenv('SOURCE_TOPIC_NAME')
destination_topic_name = os.getenv('DESTINATION_TOPIC_NAME')
bootstrap_servers = [os.getenv('BROKER')]  
group_id = os.getenv('GROUP_ID') 
ollama_path= os.getenv('OLLAMA_PATH')
model_name = os.getenv('MODEL_NAME')
query = os.getenv('QUERY') 
bot_token = os.getenv('BOT_TOKEN')
chat_id = os.getenv('CHAT_ID')

def generate_message():

    consumer = KafkaConsumer(
        source_topic_name,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest', 
        enable_auto_commit=True,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))  
    )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    consumer.subscribe([source_topic_name])
    # Poll to ensure the consumer receives its assignments
    consumer.poll(1000)
    consumer.assignment()  # Force the assignment refresh

    print("Collecting new messages...")
    dataset = []

    # Use a finite timeout for polling
    poll_timeout_ms = 10000  # Timeout in milliseconds
    while True:
        records = consumer.poll(poll_timeout_ms)
        if records:
            for tp, messages in records.items():
                for message in messages:
                    dataset.append(message.value)
                    if len(dataset) >= 10000:
                        print("Reached the dataset size limit.")
                        break
            if len(dataset) >= 10000:
                break
        else:
            # No new records within the timeout period
            print("No new messages. Moving to query.")
            break

    consumer.close()
    print(f"Collected {len(dataset)} messages for training.")
    
    # Load main ollama model for query and embedding model
    ollama = Ollama(base_url=ollama_path,
    model=model_name)
    oembed = OllamaEmbeddings(base_url=ollama_path, model=model_name)
    
    if len(dataset) > 0:
        ##messages I want embedded here
        texts = [d['message'] for d in dataset]  

        documents = text_to_docs(texts)

        # # Split content into chunks for embedding  
        print(f"splitting message documents..")  
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)

        # Embed messages into model using Chroma db vectorstore
        print(f"Embedding messages into model...")
        Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory="./chroma_db")

    # Retrieve Vector Store
    loaded_vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=oembed)


    print(f"Asking model for message...")
    print(f"Using this prompt: {query}")
    qachain=RetrievalQA.from_chain_type(ollama, retriever=loaded_vectorstore.as_retriever())

    response = qachain({"query": query})
    print(response)
    producer.send(destination_topic_name, value=response)
    producer.flush()
    

schedule.every().day.at("12:00").do(generate_message)

if __name__ == "__main__":
    generate_message()
    while True:
        schedule.run_pending()
        time.sleep(1) 