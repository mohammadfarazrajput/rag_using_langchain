from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI as ai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts import concise_prompt
from dotenv import load_dotenv

load_dotenv()
print("API LODED SUCCESSFULY!")

google_model = ai(model = "gemini-2.5-flash-lite", temperature = 1)
#Loading the base model for embedding
model_embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}
)
print ("--"*40)
print("Loaded MODEL Successfully!")
#Loading the files
document_type = int(input("1. Text File/n2.PDF file."))
document_name = input("File name with extension")
if document_type == 1:
    textloader= TextLoader(f"{document_name}", encoding= "utf-8")
    loader = textloader.load()
else:    
    pdfloader= PyPDFLoader(f"{document_name}")
    loader = pdfloader.load()
# Choosing the right split and chunk_size
if document_type == 0:
    chunk_size = 800
    chunk_overlap = 95
else:
    chunk_size = 1200
    chunk_overlap = 150
# Spilter
splitter = RecursiveCharacterTextSplitter(
    separators=["\\n\\n","\\n"," ",""],
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap
)
splitted_text = splitter.split_documents(loader)
print(f"Successfully! Created the Chunks for the document.")
#Creating the embedding and storing them in Chroma
try:
    db = Chroma(
        persist_directory="./rag_chroma_db",
        embedding=model_embedding,
        collection_name=document_name
    )
    if db._collection.count() == 0:
        db.add_documents(splitted_text)
except:
    db = Chroma.from_documents(
        documents=splitted_text,
        embedding=model_embedding,
        persist_directory="./rag_chroma_db",
        collection_name=document_name
    )
print("-"*80)
print(f"Total documents in DB: {db._collection.count()}")
print(f"Collection name: {document_name}")
print("-"*80)
#Creating a Retriever.
db_retriever = db.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":3, "fetch_k": 5, "lambda_mult": 0.5}
)
quit = False
while not quit:
    query = input("Chat with uploaded document.")
    source_context = db_retriever.invoke(query)
    source_content = "\n\n".join([doc.page_content for doc in source_context])
    print(f"Retrieved {len(source_context)} documents")
    print(f"First result preview: {source_context[0].page_content[:200] if source_context else 'No results'}")
    print("-"*80)
    text_prompt = concise_prompt()

    prompt  = PromptTemplate(
        template = text_prompt,
        input_variables=["context","question"]
    )
    parser = StrOutputParser()
    chain = prompt | google_model | parser
    result = chain.invoke({
            'context':source_content,
            'question': query
        })
    print(result)
    wanna_quit = input("To quit press 'Q'.")
    if wanna_quit.lower() == 'q':
        quit = True 