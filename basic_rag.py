from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI as ai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

#from langchain_community.retrievers import WikipediaRetriever
#from langchain_community.retrievers import ContextualCompressionRetriever
#from langchain_community.retrievers.document_compressors import LLMChainExtractor
load_dotenv()

google_model = ai(model = "gemini-2.5-flash-lite", temperature = 1)
#Loading the base model for embedding
model_embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}
)
#Loading the files
textloader= TextLoader("fake_data.txt", encoding= "utf-8")
pdfloader= PyPDFLoader("unit4.pdf")
textFile = textloader.load()
pdfFile = pdfloader.load()
# print(textFile[0].page_content)
# print(pdfFile[0].page_content)

# Spilter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\\n\\n","\\n"," ",""],
    chunk_size = 750,
    chunk_overlap = 80
)
pdf_splitter = RecursiveCharacterTextSplitter(
    separators=["\\n\\n","\\n"," ",""],
    chunk_size = 750,
    chunk_overlap = 80
)
splitted_text = text_splitter.split_documents(textFile)
# print("-"*60)
# print(splitted_text[0])
# print(len(splitted_text))

#Creating the embedding and storing them in Chroma
db = Chroma.from_documents(
    documents=splitted_text,
    embedding=model_embedding,
    persist_directory="./rag_chroma_db",
    collection_name = "fake_data"
)
#results = db.similarity_search("what is streamlit", k=2)
#print(results)
#Creating a Retriever.
db_retriever = db.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":2, "fetch_k": 5, "lambda_mult": 0.5}
)
query = input("Chat and ask the question related to the pdf.")
source_context = db_retriever.invoke(query)
source_content = []
source_content = "\n\n".join([doc.page_content for doc in source_context])

text_prompt = '''You are an expert assistant. Use ONLY the provided context to answer the question.
If the answer is not in the context, say: "I don't have enough information to answer from the context."
Do NOT use outside knowledge. Do NOT make up facts.

Context:
{context}

Question:
{question}

Answer:
Provide a clear, accurate, and well-structured response based solely on the context provided. 
Use professional academic tone with proper terminology. Structure your answer logically with appropriate explanations.
If relevant, include key definitions, concepts, or examples from the context to support your answer.
Be concise yet comprehensive - aim for clarity over brevity. Maintain objectivity and precision in your explanation.
'''

prompt  = PromptTemplate(
    template = text_prompt,
    input_variables=["context","question"]
)
parser = StrOutputParser()
chain = prompt | google_model | parser
result = chain.invoke({
        'context':source_content[0],
        'question': query
    })
print(result)
#output = google_model.invoke(llm_prompt)
# print(parser(output))