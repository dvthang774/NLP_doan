from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackManager
from langchain_community.llms import LlamaCpp, GPT4All
from langchain_community.embeddings import LlamaCppEmbeddings, GPT4AllEmbeddings
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# instantiate the LLM and embeddings models
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="/Users/user/Desktop/newProOpenCV/gpt4all/mistral-7b-openorca.Q4_0.gguf",
#     temperature=0.75,
#     max_tokens=2000,
#     top_p=1,
#     callback_manager=callback_manager,
#     verbose=True,  # Verbose is required to pass to the callback manager
# )

# embeddings = LlamaCppEmbeddings(model_path="mistral-7b-openorca.Q4_0.gguf")
embeddings = GPT4AllEmbeddings()

documents = PyPDFLoader('/Users/user/Desktop/newProOpenCV/gpt4all/demo.pdf').load_and_split()
# text_splitter =  RecursiveCharacterTextSplitter(
#         chunk_size=50, chunk_overlap=50, separator="\n"
#     )
chunk_size = 500
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
texts = text_splitter.split_documents(documents)
# gpt4all_embd = GPT4AllEmbeddings()
faiss_index = FAISS.from_documents(texts, embeddings)
faiss_index.save_local("/Users/user/Desktop/newProOpenCV/gpt4all/faiss")

# load vector store
print("loading indexes")
faiss_index = FAISS.load_local("/Users/user/Desktop/newProOpenCV/gpt4all/faiss", embeddings, allow_dangerous_deserialization=True)
print("index loaded")
# gpt4all_path = './mistral-7b-openorca.Q4_0.gguf'

question = "Khoa HUET nằm  ở đâu"
matched_docs = faiss_index.similarity_search(question, 4)
context = ""
for doc in matched_docs:
    context = context + doc.page_content + " \n\n "


template = """
Context: {context}
 - -
Question: {question}"""

# llm_chain = LLMChain(prompt=prompt, llm=llm)
# conversation = ConversationChain(
#     llm=llm, verbose=True, memory=ConversationBufferMemory()
# )
# qa = faiss_index.as_retriever(k=4)
# response = qa.invoke("Khoa HUET ở đâu")
# print(response)


callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(model_path='/Users/user/Desktop/newProOpenCV/gpt4all/mistral-7b-openorca.gguf2.Q4_0.gguf',n_ctx=2048, callback_manager=callback_manager, verbose=True,repeat_last_n=0)
prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(question)
print(llm_chain.invoke(question))