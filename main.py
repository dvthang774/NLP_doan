from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate


template = """Question: {question}

Answer: Answer by vietnamese"""

prompt = PromptTemplate.from_template(template)
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/user/Desktop/newProOpenCV/gpt4all/mistral-7b-openorca.gguf2.Q4_0.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

llm_chain = prompt | llm

question = "Power Ranger Lost Galaxy"
llm_chain.invoke({"question": question[1]})