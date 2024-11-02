#%%
import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#%%
load_dotenv(dotenv_path=".env", override=True)

#%%
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

#%%
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

result = model.invoke(messages)
parser.invoke(result)

# %%
chain = model | parser
chain.invoke(messages)
# %%

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)  # input e um dicionario
# %%
result = prompt_template.invoke({"language": "italian", "text": "hi"})

result  # retorna um ChatPromptValue

result.to_messages()
# %%

chain = prompt_template | model | parser    

chain.invoke({"language": "portuguese", "text": "what you mean trained?"})

# %%
