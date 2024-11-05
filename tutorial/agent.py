#%%
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
#%%
# Create the agent
search = TavilySearchResults(max_results=2)
search_results = search.invoke("How is the weather in Londrina - Paraná - Brazil?")
tools = [search]
#%%
search_results[0]
# %%
model = ChatOpenAI(model="gpt-4o-mini")
# %%
response = model.invoke([HumanMessage("How is the weather in Londrina - Paraná - Brazil?")])    
# %%
response.content
# %%
model_with_tools = model.bind_tools(tools)
# %%
response = model_with_tools.invoke([HumanMessage(content="Hi!")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
# %%
response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
# %%
