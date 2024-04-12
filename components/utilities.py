import streamlit as st
import openai
from streamlit.logger import get_logger
from langchain.tools import tool, Tool
from langchain.agents import AgentExecutor, create_react_agent, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_core.prompts import PromptTemplate
from bot import Movie
from langchain.docstore.document import Document
from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain

logger = get_logger(__name__)

@st.cache_data(show_spinner=False)
def is_open_ai_key_valid(api_key, model):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar")
        return False
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
    except Exception as e:
        st.error(f"{e.__class__.__name__}: {e}")
        logger.error(f"{e.__class__.__name__}: {e}")
        return False
    return True

# @st.cache_resource(hash_funcs={Movie: lambda m: m.title})
# def setup_generate(movie):
#     print(movie.llm)
#     qa_tools = [
#         Tool(
#             name="search private docs",
#             func=movie.retrieveRelevantDocs,
#             description="Useful for when you need to retrieve relevent documents about the quried movie",
#         )
#     ]
#     prompt = hub.pull("hwchase17/react")
#     # template = '''Answer the following questions as best you can. You have access to the following tools and chat history:

#     # tools:
#     # {tools}

#     # Use the following format:

#     # Question: the input question you must answer
#     # Thought: you should always think about what to do
#     # Action: the action to take, should be one of [{tool_names}]
#     # Action Input: the input to the action
#     # Observation: the result of the action
#     # ... (this Thought/Action/Action Input/Observation can repeat N times)
#     # Thought: I now know the final answer
#     # Final Answer: the final answer to the original input question

#     # Begin!

#     # Question: {input}
#     # Thought:{agent_scratchpad}'''

#     # prompt = PromptTemplate.from_template(template)
#     memory = ConversationBufferMemory(memory_key="chat_history")
#     agent = create_react_agent(movie.llm, qa_tools, prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=qa_tools, memory=memory, max_iterations=10)
#     return agent_executor

@st.cache_resource(hash_funcs={Movie: lambda m: m.title})
def setup_generate(movie):
    prompt_template = '''
    Given the following conversation chat history and a follow up question, reformulate the question \
    in the context of chat hostory.

    {chat_history}

    question: {question}
    '''

    reformulate_prompt = PromptTemplate.from_template(template=prompt_template)
    # qgen_promp
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    # question_generation_chain = LLMChain(llm=movie.llm, prompt=qgen_prompt)
    chain = ConversationalRetrievalChain.from_llm(
        llm=movie.llm.with_config(tags=["final_answer"]), 
        retriever=movie.retriever, 
        chain_type="stuff", 
        condense_question_llm=movie.llm.with_config(tags=["reformulated_prompt"]), 
        condense_question_prompt=reformulate_prompt, memory=memory
        )
    return chain


@st.cache_resource(experimental_allow_widgets=True)
def setup_search(movie_title, model):
    movie=Movie(movie_title, model)
    print(movie.title, model)
    with st.spinner("Indexing document... This may take a while⏳"):
        suggestions, plots = movie.wikiSuggestions(movie_title)
        print(suggestions)
        with st.expander("Title Suggestions(Disambiguation)", expanded=True):
            st.write(suggestions)    
    return movie, plots

@st.cache_resource(hash_funcs={Movie: lambda m: m.title, Document: lambda d:d.metadata["title"]})
def setup_database(movie, disamiguated_doc):
    if st.session_state.disambiguation_choice:
            logger.info("entered:"+str(st.session_state.disambiguation_choice))
            movie.wikiSummary(disamiguated_doc)
            movie.wikiCast(disamiguated_doc.metadata["source"])
            movie.fetchScript()
            movie.createVectorStore()
            logger.info("completed indexing")