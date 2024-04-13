import streamlit as st
import openai
from streamlit.logger import get_logger
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_core.prompts import PromptTemplate
from bot import Movie
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain

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
    movie=Movie(movie_title, model, temp=st.session_state.temperature)
    with st.spinner("Indexing document... This may take a while⏳"):
        suggestions, plots = Movie.wikiSuggestions(movie_title)
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