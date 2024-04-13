import streamlit as st
from components.sidebar import sidebar
from components.utilities import is_open_ai_key_valid, setup_generate, setup_search, setup_database
from streamlit.logger import get_logger
import asyncio
from pprint import pprint

logger = get_logger(__name__)

MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]
# load_dotenv()
st.set_page_config(page_title="kinoscript", page_icon="🎬", layout="wide")
st.header("🎬kinoscript.ai")

sidebar()

openai_api_key = st.session_state.get("OPENAI_API_KEY")
if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar. You can get a key at"
        " https://platform.openai.com/account/api-keys."
    )

st.selectbox("Model", key="model", options=MODEL_LIST)

if not is_open_ai_key_valid(openai_api_key, st.session_state.model):
    st.stop()

if "disambiguation_choice" not in st.session_state:
    st.session_state.disambiguation_choice = 0

if "number_widget" not in st.session_state:
    st.session_state.number_widget = 0

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    border = False
else:
    border = True

if "themes" not in st.session_state:
    st.session_state.themes = ""


def clear_disambiguation_choice():
    st.session_state.disambiguation_choice = 0
    st.session_state.messages = []
    st.session_state.themes = "" 

def on_disambiguation_choice_submit():
    st.session_state.disambiguation_choice = st.session_state.number_widget
    st.session_state.number_widget = None

async def generate(chain, prompt, placeholder, temperature):
    stream = ""
    async for event in chain.astream_events({"question":prompt}, version="v1",kwargs= {"temperature":temperature}):
        if event["event"] == "on_chat_model_stream" and "final_answer" in event["tags"]:
            chunk = event["data"]["chunk"].content
            stream+=chunk
            placeholder.chat_message("assistant").write(stream)
    return stream

def chat_content(widget):
    with widget:
        with st.chat_message("user"):
            st.write(st.session_state.content)
        placeholder = st.empty()
        response = asyncio.run(generate(chain, st.session_state.content, placeholder, st.session_state.temperature))
        st.session_state.messages.append({"role":"user", "content":st.session_state.content})
        st.session_state.messages.append({"role": "assistant", "content": response})

def theme_content(movie, widget, placeholder):
    movie.movieThemes()
    with widget:
        stream = ""
        for chunk in movie.themes:
            stream += chunk.content
            placeholder.write(stream)
        st.session_state.themes = stream

movie_title = st.text_input(
    "Movie Name", 
    placeholder="Enter movie name", 
    key="movie_title", 
    help="Enter the title of the movie as accurately as you can",
    on_change=clear_disambiguation_choice
    )
if not movie_title:
    st.stop()

if st.session_state.movie_title:
    movie, plots = setup_search(movie_title, st.session_state.model)
    st.number_input(
        "picks",
        min_value=1,
        max_value=10, 
        value=None, 
        key="number_widget", 
        on_change=on_disambiguation_choice_submit, 
        placeholder="Pick a number", 
        help="Choose a number corresponding to the desired movie title"
        )
    if not st.session_state.disambiguation_choice:
        st.stop()
    setup_database(movie, plots[st.session_state.disambiguation_choice-1])
    
    lcol, rcol = st.columns(2)
    
    with lcol:
        chain = setup_generate(movie)
        if not st.session_state.messages:
            chain.memory.clear()
        history_container = st.container(border=border)
        with history_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        with st.container():
            st.chat_input(key='content', on_submit=chat_content, kwargs={"widget":history_container})     
            
    with rcol:
        theme_placeholder = st.empty()
        theme_placeholder.write(st.session_state.themes)
        st.button(label="Thematic analysis", key="theme_button", on_click=theme_content, use_container_width=True, kwargs={"movie":movie, "widget":rcol, "placeholder": theme_placeholder})


# for key in st.session_state.keys():
#     del st.session_state[key]