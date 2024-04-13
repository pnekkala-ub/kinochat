import streamlit as st
import os

def sidebar():
    with st.sidebar:
        with st.expander("How To", expanded=False):
            st.markdown(
                "## How to use\n"
                "1. Enter your OpenAI API key\n"
                "2. Enter the movie title\n"
                "3. Select the desired title from the list of suggestions to the right\n"
                "4. Click the thematic analysis widget for an analysis of the movie themes\n"
                "5. Ask a question about the movie in chat\n"
            )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )

        st.session_state["OPENAI_API_KEY"] = api_key_input
        st.slider("temperature", key="temperature", min_value=0., max_value=1.0, value=0., step=0.1, format="%f", help="larger temperature generates more creative responses")
        with st.expander("FAQ", expanded=False):
            st.markdown(
            """
            # FAQ
            ## How does kinoscript.io work?
            You can extract thematic insights from the plot of the film and ask questions
            about the characters and plot from the film.

            For thematic analysis, kinoscript.io performs a summarization of the script
            and prompts the language model to deliver insights from the summary.

            When the user asks a question relevant to the film, kinoscript.io fetches the
            relevent chunks from the script and the language model uses the chunks as 
            the context to provide answers.

            ## Is my data safe?
            Yes, your data is safe. kinoscript.io does not store your API keys or
            questions. All uploaded data is deleted after you close the browser tab.

            ## Why does it take so long to get answers?
            If you are using a free OpenAI API key, it will take a while to index the whole
            script. This is because the free API key has strict [rate limits] (https://platform.openai.com/docs/guides/rate-limits/overview).
            To speed up the indexing process, you can use a paid API key. Furthermore, as the
            agent has to perform multiple steps of actions to arrive at an answer, it takes a
            while to generate the response, but once text is generated, it is streamed.

            ## Are the answers 100% accurate?
            No, the answers are not 100% accurate. kinoscript.io uses GPT-3.5 to generate
            answers. GPT-3.5 is a powerful language model, but it sometimes makes mistakes 
            and is prone to hallucinations. Also, kinoscripot.ai uses semantic search
            to find the most relevant chunks and does not see the entire script,
            which means that it may not be able to find all the relevant information and
            may not be able to answer all questions (especially summary-type questions
            or questions that require a lot of context from the document). Always check 
            with the downloaded sourcescript to make sure that the answers are correct.
            """
        )
