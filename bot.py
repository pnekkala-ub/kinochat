import os
from uuid import UUID
import requests
import bs4
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings,ChatOpenAI
from langchain.tools import tool, Tool
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_community.document_loaders import WikipediaLoader
import wikipedia
from pprint import pprint
import asyncio
from typing import Any, Dict, List

# class StreamingAsyncCallbackHandler(AsyncCallbackHandler):
#     async def on_llm_start(self, serialized: Dict[str, FAISS], prompts: List[str], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, FAISS] | None = None, **kwargs: FAISS) -> None:
#         return await super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)

class Movie:
    def __init__(self, title, model) -> None:
        self.title = title
        self.script = ""
        self.synopsis = ""
        self.llm = ChatOpenAI(model=model, temperature=0, streaming=True)
        self.themes = ""
        self.vectordb = None
        self.db_ids = []

    def fetchScript(self):
        """Fetch the movie script from IMSDB website."""
        imsdb_url = "https://imsdb.com/scripts/{}.html".format(self.title.replace(" ","-"))
        response = requests.get(imsdb_url)
        if response.ok:
            soup = bs4.BeautifulSoup(response.text, 'html.parser')
            script = soup.find('pre').get_text()
            processed_script = re.sub("\n+","\n",re.sub("\r","",script)).lstrip()
            with open(self.title,"w") as f:
                f.write(processed_script)
        if os.path.exists(os.getcwd()+"\\"+self.title):
            with open(self.title, "r") as f:
                self.script = f.read()

    def wikiSuggestions(self, query):
        movies = []
        suggestions = ""
        docs = WikipediaLoader(query=query, load_max_docs=5, doc_content_chars_max=10000).load()
        ids = list(range(1,len(docs)+1))       
        for x in docs:
            movies.append(x.metadata["source"].split("/")[-1])
        for id, sug in zip(ids, movies):    
            suggestions += (str(id)+" - "+sug+"\n")
        return suggestions, docs
       
    def wikiCast(self, wiki_url):
        response = requests.get(wiki_url)
        if response.ok:
            soup = bs4.BeautifulSoup(response.content, 'html.parser')
            cast = soup.find('span',id='Cast')
            cast_table = cast.find_next('ul')
            cast_list = []
            for li in cast_table.find_all('li'):
                actor_name = li.text.split(' as ')[0].strip()
                character_name = li.text.split(' as ')[1].strip()
                cast_list.append(actor_name + " as " + character_name)
        movie_cast = "\n".join(cast_list)
        self.synopsis += ("\n"+movie_cast)

    def wikiSummary(self, document):
        """Fetch the movie summary from Wikipedia."""
        plot = list(filter(lambda x: "== Plot ==\n" in x, document.page_content.split("\n\n\n")))[-1].strip("== Plot ==\n")
        self.synopsis += ("Movie title: "+self.title+"\n\n"+plot)
        print(self.synopsis)

    def plotSummary(self):
        """Summarize the IMSDB script."""
        tokenizer = Tokenizer("english")
        parser = PlaintextParser(self.script, tokenizer)
        lex_rank = LexRankSummarizer()
        mis = lex_rank(parser.document, sentences_count=100)
        summary = "\n".join([str(sent) for sent in mis if str(sent)])
        with open(self.title+"_summary","w") as f:
            f.write(summary)
        with open(self.title+"_summary","r") as f:
            summary = f.read()
        prompt_template  =   """You will be given a series of sentences from a source text. Your goal is to give a summary.
        The sentences will be enclosed in triple backtrips (```).


        sentences :
        ```{text}```

        SUMMARY :"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        self.synopsis += self.llm(prompt=prompt.format(text=summary))

    def movieThemes(self):
        """Extract the themes of the movie."""
        prompt_template = """You will be given a summary of a movie script. Your goal is to describe the plot themes from the summarykj nn.
        The summary of the movie script will be enclosed in triple backtrips (```).

        summary :
        ```{text}```
        
        plot themes :"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        theme_chain = prompt | self.llm
        self.themes = theme_chain.stream({"text":self.synopsis})

    def createVectorStore(self):
        """Create a Vector Databse."""
        documents=[]
        if self.script:
            documents.append(self.script.replace("\n",""))
        else:
            documents.append(self.synopsis.replace("\n",""))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs = text_splitter.create_documents(documents)
        self.db_ids = list(range(len(docs)))
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vectordb = FAISS.from_documents(docs, embeddings, ids=self.db_ids)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k":5})

    def retrieveRelevantDocs(self, query:str) -> str:
        # """Retrieve relevent documents from the vector database."""
        self.retriever.search_kwargs['k']=5
        docs = self.retriever.invoke(query)
        docs = [doc.page_content for doc in docs]
        return "\n\n".join(docs)