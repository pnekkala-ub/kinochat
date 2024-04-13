import requests
import bs4
import re
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import WikipediaLoader
from pprint import pprint

class Movie:
    def __init__(self, title, model, temp) -> None:
        self.title = title
        self.script = ""
        self.synopsis = ""
        self.llm = ChatOpenAI(model=model, temperature=temp, streaming=True)
        self.themes = ""
        self.vectordb = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        # self.db_ids = []

    def fetchScript(self):
        """Fetch the movie script from IMSDB website."""
        imsdb_url = "https://imsdb.com/scripts/{}.html".format(self.title.replace(" ","-"))
        response = requests.get(imsdb_url)
        if response.ok:
            soup = bs4.BeautifulSoup(response.text, 'html.parser')
            script = soup.find('pre').get_text()
            self.script = re.sub("\n+","\n",re.sub("\r","",script)).lstrip()

    @staticmethod
    def wikiSuggestions(query):
        movies = []
        suggestions = ""
        docs = WikipediaLoader(query=query, load_max_docs=5, doc_content_chars_max=10000).load()
        ids = list(range(1,len(docs)+1))       
        for x in docs:
            movies.append(x.metadata["source"].split("/")[-1])
        for id, sug in zip(ids, movies):    
            suggestions += (str(id)+" - "+sug+"  \n")
        return suggestions, docs
       
    def wikiCast(self, wiki_url):
        response = requests.get(wiki_url)
        cast_list = []
        if response.ok:
            soup = bs4.BeautifulSoup(response.content, 'html.parser')
            try:
                cast = soup.find('span',id='Cast')
                cast_table = cast.find_next('ul')
            except AttributeError as e:
                cast = soup.find('span',id='Voice_cast')
                cast_table = cast.find_next('ul')
            for li in cast_table.find_all('li'):
                cast_list.append(li.text)
        movie_cast = "\n".join(cast_list)
        self.synopsis += ("\n"+movie_cast)

    def wikiSummary(self, document):
        """Fetch the movie summary from Wikipedia."""
        plot = list(filter(lambda x: "== Plot ==\n" in x, document.page_content.split("\n\n\n")))[-1].strip("== Plot ==\n")
        self.synopsis += ("Movie title: "+self.title+"\n\n"+plot)

    def plotSummary(self):
        """Summarize the IMSDB script."""
        tokenizer = Tokenizer("english")
        parser = PlaintextParser(self.script, tokenizer)
        lex_rank = LexRankSummarizer()
        mis = lex_rank(parser.document, sentences_count=100)
        summary = "\n".join([str(sent) for sent in mis if str(sent)])
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
        db_ids = list(range(len(docs)))
        self.vectordb = FAISS.from_documents(docs, self.embeddings, ids=db_ids)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k":5})

    def retrieveRelevantDocs(self, query:str) -> str:
        # """Retrieve relevent documents from the vector database."""
        self.retriever.search_kwargs['k']=5
        docs = self.retriever.invoke(query)
        docs = [doc.page_content for doc in docs]
        return "\n\n".join(docs)