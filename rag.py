import os

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


class RAGPipeline:
    def __init__(
        self,
        text_file,
        vectorstore_file,
        model_name="gpt-4o-mini",
        chunk_size=1150,
        chunk_overlap=150,
        temperature=0,
    ):
        load_dotenv()

        self.text_loader = TextLoader(text_file)
        self.documents = self.text_loader.load()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[" ", ",", "\n"],
        )
        self.text_chunks = self.text_splitter.split_documents(self.documents)
        self.embeddings = OpenAIEmbeddings()

        if os.path.isfile(vectorstore_file):
            self.vectorstore = FAISS.load_local(vectorstore_file, self.embeddings)
        else:
            self.vectorstore = FAISS.from_documents(self.text_chunks, self.embeddings)
            self.vectorstore.save_local(vectorstore_file)

        self.retriever = self.vectorstore.as_retriever()
        self.model = ChatOpenAI(temperature=temperature, model=model_name)

        self.prompt_template = """ You are a helpful assistant. 
        Your task: Answer the question based only on the following context.\
                If you don't know, answer: "I don’t know." 
        Context: {context}
        Question: {question}
        """

    @staticmethod
    def prepare_context(docs):
        context = "".join(doc.page_content for doc in docs)
        return context

    def answer(self, query):
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        chain = (
            {
                "context": self.retriever | RunnableLambda(self.prepare_context),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.model
            | StrOutputParser()
        )
        return chain.invoke(query)


if __name__ == "__main__":
    pipeline = RAGPipeline(
        text_file="my.txt",
        vectorstore_file="my_vectors",
        model_name="gpt-4o-mini",
        chunk_size=1150,
        chunk_overlap=150,
        temperature=0,
    )

    query = "What is Khrystyna's hobbies?"
    answer = pipeline.answer(query)
    print(f"Answer: {answer}")
