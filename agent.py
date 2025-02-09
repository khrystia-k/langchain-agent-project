from langchain_openai import ChatOpenAI

from classifier import TopicClassifier
from rag import RAGPipeline

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool


class CustomAgent:
    def __init__(
        self,
        rag_pipeline_config,
        topic_classifier_config_eng,
        topic_classifier_config_ukr,
        llm_model="gpt-4o-mini",
        temperature=0,
        verbose=True,
    ):
        self.rag_pipeline = RAGPipeline(
            text_file=rag_pipeline_config["text_file"],
            vectorstore_file=rag_pipeline_config["vectorstore_file"],
            model_name=rag_pipeline_config.get("model_name", "gpt-4o-mini"),
            chunk_size=rag_pipeline_config.get("chunk_size", 1150),
            chunk_overlap=rag_pipeline_config.get("chunk_overlap", 150),
            temperature=rag_pipeline_config.get("temperature", 0),
        )

        self.topic_classifier_eng = TopicClassifier(
            base_model=topic_classifier_config_eng["base_model"],
            saved_model_path=topic_classifier_config_eng["saved_model_path"],
            num_labels=topic_classifier_config_eng.get("num_labels", 5),
            threshold=topic_classifier_config_eng.get("threshold", 0.6),
        )

        self.topic_classifier_ukr = TopicClassifier(
            base_model=topic_classifier_config_ukr["base_model"],
            saved_model_path=topic_classifier_config_ukr["saved_model_path"],
            num_labels=topic_classifier_config_ukr.get("num_labels", 5),
            threshold=topic_classifier_config_ukr.get("threshold", 0.6),
            class_names=topic_classifier_config_ukr.get("class_names"),
        )

        tools = [
            Tool(
                name="TopicClassifierEng",
                func=self.topic_classifier_eng.classify,
                description="Classifies the topic of the query in English",
            ),
            Tool(
                name="TopicClassifierUkr",
                func=self.topic_classifier_ukr.classify,
                description="Classifies the topic of the query in Ukrainian",
            ),
            Tool(
                name="RAG",
                func=self.rag_pipeline.answer,
                description="Answers questions about Khrystyna and her life.",
            ),
            Tool(
                name="Third tool",
                func=lambda *args,
                **kwargs: "The final answer is that the query is not relevant for classification or RAG tool",
                description="Use this tool in case using TopicClassifier and RAG is unneccessary. ",
            ),
        ]

        llm = ChatOpenAI(temperature=temperature, model=llm_model)

        template = """Answer the following questions as best as you can.
You're an agent designed for only two tasks: classifying text and answering questions about Khrystyna.
You have access to the following tools:  
{tools}

If the user requests another task, say that this task is not relevant to your tools.
When you call the TextClassifier tool, always return only the output of the classifier. If the output is "unknown," then return "unknown."

Use the following format:

Question: the input question you must answer  
Thought: you should always think about what to do  
Action: the action to take, should be one of [{tool_names}]  
Action Input: the input to the action  
Observation: the result of the action  
... (this Thought/Action/Action Input/Observation can repeat N times)  
Thought: I now know the final answer  
Final Answer: the final answer to the original input question  

Begin!

Question: {input}  
Thought: {agent_scratchpad}
"""

        prompt = ChatPromptTemplate.from_template(template)

        agent = create_react_agent(llm, tools=tools, prompt=prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools)

    def run(self, query):
        return self.agent.invoke({"input": query})


if __name__ == "__main__":
    rag_pipeline_config = {
        "text_file": "data_for_rag.txt",
        "vectorstore_file": "my_vectors",
        "model_name": "gpt-4o-mini",
        "chunk_size": 1150,
        "chunk_overlap": 150,
        "temperature": 0,
    }

    topic_classifier_config_eng = {
        "base_model": "roberta-base",
        "saved_model_path": "./saved_model_eng",
        "num_labels": 5,
        "threshold": 0.6,
    }

    topic_classifier_config_ukr = {
        "base_model": "youscan/ukr-roberta-base",
        "saved_model_path": "./saved_model_ukr",
        "num_labels": 5,
        "threshold": 0.5,
        "class_names": ["Політика", "Спорт", "Технології", "Розваги", "Бізнес"],
    }

    custom_agent = CustomAgent(
        rag_pipeline_config=rag_pipeline_config,
        topic_classifier_config_eng=topic_classifier_config_eng,
        topic_classifier_config_ukr=topic_classifier_config_ukr,
        llm_model="gpt-4o-mini",
        temperature=0,
        verbose=True,
    )

    query = "classify the text : I was at football yesterday"
    response = custom_agent.run(query)
    print(f"Agent Response: {response}")

    query = "класифікуй текст: Офіційний сайт турніру WTA 1000 в Індіан-Веллсі (США) оприлюднив заявковий лист.До заявки потрапили і чотири українські тенісистки: Марта Костюк, Еліна Світоліна, Даяна Ястремська та Ангеліна Калініна."
    response = custom_agent.run(query)
    print(f"Agent Response: {response}")
