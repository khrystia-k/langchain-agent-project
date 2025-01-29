from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai import ChatOpenAI

from classifier import TopicClassifier
from rag import RAGPipeline


class CustomAgent:
    def __init__(
        self,
        rag_pipeline_config,
        topic_classifier_config,
        llm_model="gpt-4o-mini",
        temperature=0,
        verbose=True,
    ):
        """
        Initializes the custom agent with RAGPipeline, TopicClassifier, and the LangChain agent.
        """
        # Instantiate RAGPipeline
        self.rag_pipeline = RAGPipeline(
            text_file=rag_pipeline_config["text_file"],
            vectorstore_file=rag_pipeline_config["vectorstore_file"],
            model_name=rag_pipeline_config.get("model_name", "gpt-4o-mini"),
            chunk_size=rag_pipeline_config.get("chunk_size", 1150),
            chunk_overlap=rag_pipeline_config.get("chunk_overlap", 150),
            temperature=rag_pipeline_config.get("temperature", 0),
        )

        # Instantiate TopicClassifier
        self.topic_classifier = TopicClassifier(
            base_model=topic_classifier_config["base_model"],
            saved_model_path=topic_classifier_config["saved_model_path"],
            num_labels=topic_classifier_config.get("num_labels", 5),
            threshold=topic_classifier_config.get("threshold", 0.6),
        )

        # Define tools
        tools = [
            Tool(
                name="TopicClassifier",
                func=self.topic_classifier.classify,
                description="Classifies the topic of the query. Never change the classifier's response. If you call this tool, always return only its response",
            ),
            Tool(
                name="RAG",
                func=self.rag_pipeline.answer,
                description="Answers questions about Khrystyna and her life.",
            ),
        ]

        # Initialize LLM
        llm = ChatOpenAI(temperature=temperature, model=llm_model)

        # Initialize LangChain agent
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
        )

    def run(self, query):
        """
        Runs the agent with the given query.
        """
        return self.agent.run(query)


# Example usage
if __name__ == "__main__":
    # Configuration for RAGPipeline
    rag_pipeline_config = {
        "text_file": "my.txt",
        "vectorstore_file": "my_vectors",
        "model_name": "gpt-4o-mini",
        "chunk_size": 1150,
        "chunk_overlap": 150,
        "temperature": 0,
    }

    # Configuration for TopicClassifier
    topic_classifier_config = {
        "base_model": "roberta-base",
        "saved_model_path": "./saved_model",
        "num_labels": 5,
        "threshold": 0.6,
    }

    # Create the custom agent
    custom_agent = CustomAgent(
        rag_pipeline_config=rag_pipeline_config,
        topic_classifier_config=topic_classifier_config,
        llm_model="gpt-4o-mini",
        temperature=0,
        verbose=True,
    )

    # Test the agent
    query = "classify the text : I was at football yesterday"
    response = custom_agent.run(query)
    print(f"Agent Response: {response}")
