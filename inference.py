import streamlit as st
import os

from agent import CustomAgent

# Configuration for RAG Pipeline
rag_pipeline_config = {
    "text_file": "data_for_rag.txt",
    "vectorstore_file": "vector_db",
    "model_name": "gpt-4o-mini",
    "chunk_size": 1150,
    "chunk_overlap": 150,
    "temperature": 0,
}

# Configuration for Topic Classifier in English
topic_classifier_config_eng = {
    "base_model": "roberta-base",
    "saved_model_path": "./saved_model_eng",
    "num_labels": 5,
    "threshold": 0.6,
}

# Configuration for Topic Classifier in Ukrainian
topic_classifier_config_ukr = {
    "base_model": "youscan/ukr-roberta-base",
    "saved_model_path": "./saved_model_ukr",
    "num_labels": 5,
    "threshold": 0.6,
    "class_names": ["Політика", "Спорт", "Технології", "Розваги", "Бізнес"],
}

# Initialize the agent only once
if "custom_agent" not in st.session_state:
    st.session_state.custom_agent = CustomAgent(
        rag_pipeline_config=rag_pipeline_config,
        topic_classifier_config_eng=topic_classifier_config_eng,
        topic_classifier_config_ukr=topic_classifier_config_ukr,
        llm_model="gpt-4o-mini",
        temperature=0,
        verbose=True,
    )


# Function to read text examples from directory and categorize them
def load_text_examples(directory):
    examples = {"English": {}, "Ukrainian": {}, "RAG": {}}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            category = filename.replace(".txt", "")
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                text = file.read()
                if any(
                    word in filename.lower()
                    for word in [
                        "business",
                        "sport",
                        "politics",
                        "technology",
                        "entertainment",
                    ]
                ):
                    examples["English"][category] = {
                        "name": f"Example for {category}",
                        "text": text,
                    }
                elif any(
                    word in filename.lower()
                    for word in ["бізнес", "спорт", "політика", "технології", "розваги"]
                ):
                    examples["Ukrainian"][category] = {
                        "name": f"Example for {category}",
                        "text": text,
                    }
                else:
                    examples["RAG"][category] = {
                        "name": f"Example for {category}",
                        "text": text,
                    }
    return examples


# Load examples from 'texts_examples' directory
example_texts = load_text_examples("texts_examples")

# Streamlit UI
st.set_page_config(
    page_title="Langchain Agent Project Demo", page_icon=":robot:", layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .stTextArea > div > textarea {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Langchain Agent Project Demo 👾")
st.markdown("Welcome! Select or edit an example query, or enter your own.")

# Display grouped examples
for group, categories in example_texts.items():
    st.markdown(f"# {group}")
    for category, example in categories.items():
        st.markdown(f"### {example['name']}")
        st.text_area("Text: ", example["text"], key=f"{group}_{category}", height=250)
        if st.button("**Run** ", key=f"run_{group}_{category}"):
            with st.spinner("Processing..."):
                response = st.session_state.custom_agent.run(example["text"])
                st.success(f"**Agent Response:** {response['output']}")
    st.markdown("---")

# Custom Query Input
st.markdown("## Enter Your Own Query")
query = st.text_input("Type your query here", placeholder="Type your query...")
if st.button("Run Custom Query"):
    if query:
        with st.spinner("Processing your query..."):
            response = st.session_state.custom_agent.run(query)
            st.success(f"**Agent Response:** {response['output']}")
    else:
        st.warning("Please enter a query to get a response.")

st.markdown("---")
st.markdown(
    "Made by Khrystyna Kuts | [My Github](https://github.com/khrystia-k)",
    unsafe_allow_html=True,
)
