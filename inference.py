import streamlit as st
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

# Configuration for Topic Classifier
topic_classifier_config = {
    "base_model": "roberta-base",
    "saved_model_path": "./saved_model",
    "num_labels": 5,
    "threshold": 0.6,
}

# Initialize the agent only once using Streamlit's session state
if 'custom_agent' not in st.session_state:
    st.session_state.custom_agent = CustomAgent(
        rag_pipeline_config=rag_pipeline_config,
        topic_classifier_config=topic_classifier_config,
        llm_model="gpt-4o-mini",
        temperature=0,
        verbose=True,
    )

# Pre-saved example texts
example_texts = [
    "Classify the text: The Samsung Galaxy S25 Ultra will be the flagship handset for the company's Galaxy AI software. Following the launch at the upcoming Galaxy Unpacked event, the S25 family, including the powerful Galaxy S25 Ultra, will be the basis for the development and growth of Galaxy AI through 2025 and beyond.",
    "Classify the text: Ukrainian tennis player Elina Svitolina won her 100th career match at the Grand Slam level, defeating American Caroline Dowlgade in two sets in the second round of the Australian Open. Also, tennis player Diana Jastremska reached the third round of the Australian Open 2025.",
    "Classify the text: CNN— A biopic about the complicated legacy of Michael Jackson was always going to be an ambitious undertaking. The planned release of the film, starring Jaafar Jackson as his late uncle, has been moved from April to October, reportedly due to complications with the script, according to a recent story from Puck. A representative for Lionsgate, the distributor of “Michael,” declined to comment when contacted by CNN. Here’s what we know about the project so far.",
    "When and where was Khrystyna born?",
    "What technical skills does Khrystyna have?",
    "What machine learning projects has Khrystyna done?",
    "What are hobbies of Khrystyna?"
   
]

# Streamlit UI
st.set_page_config(page_title="Langchain Agent Project Demo", page_icon=":robot:", layout="centered")


with st.container():
  
    st.title("Langchain Agent Project Demo 👾")
    st.markdown(
        """
        Welcome to the **My Custom Agent** interface! You can select one of the examples below or enter your own query to get a response.
        """
    )


    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Choose an Example Query Below:")
    with st.container():
        for idx, example in enumerate(example_texts):
            with st.expander(f"**Example {idx+1}:**", expanded=True):
                st.write(f"**Text:**")
                st.markdown(f"```{example}```")
               
                if st.button(f"Run Example {idx+1}", key=idx, help="Click to run the example", use_container_width=True):
                    with st.spinner('Processing...'):
        
                        response = st.session_state.custom_agent.run(example)
                        st.success(f"**Agent Response:** {response}")
                st.markdown("<hr>", unsafe_allow_html=True)

 
    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown("### **Or Enter Your Own Query:**")
    query = st.text_input("Type your query here", key="custom_query", placeholder="Type your query...", help="Enter your query to classify or get a response")

   
    if st.button("Run Custom Query", key="run_custom"):
        if query:
            with st.spinner('Processing your query...'):
            
                response = st.session_state.custom_agent.run(query)
                st.success(f"**Agent Response:** {response}")
        else:
            st.warning("Please enter a query to get a response.")


    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("Made  by Khrystyna Kuts | [My Github](https://github.com/khrystia-k)", unsafe_allow_html=True)
