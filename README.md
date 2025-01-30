# LangChain Agent Project 🚀

## Overview
This project is a LangChain-based agent with two main tools:
1. **Text Classifier**: A fine-tuned RoBERTa model that classifies texts into five categories:
   - Politics
   - Sport
   - Technology
   - Entertainment
   - Business
2. **Retrieval-Augmented Generation (RAG) System**: Answers questions about Khrystyna using information from `data_for_rag.txt`.

## Installation
To set up the project, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `saved_model/` - Directory containing saved fine-tuned RoBERTa model for classification.
- `texts_examples/` - Example texts for testing the classifier.
- `vector_db/` - Directory storing the vector database for RAG retrieval.
- `data_for_rag.txt` - Source document used by the RAG system to answer questions about Khrystyna.
- `EDA.ipynb` - Exploratory Data Analysis (EDA) notebook for understanding the dataset before training.
- `fine_tuning.ipynb` - Notebook for fine-tuning the RoBERTa model for classification.
- `agent.py` - Class of the LangChain agent, integrating both tools.
- `classifier.py` - Class of the text classifier.
- `rag.py` - Implements the RAG system.
- `inference.py` - Main script for running inference of the agent.

## Usage
Run the LangChain agent:
```bash
streamlit run inference.py
```


## Project Stages
### 1. **Data Preparation**
   - Collected data for classification. Dataset source: [Kaggle - Text Classification Documentation](https://www.kaggle.com/datasets/tanishqdublish/text-classification-documentation)
   - Conducted EDA using `EDA.ipynb` to understand data distributions and patterns. Checked for NaNs and duplicated values.
   - Created `data_for_rag.txt` as a knowledge base for the RAG system.

### 2. **Model Fine-Tuning**
   - Used roberta-base for fine-tuning on classification tasks. (`fine_tuning.ipynb`)
   - Сhose evaluation metric: F1 score, as the task was classification.
   - Saved the fine-tuned model in `saved_model/`.

### 4. **Building the LangChain Agent**
  - Built RAG pipeline in `rag.py`
  - Implemented class for text classifier in `classifier.py`
  - Implemented an agent in agent.py that integrates the classifier and RAG as tools

### 5. **Inference and Testing**
   - Verified performance on sample inputs.
   - UI implemented with Streamlit for interactive use in `inference.py`



