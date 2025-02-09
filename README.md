# LangChain Agent Project 🚀

## Overview  📌
This project is a LangChain-based agent with two main tools:
1. **English Text Classifier**: A fine-tuned RoBERTa base model that classifies texts into five categories:
   - Politics
   - Sport
   - Technology
   - Entertainment
   - Business
2. **Ukrainian Text Classifier**:  A fine-tuned RoBERTa base model for ukrainian language that classifies texts into five categories:
   -  Політика
   -  Спорт
   -  Технології
   -  Розваги
   -  Бізнес
3. **Retrieval-Augmented Generation (RAG) System**: Answers questions about Khrystyna using information from `data_for_rag.txt`.

## Installation ⚙️
To set up the project, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🚀
Run the LangChain agent:
```bash
streamlit run inference.py
```

## Project Structure 🗂
- `saved_model_eng/` - Directory containing the fine-tuned RoBERTa model for English classification.
- `saved_model_ukr/` - Directory containing the fine-tuned Ukrainian model.
- `texts_examples/` - Example texts for testing the agent.
- `vector_db/` - Directory storing the vector database for RAG retrieval.
- `data_for_rag.txt` - Source document used by the RAG system to answer questions about Khrystyna.
- `EDA.ipynb` - Exploratory Data Analysis (EDA) notebook for understanding the dataset before training.
- `fine_tuning.ipynb` - Notebook for fine-tuning the RoBERTa model for English classification.
- `fine_tuning_ukr.ipynb` - Notebook for fine-tuning RoBERTa model for Ukrainian classification.
- `translation.ipynb` - Notebook handling of the dataset from English to Ukrainian.
- `agent.py` - Class of the LangChain agent, integrating both tools.
- `classifier.py` - Class of the text classifier.
- `rag.py` - Implements the RAG system.
- `inference.py` - Main script for running inference of the agent.



## Project Stages 🏗️
### 1. **Data Preparation**📝
   - Collected data for classification. Dataset source: [Kaggle - Text Classification Documentation](https://www.kaggle.com/datasets/tanishqdublish/text-classification-documentation)
   - Conducted EDA using `EDA.ipynb` to understand data distributions and patterns. Checked for NaNs and duplicated values.
   - Translated the dataset to Ukrainian, saved in `ukr_dataset.csv`
   - Created `data_for_rag.txt` as a knowledge base for the RAG system.

### 2. **Model Fine-Tuning**🎯
   - Used roberta-base for fine-tuning on classification tasks. (`fine_tuning.ipynb`, `fine_tuning_ukr.ipynb`)
   - Used LoRA as fine-tuning optimization technique.
   - Сhose evaluation metric: F1 score, as the task was classification.
   - Saved the fine-tuned model in `saved_model_eng/` and `saved_model_ukr/` .

### 4. **Building the LangChain Agent**🤖
  - Built RAG pipeline in `rag.py`
  - Implemented class for text classifier in `classifier.py`
  - Implemented an agent in agent.py that integrates the classifier and RAG as tools

### 5. **Inference and Testing**🖥️
   - Verified performance on sample inputs.
   - UI implemented with Streamlit for interactive use in `inference.py`



