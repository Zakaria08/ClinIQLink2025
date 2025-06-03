# ClinIQLink-2025 Challenge

A comprehensive, modular system for medical information retrieval and question answering that combines advanced retrieval methods with multiple LLM providers in a clean, extensible architecture.

## Table of Contents

1. [Project Description](#project-description)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Setup Instructions](#setup-instructions)
5. [Quick Start](#quick-start)
6. [Benchmarking & Evaluation](#benchmarking--evaluation)
7. [Results & Performance](#results--performance)
8. [Contributing Log](#contribution-log)
9. [References](#references)

---

## Project Description

This project implements a medical question answering system that combines **Retrieval-Augmented Generation (RAG)** with multiple retrieval algorithms and LLM providers. The system is designed to answer complex medical questions by retrieving relevant information from large medical corpora and generating accurate, contextual responses.

### Key Objectives
- **Multi-Algorithm Retrieval**: Support BM25, SPLADE, UniCOIL, and MedCPT retrieval methods
- **Multi-Provider LLMs**: Integration with OpenAI GPT, Google Gemini/Gemma, and local models
- **Medical Focus**: Optimized for medical literature and clinical question answering
- **Modular Design**: Clean, extensible architecture for research and production use
- **Comprehensive Evaluation**: Robust benchmarking framework for performance analysis

### About the challenge
**ClinIQLink-2025** is a challenge that evaluates how well LLMs can retrieve and generate factually accurate medical information. The challenge focuses on two main tasks:

- **Reducing Hallucinations:** Identify and mitigate hallucinations in the model’s responses, ensuring that the generated answers are grounded in accurate, evidence-based medical knowledge.
- **Improving Accuracy:** Enhance the performance of medical QA systems, so that the answers meet the high standards expected in clinical practice.

Participants are required to develop innovative solutions to tackle these issues and submit their models. The submissions will be rigorously tested on a hidden test set provided by the challenge organizers, and rankings will be based on metrics such as precision, recall, and semantic accuracy.

For more details, please visit the [ClinIQLink-2025 website](https://brandonio-c.github.io/ClinIQLink-2025/).

---

## Features

### 🔍 **Retrieval System** (`medical_retrieval`)
- **Multiple Algorithms**: BM25, SPLADE, UniCOIL, MedCPT
- **Hybrid Methods**: Combine retrievers with Reciprocal Rank Fusion (RRF)
- **Medical Corpora**: PubMed, StatPearls, Medical Textbooks, Wikipedia
- **Scalable Indexing**: Memory-efficient batch processing for large datasets
- **Type Safety**: Full type hints and validation

### 🤖 **RAG System** (`medical_rag`)
- **Multiple LLM Providers**: OpenAI, Google Gemini/Gemma, Local models (llama-cpp)
- **Custom Prompts**: Support for TextGrad and prompt optimization with the method called "rag_answer_textgrad"
- **JSON Processing**: Automatic structured response extraction
- **Error Handling**: Robust error recovery and fallback mechanisms

### 📊 **Evaluation Framework**
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score
- **Question Types**: Multiple choice, True/False, List selection
- **Benchmarking**: Automated evaluation on medical Q&A datasets
- **Reproducible**: Deterministic results with configurable parameters

---

## Project Structure

```
ClinIQLink2025/
├── src/                           # 📦 Source Code Packages
│   ├── medical_retrieval/         # 📚 Document Retrieval Package
│   │   ├── __init__.py           #   Package interface
│   │   ├── config.py             #   Configuration settings
│   │   ├── base_retriever.py     #   Abstract base class
│   │   ├── bm25_retriever.py     #   BM25 implementation
│   │   ├── splade_retriever.py   #   SPLADE implementation
│   │   ├── unicoil_retriever.py  #   UniCOIL implementation
│   │   ├── medcpt_retriever.py   #   MedCPT implementation
│   │   ├── retrieval_system.py   #   Main orchestration
│   │   ├── doc_extracter.py      #   Document caching
│   │   ├── factory.py            #   Factory patterns
│   │   └── utils.py              #   Utility functions
│   │
│   ├── medical_rag/              # 🤖 RAG System Package
│   │   ├── __init__.py           #   Package interface
│   │   ├── rag_config.py         #   RAG configuration
│   │   ├── base_llm.py           #   LLM base class
│   │   ├── openai_llm.py         #   OpenAI implementation
│   │   ├── gemini_llm.py         #   Google Gemini/Gemma
│   │   ├── local_llm.py          #   Local model support
│   │   ├── llm_factory.py        #   LLM factory
│   │   ├── rag_system.py         #   Main RAG orchestration
│   │   ├── rag_factory.py        #   RAG factory patterns
│   │   └── rag_utils.py          #   Utility functions
│   │
│   ├── data/                     # 🗃️ Data Processing Scripts
│   │   ├── pubmed.py             #   PubMed data processing
│   │   ├── statpearls.py         #   StatPearls processing
│   │   ├── textbooks.py          #   Textbook processing
│   │   └── wikipedia.py          #   Wikipedia processing
│   │
│   ├── baseline.py               # 📊 Baseline model implementation
│   ├── Main.py                   # 🚀 Main execution script
│   └── template.py               # 📝 Prompt templates
│
├── corpus/                        # 📚 Medical Corpora Storage
│   ├── pubmed/                   #   PubMed abstracts
│   ├── selfcorpus/               #   Custom corpus
│   │   ├── chunk/                #   Chunked documents
│   │   └── index/                #   Search indices
│   ├── statpearls/               #   StatPearls reference
│   ├── textbooks/                #   Medical textbooks
│   └── wikipedia/                #   Medical Wikipedia
│
├── logs/                         # 📈 Experiment Results
│   └── saved_logs/               #   Benchmark results by method
│       ├── BM25/                 #   BM25 results
│       ├── MedCPT/               #   MedCPT results
│       ├── SPLADE/               #   SPLADE results
│       ├── UniCOIL/              #   UniCOIL results
│       ├── MedicalHybrid/        #   Hybrid method results
│       ├── OptimalHybrid/        #   Optimal combination
│       └── *.json                #   Individual benchmark files
│
├── .env                          # 🔐 Environment variables
├── .env.example                  # 🔐 Environment template
├── .gitignore                    # 🚫 Git ignore rules
├── requirements.txt              # 📦 Python dependencies
├── test_rag.py                   # 🎯 Main evaluation script
├── Benchmark_validation_testset.json      # 📋 30-question benchmark
├── Benchmark_validation_enhanced_testset.json # 📋 150-question benchmark
└── README.md                     # 📋 This documentation
```

---

## Setup Instructions

> [!IMPORTANT]  
> This system requires Python 3.11 and has optional dependencies for different LLM providers and retrieval methods.

### 1. Clone Repository
```bash
git clone https://github.com/Zakaria08/ClinIQLink2025.git
cd ClinIQLink2025
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Unix/MacOS
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirement.txt
```
> [!IMPORTANT]  
> Git-lfs is required to download and load corpora for the first time.
> Java is also requried to download for the indexing


### 4. Configure Environment Variables
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env
```

Required environment variables:
```bash
# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini/Gemma (optional)
GOOGLE_API_KEY=your_google_api_key_here

# HuggingFace for local models (optional)
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

### 5. Verify Installation
```bash
# Move to src folder
cd src/ 

# Test retrieval system
python -c "from medical_retrieval import create_retrieval_system; print('✅ Retrieval system OK')"

# Test RAG system
python -c "from medical_rag import create_medical_rag; print('✅ RAG system OK')"

# Validate environment
python -c "from medical_rag import validate_environment; print(validate_environment())"
```

---

## Quick Start

### Basic Usage

```python
from medical_rag import create_medical_rag

# Create a medical RAG system
rag = create_medical_rag(
    llm_provider="openai",          # or "gemini", "gemma3", "local"
    use_rag=True,                   # Enable retrieval
    retriever_name="MedicalHybrid", # BM25 + MedCPT
    corpus_name="MedCorp"           # All medical corpora
)

# Ask a medical question
question = {
    "question": "What are the first-line treatments for hypertension?",
    "type": "multiple_choice",
    "options": {
        "A": "ACE inhibitors",
        "B": "Thiazide diuretics",
        "C": "Beta blockers",
        "D": "Calcium channel blockers"
    }
}

# Get answer with supporting documents
answer, documents, scores = rag.answer(question)
print(f"Answer: {answer}")
print(f"Supporting documents: {len(documents)}")
```

---

## Benchmarking & Evaluation

### Running Benchmarks

The main evaluation script `Main.py` provides comprehensive benchmarking:

```bash
# Full benchmark (150 questions)
python Main.py --benchmark_file ./Benchmark_validation_enhanced_testset.json

# Quick test (30 questions)
python Main.py --benchmark_file ./Benchmark_validation_testset.json

# Custom configuration
python Main.py --limit 10 --output_dir ./my_results
```

### Benchmark Configuration

Edit the configuration variables in `Main.py`:
```python
retriever = "SPLADE"      # BM25, SPLADE, UniCOIL, MedCPT, MedicalHybrid
corpus = "MedCorp"        # Textbooks, PubMed, StatPearls, MedCorp, All
model = "Gemma3"          # Gemma3, OpenAI, Gemini, Local
documents = 3             # Number of documents to retrieve
```

### Evaluation Metrics

The system computes comprehensive metrics:
- **Accuracy**: Exact match accuracy
- **Precision**: Precision of retrieved information
- **Recall**: Coverage of relevant information
- **F1-Score**: Harmonic mean of precision and recall

Results are saved in structured format:
```
logs/rag_results_SPLADE_150QA_3_MedCorp_Gemma3_20241201_143022/
├── config.json              # Experiment configuration
├── benchmark_scores.json    # Overall metrics
├── raw_results.json        # Detailed results
└── questions/              # Per-question artifacts
    ├── question_1/
    │   ├── question.json
    │   ├── prompt.txt
    │   ├── response.txt
    │   ├── snippets.json
    │   └── parsed_answer.json
    └── ...
```

---

## Results & Performance

### Baseline Performance

| Model | Question | F1-Score |
|-------|----------|----------|
| Gemma 3 12B | 30 | 0.77 |
| Gemma 3 12B | 150 | 0.79 |
| Gemini | 30 | 0.91 |
| Gemini | 150 | 0.84 |
| Gemini | 30 | 0.77 |
| Gemini | 150 | 0.77 |



### Improvement made on Multiple Choice QA

| Retriever       | # Documents | MC vs Baseline | Model   |
| :-------------- | :---------: | :------------: | :------ |
| BM25            |      5      |    +  9.38 %   | Gemma-3 |
| ExpansionHybrid |      3      |    +  3.89 %   | Gemma-3 |
| SPLADE          |      5      |    +  3.89 %   | Gemma-3 |
| ExpansionHybrid |      5      |    +  3.72 %   | OpenAI  |
| BM25            |      15     |    +  3.72 %   | OpenAI  |
| ExpansionHybrid |      15     |    +  1.97 %   | Gemma-3 |
| BM25            |      15     |    +  1.97 %   | Gemma-3 |
| OptimalHybrid   |      15     |    +  1.88 %   | OpenAI  |

### Improvement made on True/False QA
| Retriever       | # Documents | TF vs Baseline | Model  |
| :-------------- | :---------: | :------------: | :----- |
| ExpansionHybrid |      5      |    + 13.39 %   | Gemini |
| MedicalHybrid   |      15     |    +  9.14 %   | Gemini |
| MedicalHybrid   |      5      |    +  9.14 %   | Gemini |
| ExpansionHybrid |      15     |    +  6.94 %   | Gemini |
| ExpansionHybrid |      3      |    +  6.94 %   | Gemini |
| OptimalHybrid   |      5      |    +  6.94 %   | Gemini |

### Improvement made on List QA
- No improvement have been made in my experiment with the list QA.

> [!NOTE]  
> All results are based on medical Q&A benchmarks with 150 questions across multiple medical domains. Performance may vary based on specific use cases and configurations.

---

## Contribution Log

Everything in this repository was created by me, Zakaria Omarar. Kevin Pfister used and later forked both the RAG and the entire project, using the custom entry point I provided — rag_answer_textgrad — to implement the TextGrad component.
---

## References

### Academic Papers
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP.
- Formal et al. (2021). *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*. SIGIR.
- Lin et al. (2021). *Sparse, Dense, and Attentional Representations for Text Retrieval*. TACL.
- Jin et al. (2023). *MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs*. arXiv.
- [A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models](https://arxiv.org/abs/2401.01313) (S. M Towhidul Islam Tonmoy et al., Findings 2024)

### Medical Data Sources
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/) - Biomedical literature database
- [StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK430685/) - Medical reference textbook
- [Medical Wikipedia](https://en.wikipedia.org/wiki/Portal:Medicine) - Medical articles

### Technical Libraries
- [PySerini](https://github.com/castorini/pyserini) - Information retrieval toolkit
- [Transformers](https://huggingface.co/transformers/) - Neural language models
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Local model inference
- [OpenAI API](https://openai.com/api/) - GPT model access
- [Google Generative AI](https://ai.google.dev/) - Gemini model access

### Related Projects And Challenge
- [ClinIQLink-2025 website](https://brandonio-c.github.io/ClinIQLink-2025/).
- [Benchmarking Retrieval-Augmented Generation for Medicine](https://aclanthology.org/2024.findings-acl.372/) (Xiong et al., Findings 2024)
 