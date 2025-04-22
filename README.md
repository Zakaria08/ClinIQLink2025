# ClinIQLink-2025 Challenge

**ClinIQLink-2025** is a challenge designed to push the limits of large language models (LLMs) in the medical domain. The objective is to develop methods that significantly reduce hallucinations and improve the accuracy of medical question answering (QA). Participants must submit their solutions to be tested on a hidden test set, and the leaderboard will rank submissions based solely on their factual accuracy in retrieving medical knowledge.

---

## Table of Contents

1. [Overview](#overview)  
2. [Project Requirements](#project-requirements)  
3. [Architecture and Workflow](#final-architecture-and-workflow)  
4. [Challenge Development Stages](#challenge-development-stages)  
    - [Step 1: Baseline Benchmark](#step-1-baseline-benchmark)  
    - [Step 2: Retrieval-Augmented Generation with MedRAG](#step-2-retrieval-augmented-generation-with-medrag)  
    - [Step 3: Prompt Refinement using TextGrad](#step-3-prompt-refinement-using-textgrad)  
5. [Submission and Evaluation](#submission-and-evaluation)  
6. [Installation and Setup](#installation-and-setup)  
7. [Usage](#usage)  
8. [Future Directions](#future-directions)  

---

## Overview

**ClinIQLink-2025** is a challenge that evaluates how well LLMs can retrieve and generate factually accurate medical information. The challenge focuses on two main tasks:

- **Reducing Hallucinations:** Identify and mitigate hallucinations in the model’s responses, ensuring that the generated answers are grounded in accurate, evidence-based medical knowledge.
- **Improving Accuracy:** Enhance the performance of medical QA systems, so that the answers meet the high standards expected in clinical practice.

Participants are required to develop innovative solutions to tackle these issues and submit their models. The submissions will be rigorously tested on a hidden test set provided by the challenge organizers, and rankings will be based on metrics such as precision, recall, and semantic accuracy.

For more details, please visit the [ClinIQLink-2025 website](https://brandonio-c.github.io/ClinIQLink-2025/).

---

## Project Requirements

- **Programming Language:** Python 3.11  
- **Deep Learning Framework:** PyTorch  
- **LLM Libraries:** 
  - Open-source biomedical LLMs 
  - Transformers library
- **Retrieval Toolkit:** [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) – for implementing Retrieval-Augmented Generation (RAG) in medical QA  
- **Optimization Toolkit:** [TextGrad](https://github.com/zou-group/textgrad) – for iterative prompt and output refinement using backpropagated textual feedback  
- **Additional Tools:**  
  - Git-LFS (for downloading large corpora)  
  - Java (for BM25 support)  

---

## Final Architecture and Workflow

The workflow diagram below outlines the entire process employed in the challenge:



---

## Challenge Development Stages

### Step 1: Baseline Benchmark
- **Objective:** Establish a performance baseline using a raw medical LLM without any enhancements.
- **Process:**  
  - Use a baseline model OpenBioLLM to answer medical QA pairs.
  - Evaluate using standard metrics (precision, recall, F1 score) to measure factual accuracy.
- **Potential Challenges:**
  - Low Baseline Performance: The raw model may struggle with consistency and accurate fact retrieval.
  - Hallucinations: High incidence of unsupported claims due to lack of external evidence.
  - Evaluation Metrics: We may face challenges in accurately capturing model outputs when they are presented in an unstructured format.
- **Outcome:**  
  - A baseline performance score against which improvements can be compared.
  - True/False:
      - Accuracy: 0.6
      - Precision: 1
      - Recall: 0.6
      - F1 Score: 0.75
  - Multiple Choice:
      - Accuracy: 0.5
      - Precision: 0.5
      - Recall: 1
      - F1 Score: 0.67
  - Multiple Choice:
      - Accuracy: 0.69
      - Precision: 0.92
      - Recall: 0.73
      - F1 Score: 0.81    
- **References:**
  - [ClinIQLink 2025 - LLM Lie Detector Test](https://brandonio-c.github.io/ClinIQLink-2025/)    
  - [HuggingFace - The Open Medical-LLM Leaderboard: Benchmarking Large Language Models in Healthcare](https://huggingface.co/blog/leaderboard-medicalllm)
  - [Do Large Language Models have Shared Weaknesses in Medical Question Answering?](https://arxiv.org/abs/2310.07225) (Andrew M. Bean et al., Findings 2024)
    
![ClinIQLinkBaseline1 drawio](https://github.com/user-attachments/assets/98d9bfe8-8bdb-4947-ad37-498256186cd5)
---
### Step 2: Retrieval-Augmented Generation with MedRAG
- **Objective:** Enhance model performance by integrating custom retrieval to support factual accuracy.
- **Integration:**  
  - Use a custom version of [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) to incorporate a retrieval module.
  - Retrieve relevant medical documents from sources such as PubMed, StatPearls, Textbooks, and Wikipedia.
- **Process:**  
  - Enrich the prompt with retrieved context before generating an answer.
  - Evaluate the improved answers on factual correctness.
- **Potential Challenges:**
  - Retrieval Noise: Irrelevant or low-quality documents might be retrieved, leading to misinformation.
  - Latency and Scalability: Building and querying indices (e.g., BM25, dense retrievers) can be time-consuming.
  - Context window: The context window might be too big to be processed and a need of classification of the most accurate/useful document need to be mad
- **Outcome:**
     
- **References**
  - [Benchmarking Retrieval-Augmented Generation for Medicine](https://aclanthology.org/2024.findings-acl.372/) (Xiong et al., Findings 2024)
  - [R^2AG: Incorporating Retrieval Information into Retrieval Augmented Generation](https://arxiv.org/abs/2406.13249) (Fuda Ye et al., Findings 2024)
  - [A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models](https://arxiv.org/abs/2401.01313) (S. M Towhidul Islam Tonmoy et al., Findings 2024) 

![ClinIQLink drawio](https://github.com/user-attachments/assets/5f26d9ce-7768-4e56-a901-671e8f4bbd9e)

---

### Step 3: Prompt Refinement using TextGrad
- **Objective:** Further optimize the response by refining prompts and outputs.
- **Integration:**  
  - Use [TextGrad](https://github.com/zou-group/textgrad) to iteratively refine and optimize the LLM’s outputs.
  - Apply textual gradients (similar to backpropagation) to adjust the prompt and generated answer.
- **Process:**  
  - Define a loss function that captures the accuracy and semantic similarity requirements.
  - Iteratively update the prompt and answer based on model feedback with the usage a more smarter model like DeepSeek R1.
-  **Potential Challenges:**
      - Optimization Instability: Text-based gradients might converge to local minima or produce inconsistent refinements.
      - Alignment of Loss Functions: Ensuring the textual loss correlates with true factual accuracy is non-trivial.
      - Golden prompt: Finding a single golden prompt for each type of question might not be possible.
- **Outcome:**  
- **References:**
  - [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/pdf/2406.07496) (Mert Yuksekgonul et al., Findings 2024)
  - [A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models](https://arxiv.org/abs/2401.01313) (S. M Towhidul Islam Tonmoy et al., Findings 2024) 

![ClinIQLinkStep3 drawio](https://github.com/user-attachments/assets/926ef605-698f-47ae-a035-f29a8ea0fc95)
---

## Submission and Evaluation

Participants are required to:
- **Submit a Short Paper:** Explain the novel method or approach used to reduce hallucinations and improve accuracy.
- **Upload the Model:** Submit your solution via CodaBench for evaluation.
- **Evaluation Process:**  
  - Solutions will be tested on a hidden test set provided by the challenge organizers.
  - The evaluation will focus solely on the factual accuracy of the retrieved medical knowledge.
  - A leaderboard will rank submissions based on metrics such as precision, recall, F1 score, and semantic accuracy.
  
For detailed challenge guidelines, refer to the [ClinIQLink-2025 website](https://brandonio-c.github.io/ClinIQLink-2025/).

---

## Installation and Setup



---

## Usage



---

## Future Directions


