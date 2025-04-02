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
![ClinIQLinkStep1 drawio](https://github.com/user-attachments/assets/63b695a4-39bd-4c56-ac5c-346904ec1a5c)

- **Objective:** Establish a performance baseline using a raw medical LLM without any enhancements.
- **Process:**  
  - Use a baseline model OpenBioLLM to answer medical QA pairs.
  - Evaluate using standard metrics (precision, recall, F1 score) to measure factual accuracy.
- **Outcome:**  
  - A baseline performance score against which improvements can be compared.

---
### Step 2: Retrieval-Augmented Generation with MedRAG
![ClinIQLink drawio](https://github.com/user-attachments/assets/5f26d9ce-7768-4e56-a901-671e8f4bbd9e)# ClinIQLink-2025 Challenge

- **Objective:** Enhance model performance by integrating custom retrieval to support factual accuracy.
- **Integration:**  
  - Use a custom version of [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) to incorporate a retrieval module.
  - Retrieve relevant medical documents from sources such as PubMed, StatPearls, Textbooks, and Wikipedia.
- **Process:**  
  - Enrich the prompt with retrieved context before generating an answer.
  - Evaluate the improved answers on factual correctness.
- **Outcome:**  
-  

---

### Step 3: Prompt Refinement using TextGrad
![ClinIQLinkStep3 drawio](https://github.com/user-attachments/assets/926ef605-698f-47ae-a035-f29a8ea0fc95)
- **Objective:** Further optimize the response by refining prompts and outputs.
- **Integration:**  
  - Use [TextGrad](https://github.com/zou-group/textgrad) to iteratively refine and optimize the LLM’s outputs.
  - Apply textual gradients (similar to backpropagation) to adjust the prompt and generated answer.
- **Process:**  
  - Define a loss function that captures the accuracy and semantic similarity requirements.
  - Iteratively update the prompt and answer based on model feedback with the usage a more smarter model like DeepSeek R1.
- **Outcome:**  
-

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


