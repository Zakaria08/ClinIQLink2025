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
  - List:
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

![ClinIQLink drawio](https://github.com/user-attachments/assets/fbc25f70-8353-42d6-812d-4788034f9650)

---

### Step 3: Prompt Refinement using TextGrad
- **Objective:** Further optimize the response by refining prompts and outputs.
- **Overall Architecture**
  ![RAG_TEXTGRAD drawio](https://github.com/user-attachments/assets/c4244776-f708-4b67-a4fa-bb96ee15a1da)

- **Architecture TextGrad**
 ![fertiges_diagramm drawio](https://github.com/user-attachments/assets/0313d5f8-f45c-4ffc-9b3f-08f69afefcfb)

  **Process Description TextGrad Loop**
- 1. Giving Question to RAG. Passing Context to the answering LLM. Getting a first answer from LLM (test_rag_textgrad.py) ![grafik](https://github.com/user-attachments/assets/cdfe07c1-3f08-40d1-bdc3-1e47a00e4538)

  3. Passing the first Answer into the TEXTGRAD.py file entering the TextGrad loop![grafik](https://github.com/user-attachments/assets/612fc64f-f4de-45e6-afa5-aa4773e4c909)

  4. Checking the Answer in the classification Function ![grafik](https://github.com/user-attachments/assets/0df3b6ab-5cbf-4ffc-9f4d-3f11c7ecb27a)
  5. If the Answer is good return answer, else go into prompt refinement loop ![grafik](https://github.com/user-attachments/assets/653b7cab-985f-499e-b273-3926bcc0efdb)
  6. If answer is bad get Feedback (Loss function) for our answer ![grafik](https://github.com/user-attachments/assets/9fd50040-33d8-46f1-a441-bdffbf6200f1)![grafik](https://github.com/user-attachments/assets/482a6907-f02e-4b9d-b7bf-abaa0949e46c)
  7. Optimizing Prompt based on Feedback (only Question(head) being changed)![grafik](https://github.com/user-attachments/assets/9efcd9c7-ceb4-4e08-8d40-dcee896df78f)
     ![grafik](https://github.com/user-attachments/assets/d42f1094-2f7c-4462-81d0-11fbdb41fe3e)
  9. After prompt refinement rebuilding prompt and giving it to RAG or LLM
![grafik](https://github.com/user-attachments/assets/a1ffc3e9-59da-49db-9a15-d54e49618930)
![grafik](https://github.com/user-attachments/assets/998e3707-a650-4499-ae61-4352684abf52)
  10. If no answer can be found in all iterations return last answer.
      ![grafik](https://github.com/user-attachments/assets/ddd9fd37-f57d-4c2a-966d-4bfe8427041a)


- **Integration:**  
  - Use Chatgpt 4 to refine the prompt
  - Use Chatgpt 3.5 to answer questions
  - Use TextGrad Library to improve Prompts
  

-  **Challenges:**
      - Library works only with predefined LLM's which have an API
      - Uses many Ressources because it takes several requests to improve the prompt and to answer the question
      - Prompts were refined for LLM's and not RAG systems.
- **Outcome:**

  
**Baseline**
- True/False:
  - Accuracy: 0.64  
  - Precision: 1  
  - Recall: 0.64  
  - F1 Score: 0.78  
- Multiple Choice:
  - Accuracy: 0.54  
  - Precision: 0.54  
  - Recall: 1  
  - F1 Score: 0.70  
- List:
  - Accuracy: 0.62  
  - Precision: 0.86  
  - Recall: 0.69  
  - F1 Score: 0.77  

**TextGrad without RAG**
- True/False:
  - Accuracy: 0.60  
  - Precision: 1  
  - Recall: 0.60  
  - F1 Score: 0.75  
- Multiple Choice:
  - Accuracy: 0.58  
  - Precision: 0.58  
  - Recall: 1  
  - F1 Score: 0.73  
- List:
  - Accuracy: 0.68  
  - Precision: 0.87  
  - Recall: 0.75  
  - F1 Score: 0.81  

**TextGrad with RAG**
- True/False:
  - Accuracy: 0.44  
  - Precision: 1  
  - Recall: 0.44  
  - F1 Score: 0.61  
- Multiple Choice:
  - Accuracy: 0.40  
  - Precision: 0.40  
  - Recall: 1  
  - F1 Score: 0.57  
- List:
  - Accuracy: 0.54  
  - Precision: 0.84  
  - Recall: 0.61  
  - F1 Score: 0.70  


![Baseline_vs_Textgrad](https://github.com/user-attachments/assets/ee19d075-d2fb-461b-83eb-0c8f7775449a)


![Textgrad_with_rag](https://github.com/user-attachments/assets/c180ea47-76cb-493d-89de-5f932a10c53c)

- **Understanding of Log Data**
  - Every question has its own question folder. Going into the question folder you can see the answers the LLMs gave us

    **failed_attempt_question_XY.txt**
    - Initial Prompt: Question from the Testset
    - Full Prompt: Prompt that was given the LLM to answer the model. Is the same as initial prompt in the first iteration otherwise refined prompt
    - Answer: Given Answer from the LLM
    - Feedback: Feedback from the evaluation function. What can be improved in this prompt?
    - Improved full prompt: Refined prompt to give to the LLM

    **Parsed_answer.json**
    Answer that is being used for the final evaluation of the model

    **question.json**
    Initial question with its options and ground Truth

    **snippets_Number.json**
    Parts of the RAG that had been retrieved to answer the question.
    Number indicates in which iteration the Documents were retrieved. Since TextGrad Loop has several Iterations

- **Possible Future Outlook**
  - Optimize System Prompt with training Set
  - Optimize Prompt for RAG System instead of LLM to get better retrieval
    
    

- **References:**
  - [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/pdf/2406.07496) (Mert Yuksekgonul et al., Findings 2024)
  - [ClinIQLink-2025 website](https://brandonio-c.github.io/ClinIQLink-2025/).
  - [A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models](https://arxiv.org/abs/2401.01313) (S. M Towhidul Islam Tonmoy et al., Findings 2024) 

![ClinIQLinkStep3 drawio](https://github.com/user-attachments/assets/8da3a491-05d6-45b2-be49-04868d7ee557)

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


