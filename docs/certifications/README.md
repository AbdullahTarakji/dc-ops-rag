# Certification mapping

This project doubles as hands-on preparation for a set of AI certifications. The table below
is filled in as iterations land, and it records the **gaps** as carefully as the coverage:
claiming an exam objective is covered when it is not would make the whole table useless.

Syllabi checked on 2026-08-18. Microsoft exams: AI-103 skills measured as of 16 April 2026,
AI-901 as of 15 April 2026, AI-300 as published. All three are Microsoft Foundry–centric.

## Coverage summary

| Certification | Covered here | Must be studied separately |
|---|---|---|
| IBM *Fundamentals of AI Agents Using RAG and LangChain* | RAG pipeline, encoders and tokenizers, prompt templates, LangGraph tools and agents | watsonx / Granite specifics |
| IBM *Build RAG Applications* | Chunking, embeddings, vector databases, retrievers, Gradio front end | LlamaIndex (planned as an optional adapter) |
| IBM *Develop Generative AI Applications* | Prompt engineering, structured output, application integration | IBM Cloud tooling |
| **AI-901** Azure AI Fundamentals | Responsible AI, model selection, deployment parameters, generative + agentic + information-extraction workloads | Vision, speech, Foundry portal walkthroughs |
| **AI-103** Azure AI Apps and Agents Developer | RAG in an application, hybrid and vector search, agent tool schemas, memory, monitoring, evaluators, observability, safety filters, CI/CD | Image and video generation, speech, Content Understanding, Foundry portal specifics |
| **AI-300** Operationalizing ML and GenAI | GenAIOps: evaluation datasets, groundedness metrics, tracing, token and cost accounting, RAG optimisation, prompt versioning, MLflow tracking and model registry | Azure ML workspaces, Bicep/IaC (partly, iteration 8), distributed training, fine-tuning |
| Databricks *AI Agent Fundamentals* | Agent concepts, tool calling, agent evaluation with judges | Mosaic AI and Agent Bricks product surface |
| Databricks *Fundamentals* | — | Lakehouse, Delta, Unity Catalog |

## Objective traceability

Filled in per iteration: exam objective → the file that implements it → the lesson that
explains it.

| Exam | Objective | Where in this repository | Lesson |
|---|---|---|---|
| AI-103 | Choose an appropriate method for retrieval and indexing | `domain/ports.py`, `application/retrieval_service.py` | [01](../learn/01-ports-and-adapters.md) |
| AI-300 | Optimise retrieval by tuning chunk sizes and retrieval strategies | `application/fusion.py`, `configs/` | 03, 05 *(planned)* |

*More rows land with each iteration.*

## Self-quiz

Built in the final iteration from the questions in each lesson's "Interview questions"
section, which are chosen to overlap with what these exams actually test.
