# MSc Internship

# Benchmarking Large Language Models: Via Code-Based Particle-Physics Challenges

## Abstract

This thesis investigates how to evaluate scientific understanding in large language models (LLMs) in a way that goes beyond multiple-choice benchmarks and short-form reasoning. I develop a domain-grounded, philosophy-informed benchmark pipeline for particle physics in which LLMs must generate runnable (machine-learning) Python code from a single prompt, train on provided data, and are scored automatically via task-specific scalar metrics under controlled execution constraints.

The framework operationalises scientific understanding as measurable agentic competence: an LLMs
ability to translate an open-ended problem description into correct data handling, modelling choices and training conventions that succeed on contemporary particle physics tasks. I implement two modular challenges. FOURTOPS is a binary event-classification task scored by the area under the ROC curve (AUC). TRACKFORMERS frames hit-to-track association as cluster classification scored by the FitAccuracy, a TrackML-style metric that rewards reconstructed tracks that are sufficiently large, pure, and efficient.

Across three benchmark runs, eight contemporary LLMs are evaluated on both challenges. The results
show variation in robustness and performance: some LLMs frequently fail due to subtle contract violations or fragile preprocessing, while state-of-the-art general-purpose systems produce competitive pipelines. The strongest models approach research-grade reference performance, with OpenAI’s GPT-5.2 Pro achieving the highest mean and single-run AUC and Google’s Gemini 3 Pro Preview achieving the highest mean and single-run FitAccuracy. Overall, the benchmark provides a reproducible measurement instrument for tracking progress in physics-oriented code generation, and supports a graded view of scientific understanding based on ability reflected by the epistemic depth of the task.

## What this repository demonstrates
- A semi-automated pipeline for LLM benchmarking in particle physics ML challenges
- A framework for operationalising the philosophical notion of Scientific Understanding
- Contemporary Foundation Models can create competitive discriminating ML models in single-shots

## Status

Research/coursework code from 2025. Not intended as production software.

