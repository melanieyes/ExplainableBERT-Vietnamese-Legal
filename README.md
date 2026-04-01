# Vietnamese Legal Relation Classification with Explainability and Hypothesis Testing

## Overview

This project builds an AI pipeline for **Vietnamese legal text classification** using **PhoBERT** and evaluates whether the model remains stable when legal wording is changed. The study is based on the **VNLegalText** dataset and focuses on classifying legal reference relations in Vietnamese legal documents.

The project does not only measure predictive performance. It also includes **explainability analysis** and **hypothesis testing** to examine whether the model behaves robustly under textual perturbations and formatting-based counterfactual changes.

---

## Research Goal

The main goal of this project is to answer the following question:

**Can a Vietnamese language model accurately classify legal reference relations, and do its predictions remain stable when legal wording is modified without substantially changing meaning?**

To address this, the project combines:

- supervised classification
- explainability methods
- perturbation analysis
- counterfactual testing
- binomial hypothesis testing

---

## Dataset

The project uses the **VNLegalText** dataset, which contains Vietnamese legal documents annotated with legal references and relation labels.

### Main processing steps
- download and extract legal documents
- parse tagged legal references from the raw files
- reconstruct sentence-level context
- identify the target reference and its relation label
- reduce sparse labels into a smaller and more stable label set
- split the dataset into train, validation, and test sets using stratified sampling

---

## Model

The classification model is based on **PhoBERT**, a pretrained Vietnamese language model.

### Model setup
- backbone: `vinai/phobert-base`
- task: sequence classification
- tokenizer: PhoBERT tokenizer
- framework: Hugging Face Transformers + PyTorch

### Input format
Each example is converted into a text input that preserves:
- the legal sentence context
- the marked target reference
- surrounding references when available

---

## Methodology

### 1. Supervised Classification
The model is fine-tuned on labeled legal relation data.

### 2. Standard Evaluation
Performance is evaluated using:
- accuracy
- macro F1
- weighted F1
- classification report by class

### 3. Explainability
The project includes:
- **saliency analysis** to identify influential tokens
- **perturbation analysis** to test wording sensitivity
- **counterfactual formatting analysis** to test robustness to stylistic or structural changes

### 4. Hypothesis Testing
A **one-sided binomial test** is used to evaluate whether prediction stability is significantly better than chance.

#### Hypotheses
- **H0:** Model stability under perturbation is not better than chance
- **H1:** Model stability under perturbation is better than chance

---

## Results

### Classification Performance
The model achieved strong performance on the test set:

- **Test Accuracy:** 0.9729
- **Test Macro F1:** 0.9497

This suggests that PhoBERT performs very well on Vietnamese legal relation classification.

### Class-level Performance
The strongest-performing classes include:
- **CC**
- **DaC**
- **DSD**

The relatively weaker classes include:
- **HHL**
- **BTT**

This indicates that some legal relation types are easier for the model to distinguish than others.

### Hypothesis Testing Result
For the perturbation robustness test:

- **Successes:** 79
- **Total examples:** 96
- **Observed stability rate:** 0.8229
- **Chance baseline:** 0.50
- **p-value:** 4.97e-11

### Interpretation
The model remained stable in **82.3%** of perturbed cases, which is substantially above the 50% chance baseline. The extremely small p-value indicates that this result is statistically significant. Therefore, the null hypothesis is rejected, and the result supports the claim that the model’s predictions are significantly more stable than chance under wording perturbations.

---

## Tools and Libraries

This project uses the following tools:

- Python
- Jupyter Notebook
- PyTorch
- Hugging Face Transformers
- scikit-learn
- SciPy
- pandas
- NumPy
- matplotlib

---

## Project Pipeline

1. Load and parse VNLegalText  
2. Build structured classification examples  
3. Clean labels and create reduced label set  
4. Split data into training, validation, and test sets  
5. Tokenize text with PhoBERT tokenizer  
6. Fine-tune PhoBERT for sequence classification  
7. Evaluate classification performance  
8. Visualize confusion matrix and per-class metrics  
9. Run saliency analysis  
10. Create perturbation and counterfactual examples  
11. Test prediction stability  
12. Conduct binomial hypothesis testing  

---

## Significance

This project contributes to the growing area of **Vietnamese legal AI** by showing how a Vietnamese language model can be used not only for accurate legal text classification, but also for interpretable and statistically grounded evaluation.

It is relevant to:
- Vietnamese legal text mining
- legal document analysis
- explainable NLP
- low-resource legal AI
- digital humanities in Vietnamese Studies

---

## Future Work

Possible next steps include:

- testing more Vietnamese legal language models
- expanding to legal question answering or retrieval
- using larger and more diverse legal corpora
- improving counterfactual generation
- comparing multiple explainability techniques
- studying robustness across more complex legal reasoning tasks

---

## References

- VNLegalText dataset: https://github.com/mlalab/VNLegalText
- PhoBERT paper: https://aclanthology.org/2020.findings-emnlp.92/
- PhoBERT model: https://huggingface.co/vinai/phobert-base
- SciPy binomtest documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
