---
layout: post
tags: [soopr, config]
---

# Introduction
Incident reports are an important tool for safety and quality improvement in Radiation Oncology, but manual review is time-consuming and requires subject matter expertise. Here is how we simplify the process using NLP!

# Method
We implemented two datasets to train our models: 7,094 reports from our institution (UW), and 571 from IAEA SAFRON, all of which had severity scores labeled by clinical content experts. We then trained 2 machine-learning models (SVM and BlueBERT) using one dataset and then evaluated using the other dataset.

![fig1](images/ILS-dataset.png)

# Results
![fig2](images/ILS-results.png)
