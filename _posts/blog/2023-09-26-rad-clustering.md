---
layout: post
tags: [overview]
---

# Project description
This is a project that look at 20 years of research funded by NCI to the department of radiation oncology and diagnostic radiology. For this project, I use a k-mean clustering approach to "define" a topic. Since this is a mathematical definition of a research topic, we also manually validate the machine-labeled topic of 400 random research abstracts. Details of our method can be found on arXiv[^1].

# Dataset
From NIH RePORTER, we retrieved the 19,945 NIH grants funded from FY 2000 through 2020 awarded to principal investigators (PI) with primary affiliation in departments of radiation oncology or diagnostic radiology. This dataset is further refined using the following pipeline
![fig1](images/rad_clustering_fig1.png)

# t-SNE visualization
![fig2](images/rad_clustering_fig2.png)

# Findings
![fig3](images/rad_clustering_fig3.png)
![fig4](images/rad_clustering_fig4.png)
![fig5](images/rad_clustering_fig5.png)
![fig6](images/rad_clustering_fig6.png)

---
{: data-content="footnotes"}

[^1]: [arXiV link](https://arxiv.org/abs/2306.13075)
