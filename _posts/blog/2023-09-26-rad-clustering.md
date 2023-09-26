---
layout: post
tags: [overview]
---

# Introduction
This is a project that look at 20 years of research funded by NCI to the department of radiation oncology and diagnostic radiology. For this project, I use a k-mean clustering approach to "define" a topic. Since this is a mathematical definition of a research topic, we also manually validate the machine-labeled topic of 400 random research abstracts. Details of our method can be found on arXiv[^1].

# Dataset
From NIH RePORTER, we retrieved the 19,945 NIH grants funded from FY 2000 through 2020 awarded to principal investigators (PI) with primary affiliation in departments of radiation oncology or diagnostic radiology. This dataset is further refined using the following pipeline
![fig1](images/rad_clustering_fig1.png)

# Method
The core method of this project is topic clustering and there are multiple initial steps before writing your first line of code:
- What is the clustering method?
- What is the number of topics? Defining k, or the number of topics, remains the most challenging steps since this is very objective and requires expert opinion. Luckily, there are a few mathematically methods to define a topic (i.e. elbow plot).
- How to vectorize a document? When reading a sentence, we can see letters. However, computer technically do not understand letters. You have to convert everything into numbers so how do we convert a document into vectors (or a string of numbers)? Tf-idf is always a good place to start. Transformer is a more robust way but it is very resource-intensive. Here we use weighted tf-idf which is more advanced than the trandition tf-idf and requires less resources than transformer.

# t-SNE visualization
![fig2](images/rad_clustering_fig2.png)

# Findings
Average annual growth and total funding analysis for granularity-level k=15 estimated research topics at FY 2000 and FY 2020. Topics are numbered and color-coded according to their absolute growth rate, with topic 1 and red corresponding to the greatest _increase_ and topic 15/60 and blue corresponding to the greatest _decrease_ in funding. 
![fig3](images/rad_clustering_fig3.png)
![fig4](images/rad_clustering_fig4.png)
![fig5](images/rad_clustering_fig5.png)

We also check the newly emerged and extinct research topics funded by NCI to departments of diagnostic radiology or radiation oncology from 2000-2020. These results do not indicate that there was no research in these topics before or after these dates, but that collective research did not rise to the level to be detected at a k=60 granularity. 
![fig6](images/rad_clustering_fig6.png)

---
{: data-content="footnotes"}

[^1]: [arXiV link](https://arxiv.org/abs/2306.13075)
