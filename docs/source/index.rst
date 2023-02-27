.. hw6-hmm documentation master file, created by
   sphinx-quickstart on Sat Feb 11 16:27:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lab 6: Inferring CRE Selection Strategies from Chromatin Regulatory State Observations using a Hidden Markov Model and the Viterbi Algorithm
============================================================================================================================================

The aim of hw6 is to implement the Viterbi algorithm, a dynamic program that is a common decoder for Hidden Markov Models (HMMs). The lab is structured by training objective, project deliverables, and experimental deliverables:

**Training Objective**: Learn how to design reusable Python packages with automated code documentation and develop testable (user case) hypotheses using the Viterbi algorithm to decode the best path of hidden states for a sequence of observations.

**Project Deliverable**: Produce a simple report for functional characterization inferred from a binary regulatory observation state pattern across cardiac developmental timepoints.

**Experimental Deliverable**: Construct a positive control library for massively parallel reporter assays (MPRAs) and CRISPRi/a experiments in primitive and progenitor cardiomyocytes (i.e., cardiogenomics).

Key Words
==========
Chromatin; histones; nucleosomes; genomic element; accessible chromatin; chromatin states; genomic annotation; candidate cis-regulatory element (cCRE); Hidden Markov Model (HMM); ENCODE; ChromHMM; cardio-genomics; congenital heart disease(CHD); TBX5


Functional Characterization Report
===================================

Please evaluate the project deliverable and briefly answer the following speculative question, with an eye to the project's limitations as related to the theory, model design, experimental data (i.e., biology and technology). We recommend answers between 2-6 sentences. It is OK if you are not familiar already with this biological user case; you can receive full points for your best-effort answer.

1. Speculate how the progenitor cardiomyocyte Hidden Markov Model and primitive cardiomyocyte regulatory observations and inferred hidden states might change if the model design's sliding window (default set to 60 kilobases) were to increase or decrease?


Increasing/decreasing the size of the sliding window would affect the resolution at which the underlying genomic sequence is analyzed. Since many individual regulatory elements are smaller than 60kb, I can imagine that a smaller window might be able to better infer cCREs, which are more specific the use of ENCODE and ATAC-Seq data. A larger window would recognize longer-range patterns, which may help infer more generalized regulatory regions, or our CREs.  Ultimately, the sliding window size affects the accuracy of predicting the hidden states. 


2. How would you recommend integrating additional genomics data (i.e., histone and transcription factor ChIP-seq data) to update or revise the progenitor cardiomyocyte Hidden Markov Model? In your updated/revised model, how would you define the observation and hidden states, and the prior, transition, and emission probabilities? Using the updated/revised design, what new testable hypotheses would you be able to evaluate and/or disprove?


I would recommend incorporating relevant genomics data to investigate the types of CREs we are interested in. For example, we may be interested in whether or not certain histone marks are indicative of TBX5 enhancers. I may use different histone ChIP-Seqs for marks of active enhancers to define the amount of regulatory activity in the observation states, and the hidden states would be which specific histone marker was queried. The emission probabilities would be based on how likely a histone mark is associated with enhancer activity. 


3. Following functional characterization (i.e., MPRA or CRISPRi/a) of progenitor and primitive cardiomyocytes, consider all possible scenarios for recommending how to update or revise our genomic annotation for *cis*-candidate regulatory elements (cCREs) and candidate regulatory elements (CREs)?

Functional characterization would provide insight into how accurate our predictions were. We may be able to characterize specific observation states such as high, medium, and low activity for regions following a CRISPRi/a screen. Functional characterization could also be used in defining our hidden states. For example, we could try to predict hidden states for cCREs identified from a CRISPRi/a screen, ATAC-Seq, and ENCODE data, vs. CREs identified from only ATAC-Seq data. 


Models Package 
======================
.. toctree::
   :maxdepth: 2
   
   modules
