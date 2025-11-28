# ProKDF - Lung Nodule Classification 
A novel probabilistic knowledge-integrated ML framework with clinical interpretability for benign–malignant lung nodule classification. 

**Article: Bayesian Probabilistic Knowledge from Diameter Prior for Decision Fusion to Detect Lung Nodule Heterogeneity (Under Review).**


Recent advances in deep learning (DL) have shown promising results in the identification of malignant lung nodules from computed tomography (CT) scans. However, conventional DL models primarily rely on spatial features and often lack clinical interpretability and the ability to incorporate domain-specific priors. To address these limitations, we integrate Bayesian diameter posterior probabilistic knowledge, allowing the model to leverage the well-established correlation between nodule size and malignancy likelihood. To enhance interpretability, we introduce a new deep-texture-shape (DTS) scoring scheme that offers oncologists a transparent, component-wise justification for malignancy prediction. Furthermore, relying solely on spatial features may overlook critical textural patterns, which play a significant role in malignancy assessment. To overcome this, we utilize local binary patterns (LBP), histogram of oriented gradients (HOG), and gray level co-occurrence matrix (GLCM) to extract rich textural features, capturing the fine-grained patterns that contribute to malignancy characterization. This multi-faceted approach not only improves prediction accuracy but also enhances the model’s robustness. The proposed method, probabilistic knowledge-based decision fusion (ProKDF), is evaluated on two public datasets: the Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI) and the International Society for Optics and Photonics and American Association of Physicists in Medicine (SPIE-AAPM). It achieves 87.48\% $\pm$ 2.09\% F1-score and 92.56\% $\pm$ 1.16\% area under the receiver operating characteristic curve (AUC) on LIDC-IDRI. Our findings suggest that integrating shape-based probabilistic knowledge and texture information enhances model performance and interpretability in detecting nodule heterogeneity.

## Motivation:
<img width="487" height="184" alt="image" src="https://github.com/user-attachments/assets/0f59efe4-4dbb-47ba-b42d-84532cc372d0" />


## Input Datasets:
 - LIDC-IDRI https://www.cancerimagingarchive.net/collection/lidc-idri/
 - SPIE-AAPM https://www.cancerimagingarchive.net/collection/spie-aapm-lung-ct-challenge/
<img width="512" height="281" alt="image" src="https://github.com/user-attachments/assets/d0ad504c-6cf8-4e5b-b8ea-b312ba2033c5" />


## ProKDF Framework: 
Comprehensive workflow of the proposed ProKDF method, including preprocessing, multi-view nodule extraction, and Bayesian posterior knowledgeguided decision fusion for malignancy classification with interpretation.
<img width="959" height="494" alt="image" src="https://github.com/user-attachments/assets/b4952e33-082a-4485-b78f-5c479113af18" />

## Knowledge Base: 
Diameter malignancy likelihood
<img width="456" height="242" alt="image" src="https://github.com/user-attachments/assets/a8c7a546-d439-4444-90b0-436f68df7c70" />

## Textural Insights:
- Local Binary Patterns (LBP)
- Histogram of Oriented Gradients (HOG)
- Gray Level Co-Occurrence Matrix (GLCM)

Few examples of LBP texture are:

   <img width="449" height="219" alt="image" src="https://github.com/user-attachments/assets/f7ea33c0-6c2a-4208-b554-d0716cdbb541" />

The impact of GLCM parameters in textural point of view:

   <img width="524" height="232" alt="image" src="https://github.com/user-attachments/assets/fbd4fef6-18ee-435d-8d00-16d500e018ca" />


## Results:
<img width="482" height="349" alt="image" src="https://github.com/user-attachments/assets/0be0066f-cfff-4ef2-b7c8-82efa65be919" />

<img width="458" height="250" alt="image" src="https://github.com/user-attachments/assets/6934d646-cd03-424c-ad2c-83d1cc31fb57" />

<img width="568" height="257" alt="image" src="https://github.com/user-attachments/assets/18eb9494-0900-4857-9507-2125127ba99c" />

<img width="595" height="231" alt="image" src="https://github.com/user-attachments/assets/31c4e154-4384-463f-948d-6c6d671ca548" />


## Conclusion:
The study presents a novel Bayesian posterior probabilistic
knowledge-based decision fusion (ProKDF) method for effectively classifying benign versus malignant nodules. The probabilistic knowledge is retrieved from the nodule size, which is
a clinically accessible but significant medical prior. The proposed classification model is powered by four optimal models
focusing on the deep and textural contextual information of
lung nodules. We conducted a series of relevant experiments
to determine the optimal model, knowledge reliance factor,
the effect of input data modality, and the number of views.
The competitive performance, along with interpretability and
domain knowledge, makes the model highly reliable in clinical
settings. In the medical domain, where data scarcity is a significant challenge, this novel approach of leveraging domainspecific priors represents a promising pathway.






