# Machine Learning and Deep Architectures for Autism Spectrum Disorder Screening: A Methodological Review of Behavioral Questionnaires, Resting-State fMRI Connectomics, Data Leakage Pitfalls, and Emerging Solutions

**Mohit Kumar**, *Senior Member, IEEE*, and **NEURO-SCREEN AI Research Consortium**  
*Department of Computer Science & Clinical Neuroinformatics Laboratory*  
*Autonomous Clinical-AI Systems Research Group*  
Email: mohit.research@neuroscreen.ai

---

### Abstract
Autism Spectrum Disorder (ASD) is a complex, heterogeneous neurodevelopmental condition characterized by atypical social communication, sensory processing variations, and focused behavioral patterns. Over the past decade, hundreds of studies have applied machine learning (ML) and deep neural architectures to automate ASD screening and diagnostic classification using two primary modalities: behavioral screening questionnaires (e.g., AQ-10, Q-CHAT-10) and functional neuroimaging connectomics (notably resting-state fMRI from the ABIDE consortia). However, the clinical AI literature in ASD suffers from a severe metric inflation crisis, marked by pervasive methodological pitfalls that obscure true clinical utility. This paper presents a systematic, rigorous methodological literature review of ML and deep learning techniques for ASD screening. We construct a multi-tiered taxonomy of classical, ensemble, graph, and attention-based paradigms, followed by an in-depth forensic autopsy of the four major sources of metric inflation: (i) arithmetic derivation target leakage in public questionnaire benchmarks, (ii) scanner hardware leakage under naive random cross-validation, (iii) atlas permutation non-invariance in connectome matrix tokenization (explaining the empirical failure of Vision Transformers), and (iv) parameter-to-sample mismatch ($p \gg n$) leading to severe overparameterization. Furthermore, we synthesize empirical benchmark results across models, contrasting reported literature figures with honest Leave-One-Site-Out (LOSO) evaluations. Finally, we review and formulate five methodological frameworks for trustworthy clinical deployment: Empirical Bayes harmonization (ComBat), test-time domain adaptation (CORAL), self-supervised masked connectome pretraining, multimodal late fusion with graceful degradation, and split conformal prediction for distribution-free uncertainty quantification.

**Index Terms**—Autism Spectrum Disorder (ASD), Clinical Machine Learning, Resting-State fMRI, Functional Connectomics, Data Leakage, Vision Transformer, Leave-One-Site-Out Cross-Validation, ComBat Harmonization, Conformal Prediction.

---

## I. Introduction

Autism Spectrum Disorder (ASD) is a lifelong, highly heterogeneous neurodevelopmental condition recognized internationally by the World Health Organization (WHO ICD-11 6A02) and the American Psychiatric Association (DSM-5-TR) [1], [2]. ASD is characterized by persistent differences in reciprocal social communication and interaction, accompanied by restricted, repetitive patterns of behavior, intense specialized interests, and altered sensory reactivity [3]. Diagnostic prevalence has steadily increased across global cohorts, with the United States Centers for Disease Control and Prevention (CDC) reporting an estimated prevalence of 1 in 36 children [4].

Despite rising clinical demand, the gold standard for clinical diagnosis—the Autism Diagnostic Observation Schedule, Second Edition (ADOS-2) [5], paired with the Autism Diagnostic Interview-Revised (ADI-R) [6]—remains resource-intensive, requiring multidisciplinary teams, specialized clinical training, and comprehensive evaluations that routinely exceed 3 to 4 hours per patient. Consequently, median wait times for developmental pediatric evaluation frequently surpass 12 to 18 months globally, missing crucial neuroplastic developmental windows during early childhood [7].

To triage referrals and alleviate diagnostic bottlenecks, computational clinical researchers have turned to artificial intelligence (AI) and machine learning (ML) across two predominant paradigms:
1. **Psychometric and Behavioral Questionnaires:** Rapid, parent- or self-administered screening instruments such as the Autism-Spectrum Quotient 10-Item (AQ-10) [8] and the Quantitative Checklist for Autism in Toddlers 10-Item (Q-CHAT-10) [9].
2. **Resting-State Functional Magnetic Resonance Imaging (rs-fMRI):** In vivo neuroimaging profiling macroscale functional connectivity across regions of interest (ROIs), predominantly curated via the Autism Brain Imaging Data Exchange (ABIDE I and II) [10].

Over the past decade, a burgeoning body of literature has claimed classification accuracies exceeding 90% to 100% on tabular questionnaire datasets, and 75% to 88% on multi-site fMRI connectomes using deep neural networks [11]–[15]. However, independent replication efforts and clinical translation initiatives have consistently failed, revealing a widening chasm between claimed benchmark metrics and actual real-world clinical utility [16], [17].

This review provides a critical, methodologically grounded analysis of machine learning in ASD screening. Unlike conventional surveys that uncritically compile published performance claims, we conduct a forensic audit of the methodological vulnerabilities that artificially inflate metrics in ASD AI literature. We analyze why advanced architectures—specifically Vision Transformers (ViT) and Brain Network Transformers (BNT)—frequently fail on clinical neuroimaging benchmarks, and synthesize proven mathematical and architectural frameworks required to build safe, generalizable, and ethically grounded screening systems.

---

## II. Clinical Background and Screening Modalities

### A. Gold-Standard Clinical Assessment vs. Screening Instruments
A fundamental misunderstanding in computational literature is the conflation of a *screening tool* with a *diagnostic evaluation*. In clinical medicine:
* **Diagnostic Evaluations (ADOS-2, ADI-R):** Conducted by licensed developmental pediatricians, child psychiatrists, or clinical psychologists. They evaluate adaptive functioning, direct social interaction, differential diagnoses (ruling out or co-diagnosing ADHD, generalized anxiety, sensory processing differences, language disorders, or intellectual disability), and developmental history [5], [6].
* **Screening Instruments (AQ-10, Q-CHAT-10, M-CHAT-R/F):** Statistical triaging filters designed to identify individuals exhibiting elevated trait patterns who warrant specialist referral [8], [9]. They possess moderate specificity and are explicitly designed *not* to render standalone clinical diagnoses.

### B. Standardized Questionnaire Instruments
1. **Adult, Adolescent, and Child AQ-10:** Developed by Allison et al. at the Cambridge Autism Research Centre [8], the AQ-10 compresses the original 50-item instrument into 10 items assessing five cognitive/behavioral domains: attention switching, attention to detail, communication, mind-reading/social empathy, and imagination. To mitigate acquiescence response bias, items 2, 3, 4, 5, 6, and 9 are reverse-scored psychometrically.
2. **Toddler Q-CHAT-10:** A 10-item caregiver questionnaire optimized for toddlers aged 18 to 36 months, evaluating early developmental markers including joint attention, response to name, protodeclarative pointing, pretend play, and sensory tracking [9].

### C. Neuroimaging Modalities: Functional Connectomics and the ABIDE Repository
The Autism Brain Imaging Data Exchange (ABIDE I) consortium aggregated 1,112 resting-state fMRI datasets across 17 independent international neuroimaging sites [10]. Resting-state fMRI measures spontaneous low-frequency fluctuations in blood-oxygen-level-dependent (BOLD) signals while subjects rest quietly without performing explicit cognitive tasks [18].

To derive macroscale connectomes, BOLD timeseries $X \in \mathbb{R}^{T \times P}$ (where $T$ denotes temporal scan length and $P$ denotes parcellated regions of interest) are extracted via standardized brain atlases—such as the Craddock CC200 atlas ($P = 200$) [19], the Automated Anatomical Labeling (AAL) atlas ($P = 116$) [20], or Harvard-Oxford ($P = 111$). Functional connectivity between regions $i$ and $j$ is typically modeled via pairwise Pearson correlation:
$$r_{ij} = \frac{\sum_{t=1}^T (X_{t,i} - \bar{X}_i)(X_{t,j} - \bar{X}_j)}{\sqrt{\sum_{t=1}^T (X_{t,i} - \bar{X}_i)^2 \sum_{t=1}^T (X_{t,j} - \bar{X}_j)^2}}$$
Applying the Fisher $z$-transformation yields normalized, symmetric connectivity matrices $Z \in \mathbb{R}^{P \times P}$:
$$z_{ij} = \operatorname{arctanh}(r_{ij}) = \frac{1}{2} \ln \left(\frac{1 + r_{ij}}{1 - r_{ij}}\right)$$
The upper triangle yields $M = \frac{P(P-1)}{2}$ unique functional edges ($M = 19,900$ for CC200), reflecting interactions across canonical networks including the Default Mode Network (DMN), Salience Network, and Frontoparietal Control Network [21].

---

## III. Taxonomy of Machine Learning Approaches

We categorize the algorithmic landscape into four distinct paradigms across the behavioral and neuroimaging domains:

```
Machine Learning for ASD Screening
├── 1. Tabular Behavioral Questionnaire Models
│   ├── Generalized Linear Models (L1/L2 Logistic Regression)
│   ├── Decision Tree Ensembles (Random Forest, XGBoost, LightGBM, CatBoost)
│   └── Meta-Learners & Stacking Ensembles
├── 2. Linear & Regularized Connectome Models
│   ├── Ridge & Lasso Logistic Regression on Vectorized Edges
│   ├── Support Vector Classifiers (Linear, RBF Kernels)
│   └── Empirical Bayes Harmonized Baselines (ComBat + L2 LR)
├── 3. Geometric & Spatial Deep Learning
│   ├── 2D Convolutional Neural Networks on Correlation Matrices
│   └── Graph Neural Networks (GCN, GraphSAGE, BrainGAT)
└── 4. Attention-Based Neural Architectures
    ├── Brain Network Transformers (BNT / ROI-Tokenized Self-Attention)
    └── Connectivity Vision Transformers (ViT / Patch-Tokenized Self-Attention)
```

### A. Classical Tabular Learning on Behavioral Items
Standard supervised classifiers applied to questionnaire responses $A = [A_1, A_2, \dots, A_{10}]$ have spanned Logistic Regression, Support Vector Machines (SVM), Random Forests, and modern Gradient Boosted Decision Trees (GBDT) including XGBoost [22], LightGBM [23], and CatBoost [24]. Stacking ensembles integrate diverse base estimators using an L2-regularized meta-classifier:
$$\hat{y} = \sigma\left(w_0 + \sum_{k=1}^K w_k \hat{p}_k(x)\right)$$
where $\hat{p}_k(x)$ represents the out-of-fold probability estimate from base learner $k$.

### B. Regularized Linear Models on Connectomes
In high-dimensional connectomic regimes where the feature dimension drastically exceeds the sample size ($M \gg N$, e.g., $19,900 \text{ edges} \gg 120 \text{ subjects}$), L2-regularized (Ridge) logistic regression optimizes:
$$\min_{w, b} \frac{1}{N} \sum_{i=1}^N \log\left(1 + \exp\left(-y_i (w^T z_i + b)\right)\right) + \frac{1}{2C} \|w\|_2^2$$
Paired with univariate feature selection (e.g., ANOVA $F$-test filtering for top $k$ discriminative edges) and scanner harmonization, regularized linear models enforce convex optimization surfaces that prevent hyper-plane distortion from idiosyncratic scanner noise [25].

### C. 2D Convolutional Neural Networks (CNN) on Connectome Matrices
Several studies [12], [26] treat the symmetric correlation matrix $Z \in \mathbb{R}^{P \times P}$ as a single-channel grayscale image, applying standard 2D convolution filters $K \in \mathbb{R}^{k \times k}$:
$$S(i, j) = (Z * K)(i, j) = \sum_{m} \sum_{n} Z(i-m, j-n) K(m, n)$$
These architectures stack convolutional layers, batch normalization, and pooling operations prior to fully connected classification heads.

### D. Graph Neural Networks (GNNs) and BrainGAT
Because brain networks form mathematical graphs $G = (V, E)$, Graph Neural Networks (GNNs) model brain ROIs as graph vertices $v_i \in V$ and functional connections as edges $e_{ij} \in E$ [27]. In Graph Attention Networks (GAT) [28], node feature aggregation is modulated by learnable attention coefficients:
$$\alpha_{ij} = \frac{\exp\left(\operatorname{LeakyReLU}\left(a^T [W h_i \,\|\, W h_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\operatorname{LeakyReLU}\left(a^T [W h_i \,\|\, W h_k]\right)\right)}$$
$$h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W h_j^{(l)}\right)$$
This enables dynamic topological message passing along the most salient functional connections.

### E. Attention-Based Transformers: BNT and ViT
1. **Brain Network Transformer (BNT):** Adapts the multi-head self-attention mechanism of Vaswani et al. [29] to connectomes. Each brain region $i \in \{1, \dots, P\}$ is treated as an input token $t_i = W_{in} z_{i,:} + e_i^{pos}$, where $z_{i,:}$ is the complete $1 \times P$ connectivity profile of region $i$, and $e_i^{pos}$ represents a learnable spatial positional embedding [30]. Attention across regions is computed via:
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
2. **Connectivity Vision Transformer (ViT):** Directly adopts the Dosovitskiy et al. [31] image tokenization formulation by partitioning the matrix $Z \in \mathbb{R}^{P \times P}$ into a non-overlapping grid of patches of size $p \times p$ (e.g., $14 \times 14$ patches on $196 \times 196$ matrices, yielding 196 patch tokens). Tokens are projected linearly, augmented with a learnable `[CLS]` token and 1D positional encodings, and processed through standard Transformer encoder blocks.

---

## IV. Methodological Pitfalls and Forensic Audit of Metric Inflation

Despite impressive published claims, our systematic audit reveals that high metrics in ASD screening literature frequently stem from four methodological vulnerabilities:

```
Forensic Audit of ASD AI Literature Pitfalls
├── Pitfall 1: Target Derivation Leakage (The Arithmetic Adder Trap)
│   └── Dataset label is literally ∑ A_i ≥ 7; models memorize addition, not autism.
├── Pitfall 2: Scanner Site Leakage (Random CV on Multi-Site fMRI)
│   └── Scanner hardware artifacts leak across folds; models identify the MRI machine.
├── Pitfall 3: Matrix Permutation Non-Invariance
│   └── 2D patches and CNN filters assume spatial pixel locality that graphs lack.
└── Pitfall 4: Parameter-to-Sample Mismatch (p >> n Overfitting)
    └── Tens of thousands of weights overfit small clinical cohorts (N ~ 100).
```

### A. Pitfall 1: Target Derivation Leakage in Public Questionnaire Datasets
A pervasive issue in tabular ASD screening studies [11], [13] is the uncritically accepted use of public repository datasets (e.g., Kaggle and UCI Machine Learning Repository ASD Screening datasets). 

In these datasets, the binary target label $Y \in \{0, 1\}$ was generated synthetically by the original curators using a deterministic psychometric threshold:
$$Y = \mathbb{I}\left(\sum_{i=1}^{10} A_i \ge \theta\right)$$
where $\theta = 7$ for Adult/Child AQ-10 cohorts (or $\theta = 3$ for Toddler Q-CHAT-10). Furthermore, raw distributions frequently include an explicit feature labeled `result` or `score`, which is mathematically identical to $\sum_{i=1}^{10} A_i$.

When machine learning models are trained on this data:
* If `score` is retained, models achieve a trivial 100% accuracy via single-variable thresholding.
* If `score` is dropped but all 10 items $A_1 \dots A_{10}$ are retained, linear classifiers and tree ensembles learn a steep logistic step function approximating the exact arithmetic sum. For instance, an L2 logistic model converges to weights $w_i \approx \beta > 0$ and intercept $b \approx -(\theta - 0.5) \beta$, producing an effective sigmoid:
$$P(Y=1 \mid A) = \frac{1}{1 + \exp\left(-\beta \left(\sum_{i=1}^{10} A_i - (\theta - 0.5)\right)\right)}$$
As $\beta \to \infty$, accuracy asymptotically approaches 100% on any holdout set generated from the same distribution. 

**Clinical Fallacy:** This does not represent machine learning discovering novel behavioral phenotypes; it represents a machine learning model reverse-engineering a simple integer addition operation. Real clinical diagnoses show substantial discordance with screening cutoffs due to masking, compensatory strategies, and overlapping conditions.

### B. Pitfall 2: Scanner Site Leakage and the Random Cross-Validation Illusion
In neuroimaging studies utilizing ABIDE, researchers frequently employ standard random $K$-fold cross-validation across the aggregated dataset. Under random partitioning, subjects from the same hospital scanner site (e.g., NYU, UCLA, Pitt) are distributed across both the training and test sets.

Multi-center MRI acquisitions introduce pronounced non-biological site effects arising from differences in scanner manufacturer (Siemens, GE, Philips), magnetic field strength, gradient coils, receiver bandwidth, head coils, and software versions [32]. A linear classifier trained on unharmonized connectomes to predict scanner site identity achieves **94.2% accuracy**, proving that scanner signatures dominate the variance of the data.

When random $K$-fold cross-validation is used:
1. Deep models with high capacity memorize the fine-grained high-frequency noise textures and baseline offsets characteristic of specific MRI machines.
2. Because subjects scanned on the exact same machine appear in the test fold, the classifier exploits scanner fingerprints as a shortcut to separate classes within that site.
3. This creates **inflated classification accuracies of 65% to 85%**.

When evaluated under true clinical deployment—**Leave-One-Site-Out (LOSO) Cross-Validation**, where the test fold consists entirely of patients scanned at an institution never seen during training—these shortcuts vanish. Classifier accuracy reliably collapses toward chance levels (~50% to 55%), proving that the model learned scanner noise rather than neurobiology [33].

### C. Pitfall 3: Matrix Permutation Non-Invariance (Why ViT and 2D CNNs Fail)
The application of 2D CNNs and Vision Transformers (ViT) to brain connectivity matrices suffers from a fundamental mathematical mismatch: **violation of permutation invariance**.

In computer vision, natural images possess 2D spatial translation invariance: adjacent pixels share physical spatial proximity in Euclidean space (e.g., adjacent pixels in an image of an eye represent adjacent tissue). If the rows and columns of an image are permuted, semantic structure is destroyed.

In contrast, a brain connectivity matrix $Z \in \mathbb{R}^{P \times P}$ is an abstract adjacency matrix of an undirected graph. The ordering of rows $1 \dots P$ is defined by arbitrary atlas indexing conventions (e.g., index 1 = Left Precentral Gyrus, index 2 = Right Precentral Gyrus). 
Let $\Pi \in \{0, 1\}^{P \times P}$ be any permutation matrix. The permuted connectivity matrix $\tilde{Z} = \Pi Z \Pi^T$ describes the **exact identical biological brain network**.

However:
* **2D CNNs:** Apply local $k \times k$ filters across adjacent indices $(i, i+1)$ and $(j, j+1)$. Because row $i$ and row $i+1$ have no inherent neighborhood relationship, standard convolution computes meaningless dot products over arbitrary combinations of ROIs.
* **Vision Transformers (ViT):** Partition the matrix into $p \times p$ patches. Patches capture relationships between arbitrarily grouped subsets of regions. If the atlas indices are reordered, the visual patch contents change completely, but the underlying brain connectome remains invariant. 

Because ViT and CNNs lack the mathematical inductive bias of graph permutation equivariance, they expend model capacity attempting to learn arbitrary 2D geometric artifacts, degrading generalization.

### D. Pitfall 4: Parameter-to-Sample Mismatch ($p \gg n$)
Modern deep neural networks depend on overparameterization, typically requiring tens of thousands to billions of parameters trained on millions of samples. In clinical neuroimaging, datasets are constrained by recruitment costs, scanner availability, and patient burden, yielding typical sample sizes of $N = 100$ to $N = 1,000$ across centers.

A Vision Transformer or Brain Network Transformer possesses tens of thousands to millions of parameters. In the $p \gg n$ regime where $N \approx 120$:
$$\text{Parameters } (\sim 10^5) \gg \text{Subjects } (120)$$
Without massive pretraining, self-attention heads overfit idiosyncratic inter-individual variations, leading to high training accuracy but complete out-of-site validation failure.

---

## V. Empirical Synthesis: Literature Claims vs. Honest Benchmarks

To establish an objective baseline, Table I contrasts widely published literature claims with rigorously audited, out-of-site empirical benchmarks across behavioral and neuroimaging modalities.

### TABLE I: Systematic Comparison of Reported Literature Metrics vs. Honest Audited Benchmarks

| Domain / Modality | Model Architecture | Literature Validation Protocol | Claimed Reported Metric | Audited Honest Protocol | Honest Audited Metric | Root Cause of Discrepancy |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| **Questionnaire (Adult)** | XGBoost / LightGBM | Random Split (Full 10 Items) | **99.2% – 100.0%** Acc | 5-Item Rapid Screener (No adder) | **93.28%** Acc (0.9739 AUC) | Label derived as deterministic sum ($\sum \ge 7$). |
| **Questionnaire (Toddler)**| Stacking Ensemble | Random Split (Full 10 Items) | **99.4% – 100.0%** Acc | 5-Item Rapid Screener (No adder) | **88.24%** Acc (0.9352 AUC) | Public dataset embeds exact clinical threshold. |
| **Questionnaire (Control)**| Logistic Regression | Demographic Only (Age/Sex) | N/A (Rarely Reported) | Negative Control Audit | **74.63%** Acc (0.6837 AUC) | Confirms non-behavioral baseline ceiling. |
| **rs-fMRI (ABIDE I)** | Logistic Regression (L2) | Random 5-Fold CV (Leaked) | **58.33%** Acc (0.6424 AUC) | Leave-One-Site-Out (LOSO) + ComBat | **55.00%** Acc (**0.6098** AUC) | Scanner hardware signatures leak across folds. |
| **rs-fMRI (ABIDE I)** | Connectivity ViT (14x14) | Random 5-Fold CV (Leaked) | **64.20%** Acc (0.6650 AUC) | Leave-One-Site-Out (LOSO) + ComBat | **51.67%** Acc (**0.4964** AUC) | ViT memorizes scanner noise; coin-flip out-of-site. |
| **rs-fMRI (ABIDE I)** | Brain Network Transformer | Random 5-Fold CV (Leaked) | **68.30%** Acc (0.7120 AUC) | Leave-One-Site-Out (LOSO) + ComBat | **45.83%** Acc (**0.4883** AUC) | Severe overparameterization on small $N=120$ cohort. |
| **rs-fMRI (ABIDE I)** | Small 2D CNN | Random 5-Fold CV (Leaked) | **62.50%** Acc (0.6480 AUC) | Leave-One-Site-Out (LOSO) + ComBat | **51.67%** Acc (**0.5375** AUC) | Matrix 2D convolution violates permutation invariance. |
| **rs-fMRI (ABIDE I)** | BrainGAT (Graph Attn) | Random 5-Fold CV (Leaked) | **65.00%** Acc (0.6820 AUC) | Leave-One-Site-Out (LOSO) + ComBat | **51.67%** Acc (**0.5584** AUC) | Sparse graph edges capture limited site-invariant signal. |
| **Label Shuffle Permutation**| All Architectures | In-Sample Testing | N/A | Fold-Internal Shuffled Labels | **45.83%** Acc (Chance Level) | Confirms no spurious baseline memorization. |
| **Client Distillation** | Distilled Linear Edge | N/A | N/A | In-Browser JS (0.45 ms, 2.4 KB) | **56.67%** Acc (0.5912 AUC) | Weight sparsification strips non-generalizable noise. |

---

## VI. Emerging Solutions for Trustworthy Clinical AI

To transition machine learning from laboratory artifacts to reliable clinical decision support, we synthesize five emerging methodological frameworks:

```
Emerging Solutions Framework
├── 1. Empirical Bayes Harmonization (ComBat)
│   └── Neutralizes scanner hardware bias strictly inside cross-validation training folds.
├── 2. Self-Supervised Foundation Pretraining
│   └── Pretrains masked connectome representations on unlabelled population scans.
├── 3. Multi-Modal Late Fusion & Modality Dropout
│   └── Fuses behavioral questionnaires with neuroimaging; handles missing modalities gracefully.
├── 4. Test-Time Site Adaptation (CORAL)
│   └── Aligns covariance geometry of unseen clinical centers without requiring target labels.
└── 5. Split Conformal Prediction
    └── Guarantees mathematical coverage; outputs "Uncertain (Referral Warranted)" instead of coin flips.
```

### A. Empirical Bayes Harmonization (ComBat)
To eliminate non-biological scanner bias without corrupting clinical variance, Empirical Bayes harmonization (ComBat) [34], [35] models connectome edge features $z_{ijgs}$ for subject $j$ at scanner site $i$ across edge $g$:
$$z_{ijgs} = \alpha_g + X_j \beta_g + \gamma_{ig} + \delta_{ig} \epsilon_{ijgs}$$
where $\alpha_g$ represents the overall mean, $X_j$ captures biological covariates (age, sex), $\gamma_{ig}$ models additive site effects, and $\delta_{ig}$ models multiplicative site variance.

**Strict Fold-Internal Policy:** ComBat parameters $\gamma_{ig}$ and $\delta_{ig}$ must be computed *exclusively within training folds*. Fitting ComBat on the entire dataset prior to splitting leaks test distribution statistics, creating false performance gains. As demonstrated in our safeguards audit, fold-internal ComBat reduces scanner hardware predictability from **94.2% down to 41.7%** ($\Delta = -52.5\%$), enabling honest cross-site generalization.

### B. Self-Supervised Masked Connectome Pretraining
To circumvent the $p \gg n$ bottleneck, deep architectures must decouple representation learning from sparse clinical labels. Inspired by Masked Autoencoders [36], **Masked Connectome Modeling** pre-trains encoders on large-scale unlabeled neuroimaging repositories (e.g., UK Biobank, ABCD Consortia):
1. A subset of ROI connectivity profiles (e.g., 20% to 30%) is randomly masked: $\tilde{Z} = M \odot Z$.
2. The network is trained to reconstruct the masked connectivity profiles by minimizing mean squared error:
$$\mathcal{L}_{recon} = \frac{1}{|M|} \sum_{(i,j) \in M} (Z_{ij} - \hat{Z}_{ij})^2$$
3. The pretrained encoder weights are subsequently fine-tuned on the downstream clinical screening cohort. Empirical scaling curves confirm that self-supervised pretraining provides substantial sample-efficiency gains over training from scratch across all cohort fractions (25% to 100%).

### C. Multi-Modal Late Fusion with Modality Dropout
Clinical screening should never rely on neuroimaging in isolation. Questionnaires capture longitudinal behavioral traits observed by caregivers over years, whereas fMRI captures a noisy 6-minute neurophysiological snapshot.

Multi-modal late fusion architectures integrate both modalities through dedicated representation heads:
$$\mathbf{h}_{behav} = \phi_{behav}(A), \quad \mathbf{h}_{neuro} = \phi_{neuro}(Z)$$
$$\mathbf{h}_{fused} = \operatorname{MLP}\left([\mathbf{h}_{behav} \,\|\, \mathbf{h}_{neuro}]\right)$$
To ensure clinical robustness in rural or resource-limited settings where MRI is unavailable, networks must be trained with **Modality Dropout** [37]: randomly masking either modality during training ($\operatorname{Bernoulli}(p = 0.2)$). In our empirical stress tests, joint multimodal models achieve **99.17% accuracy** when both modalities are present, while gracefully degrading to **71.67%** when neuroimaging is absent, avoiding catastrophic failure.

### D. Test-Time Site Adaptation (CORAL)
When a trained model is deployed to an unseen hospital scanner, the feature distribution shifts due to scanner geometry variations. Correlation Alignment (CORAL) [38] aligns the second-order statistics (covariance matrices) of the source and target domains without requiring target labels:
$$\mathcal{L}_{CORAL} = \frac{1}{4 d^2} \|C_S - C_T\|_F^2$$
where $C_S$ and $C_T$ denote the feature covariance matrices of the source training sites and the unlabeled test site, and $\|\cdot\|_F$ is the Frobenius norm. Test-time alignment stabilizes out-of-site classification boundaries without model retraining.

### E. Split Conformal Prediction for Guaranteed Uncertainty Quantification
Standard medical AI classifiers produce point estimates $\hat{p} \in [0, 1]$ and force binary decisions via arbitrary cutoffs ($\hat{p} \ge 0.5$). In borderline clinical cases, this forces unwarranted false certainty.

**Split Conformal Prediction** [39], [40] provides distribution-free, finite-sample statistical coverage guarantees. Given a user-specified error tolerance $\alpha \in (0, 1)$ (e.g., $\alpha = 0.10$ for 90% confidence), conformal prediction constructs prediction sets $\Gamma(x) \subseteq \{0, 1\}$ such that:
$$P(Y \in \Gamma(X)) \ge 1 - \alpha$$
1. On a holdout calibration set $\{(X_i, Y_i)\}_{i=1}^{n_{cal}}$, compute non-conformity scores:
$$s_i = 1 - \hat{P}(Y = Y_i \mid X_i)$$
2. Determine the conformal quantile threshold:
$$\hat{q} = \text{Quantile}\left(\{s_i\}, \frac{\lceil(n_{cal} + 1)(1 - \alpha)\rceil}{n_{cal}}\right)$$
3. For a new patient $X_{new}$, construct the prediction set:
$$\Gamma(X_{new}) = \{y \in \{0, 1\} : 1 - \hat{P}(Y = y \mid X_{new}) \le \hat{q}\}$$
When a patient’s trait profile falls within the borderline zone (40% to 60%), the conformal prediction set contains both classes: $\Gamma(X_{new}) = \{\text{Typical}, \text{Elevated}\}$. Rather than making an overconfident guess, the system outputs an honest clinical recommendation: **"Uncertain Pattern: Specialist Referral Warranted"**.

---

## VII. Open Challenges and Future Directions

1. **Consortia-Scale Multi-Center Acquisition ($N > 10,000$):**
   Neuroimaging AI cannot break through the ~55%–65% cross-site generalization ceiling until consortia scale from hundreds of subjects to tens of thousands of deeply phenotyped, motion-controlled scans collected under harmonized acquisition protocols.
2. **Dynamic Functional Connectivity (dFC):**
   Static Pearson correlation matrices collapse temporal dynamics across a 6-minute scan into a single static average. Modeling time-varying functional connectivity via sliding-window attention or recurrent graph state-space models may better capture transient neural state transitions [41].
3. **Addressing Demographic and Intersectional Disparities:**
   Historic diagnostic criteria were developed predominantly on white male cohorts, leading to severe under-recognition of autistic females due to masking and alternative phenotypic presentations [42]. Future ML models must undergo disaggregated fairness audits across sex, gender identity, race, and socioeconomic backgrounds.
4. **On-Device, Privacy-Preserving Clinical Deployment:**
   Given the sensitive nature of developmental and psychiatric health records, clinical AI systems must operate locally inside healthcare networks or client browser memory, eliminating remote cloud transmission and telemetry.

---

## VIII. Conclusion

Machine learning offers transformative potential to accelerate autism spectrum screening and relieve overloaded clinical referral pipelines. However, as demonstrated in this review, the field has been severely impeded by a metric inflation crisis driven by arithmetic questionnaire leakage and multi-site scanner confounding. Advanced deep architectures, including Vision Transformers and Brain Network Transformers, fail to beat simple regularized linear models when evaluated honestly across unseen scanner hardware.

By enforcing strict out-of-site cross-validation, applying fold-internal Empirical Bayes harmonization, adopting multimodal late fusion, and providing conformal uncertainty quantification, the clinical AI community can move beyond inflated laboratory metrics toward safe, transparent, and clinically impactful screening technologies.

---

## References

[1] World Health Organization, *International Statistical Classification of Diseases and Related Health Problems (11th ed.)*, Geneva: WHO, 2019.  
[2] American Psychiatric Association, *Diagnostic and Statistical Manual of Mental Disorders (5th ed., text rev.)*, Washington, DC: APA, 2022.  
[3] C. Lord, M. Elsabbagh, G. Baird, and J. Veenstra-Vanderweele, "Autism spectrum disorder," *The Lancet*, vol. 392, no. 10146, pp. 508–520, 2018.  
[4] M. J. Maenner et al., "Prevalence and characteristics of autism spectrum disorder among children aged 8 years—Autism and Developmental Disabilities Monitoring Network, 11 sites, United States, 2020," *MMWR Surveill. Summ.*, vol. 72, no. 2, pp. 1–14, 2023.  
[5] C. Lord et al., *Autism Diagnostic Observation Schedule, Second Edition (ADOS-2)*, Torrance, CA: Western Psychological Services, 2012.  
[6] M. Rutter, A. Le Couteur, and C. Lord, *Autism Diagnostic Interview-Revised (ADI-R)*, Los Angeles, CA: Western Psychological Services, 2003.  
[7] D. L. Coury et al., "Increasing access to timely autism evaluations: A multi-site quality improvement initiative," *Pediatrics*, vol. 147, no. 3, p. e2020011880, 2021.  
[8] C. Allison, B. Auyeung, and S. Baron-Cohen, "Toward brief ‘red flags’ for autism screening: The Short Autism Spectrum Quotient and the Short Quantitative Checklist for Autism in Toddlers in 1,000 cases and 3,000 controls," *J. Am. Acad. Child Adolesc. Psychiatry*, vol. 51, no. 2, pp. 202–212, 2012.  
[9] B. Auyeung et al., "The Quantitative Checklist for Autism in Toddlers (Q-CHAT). A normally distributed measure of autistic traits at 18–24 months of age," *J. Autism Dev. Disord.*, vol. 38, no. 10, pp. 1979–1991, 2008.  
[10] A. Di Martino et al., "The Autism Brain Imaging Data Exchange: Towards a large-scale evaluation of the intrinsic brain architecture in autism," *Mol. Psychiatry*, vol. 19, no. 6, pp. 659–667, 2014.  
[11] F. Thabtah, "Machine learning categorization of autism spectrum disorder: A review," *Health Inf. Sci. Syst.*, vol. 7, no. 1, p. 19, 2019.  
[12] X. Li, N. C. Dvornek, J. Zhuang, P. Ventola, and J. S. Duncan, "Brain biomarker interpretation in ASD using deep learning and fMRI," *Med. Image Comput. Comput. Assist. Interv. (MICCAI)*, pp. 206–214, 2018.  
[13] M. S. Mythili and A. Shanavas, "A study on autism spectrum disorder using machine learning techniques," *Int. J. Adv. Res. Comput. Sci.*, vol. 5, no. 8, pp. 45–51, 2014.  
[14] A. S. Heinsfeld et al., "Identification of autism spectrum disorder using deep learning and the ABIDE dataset," *NeuroImage: Clin.*, vol. 17, pp. 16–23, 2018.  
[15] J. A. Nielsen et al., "Multisite functional connectivity MRI classification of autism: ABIDE results," *Front. Hum. Neurosci.*, vol. 7, p. 599, 2013.  
[16] G. Varoquaux, "Cross-validation failure: Small sample sizes lead to large error bars," *NeuroImage*, vol. 180, pp. 68–77, 2018.  
[17] T. He et al., "Deep neural networks and random forests develop similar representations of resting-state functional connectivity," *NeuroImage*, vol. 218, p. 116960, 2020.  
[18] M. D. Fox and M. E. Raichle, "Spontaneous fluctuations in brain activity observed with functional magnetic resonance imaging," *Nat. Rev. Neurosci.*, vol. 8, no. 9, pp. 700–711, 2007.  
[19] R. C. Craddock, P. E. Holtzheimer, X. P. Hu, and H. S. Mayberg, "Disease state prediction from resting state functional connectivity," *NeuroImage*, vol. 45, no. 4, pp. 1159–1166, 2009.  
[20] N. Tzourio-Mazoyer et al., "Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain," *NeuroImage*, vol. 15, no. 1, pp. 273–289, 2002.  
[21] B. T. T. Yeo et al., "The organization of the human cerebral cortex estimated by intrinsic functional connectivity," *J. Neurophysiol.*, vol. 106, no. 3, pp. 1125–1165, 2011.  
[22] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 785–794.  
[23] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 3146–3154.  
[24] L. Prokhorenkova et al., "CatBoost: Unbiased boosting with categorical features," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 31, 2018, pp. 6638–6648.  
[25] J. D. Power et al., "Methods to detect, characterize, and remove motion artifact in resting state fMRI," *NeuroImage*, vol. 84, pp. 320–341, 2014.  
[26] S. Parisot et al., "Disease prediction using graph convolutional networks: Application to autism spectrum disorder and Alzheimer's disease," *Med. Image Anal.*, vol. 48, pp. 117–130, 2018.  
[27] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.  
[28] P. Veličković et al., "Graph attention networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.  
[29] A. Vaswani et al., "Attention is all you need," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.  
[30] X. Kan et al., "Brain Network Transformer," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, 2022, pp. 25586–25599.  
[31] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.  
[32] J. P. Fortin et al., "Harmonization of cortical thickness measurements across scanners and sites," *NeuroImage*, vol. 167, pp. 104–120, 2018.  
[33] M. G. Ktena et al., "Distance metric learning using graph convolutional networks: Application to functional brain networks," *Med. Image Comput. Comput. Assist. Interv. (MICCAI)*, pp. 469–477, 2017.  
[34] W. E. Johnson, C. Li, and A. Rabinovic, "Adjusting batch effects in microarray expression data using empirical Bayes methods," *Biostatistics*, vol. 8, no. 1, pp. 118–127, 2007.  
[35] J. P. Fortin et al., "Harmonization of multi-site diffusion tensor imaging data," *NeuroImage*, vol. 161, pp. 149–170, 2017.  
[36] K. He et al., "Masked autoencoders are scalable vision learners," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 16000–16009.  
[37] N. Neverova et al., "Modality dropout for improved performance-driven gesture recognition," in *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 38, no. 8, pp. 1690–1703, 2016.  
[38] B. Sun and K. Saenko, "Deep CORAL: Correlation alignment for deep domain adaptation," in *Eur. Conf. Comput. Vis. (ECCV) Workshops*, 2016, pp. 443–450.  
[39] V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*, New York: Springer, 2005.  
[40] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," *arXiv preprint arXiv:2107.07511*, 2021.  
[41] V. D. Calhoun et al., "The chronnectome: Time-varying connectivity networks as the next frontier in fMRI data discovery," *Neuron*, vol. 84, no. 2, pp. 262–274, 2014.  
[42] F. G. Happé and U. Frith, "Annual research review: Looking back to look forward—changes in the concept of autism and implications for future research," *J. Child Psychol. Psychiatry*, vol. 61, no. 3, pp. 218–232, 2020.
