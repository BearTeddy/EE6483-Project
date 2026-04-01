# Literature Survey Foundations (Assignment a1 "proof")

This note addresses the required clarification of problem definitions, key challenges, and common solution types across different sentiment-analysis settings.

## 1) Problem Definitions

- `Document-level sentiment classification`: predict one sentiment label for an entire review.
- `Sentence-level sentiment classification`: predict sentiment for each sentence.
- `Aspect-based sentiment analysis (ABSA)`: predict sentiment toward specific aspects (for example, `price`, `quality`).
- `Target-dependent sentiment analysis (TDSA)`: predict sentiment toward a provided entity/target in context.
- `Emotion-aware sentiment analysis`: predict affective states (joy, anger, etc.) alongside or instead of polarity labels.

## 2) Core Challenges

- `Context and compositionality`: negation ("not good"), intensifiers ("very good"), contrast ("good but expensive").
- `Domain-specific vocabulary`: words may flip polarity by domain (for example, "unpredictable plot" vs "unpredictable product behavior").
- `Class imbalance`: real data often has skewed label distribution.
- `Annotation quality`: noisy/inconsistent labels reduce achievable performance.
- `Out-of-distribution drift`: language style and product categories change over time.
- `Granularity mismatch`: review-level labels are weak supervision for aspect-level targets.

## 3) Supervised vs Unsupervised Settings

### Supervised sentiment learning

- Definition: train with labeled examples `(text, sentiment)`.
- Common solutions:
  - Classical ML over sparse features: TF-IDF + Logistic Regression / SVM / Naive Bayes.
  - Neural sequence models: CNN, RNN (LSTM/GRU), attention variants.
  - Pretrained language models: BERT/RoBERTa fine-tuning.
- Strengths: high task accuracy when labels are sufficient and in-domain.
- Weaknesses: labeling cost and reduced robustness under domain shift.

### Unsupervised / weakly supervised sentiment learning

- Definition: no gold labels, or only weak proxy labels.
- Common solutions:
  - Lexicon-based polarity scoring.
  - Topic-sentiment co-modeling.
  - Self-training/pseudo-labeling.
  - Contrastive pretraining + clustering.
- Strengths: no/low annotation cost.
- Weaknesses: lower precision and harder calibration.

## 4) Closed-set vs Open-set

### Closed-set classification

- Definition: test-time labels are among known training labels.
- Typical setup for this project: binary `{negative, positive}`.
- Common solutions: standard softmax classifiers, calibrated probabilities, threshold tuning.

### Open-set / open-world sentiment

- Definition: unknown intents, mixed emotions, or unseen classes can appear at test time.
- Challenges: overconfident wrong predictions on unknown samples.
- Common solutions:
  - Confidence/energy-based rejection.
  - Distance-to-centroid or Mahalanobis uncertainty checks.
  - Open-set fine-tuning / outlier exposure.

## 5) With vs Without Domain Shift

### Without domain shift (in-domain)

- Train/test from similar distribution.
- Strong baselines: TF-IDF+SVM and BERT-style fine-tuning.
- Typical optimization: hyperparameter tuning and regularization.

### With domain shift (cross-domain)

- Example: train on product reviews, deploy on hotel reviews.
- Challenges: vocabulary and sentiment cues shift; label priors may change.
- Common solutions:
  - Domain-adaptive pretraining (DAPT/TAPT) for language models.
  - Adversarial domain adaptation.
  - Multi-domain training and feature alignment.
  - Semi-supervised target-domain self-training.

## 6) Baseline Choice and Improvement Direction for This Project

- Suitable baseline: `TF-IDF + linear classifier (LogReg/SVM)` for fast, reproducible performance.
- Neural extension: `TextCNN` and `BiLSTM/BiGRU` for sequence-aware modeling.
- Stronger extension: `BERT` and `RoBERTa` fine-tuning for context-rich representations.
- Recommended improvements:
  - Cross-validation and threshold calibration for robustness.
  - Error slicing (length, negation, aspect keywords).
  - Domain-shift stress testing with external review domains.

## Key references from assignment brief

1. Blitzer et al., ACL 2007 (domain adaptation for sentiment classification).  
2. Mikolov et al., 2013 (word2vec).  
3. Pennington et al., EMNLP 2014 (GloVe).  
4. Devlin et al., NAACL 2019 (BERT).
