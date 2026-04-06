# 🧠 Deep Learning Core: From First Principles to Transformers

This repository documents the architectural progression of modern Deep Learning, built from the ground up. 

Rather than relying purely on high-level abstractions, I implemented foundational models—including an Autograd engine, Vectorized Optimizers, and the Transformer architecture—from scratch. This ensures a strict, first-principles understanding of gradient flow, numerical stability, and tensor memory allocation before scaling to production frameworks like PyTorch and HuggingFace.

---

## 🏗️ Architecture & Progression

The repository is structured to demonstrate the evolution from scalar-based computational graphs to matrix-based neural networks, culminating in state-of-the-art sequence models.

### Part I: Statistical Foundations & The Autograd Engine
* **`01_statistical_learning/`**: Fully vectorized implementations of Linear and Logistic Regression.
* **`02_autograd_engine/`**: A custom reverse-mode automatic differentiation engine operating on scalar values (inspired by Micrograd), demonstrating topological sorting for backpropagation.
* **`03_unsupervised/`**: High-performance, broadcasting-optimized implementations of K-Means and PCA (using `np.linalg.eigh` for symmetric covariance matrices).
* **`04_trees/`**: An ID3/C4.5 style Decision Tree utilizing Information Gain and recursive node splitting.

### Part II: The Matrix Engine
* **`05_deep_learning_core/`**: A custom, modular deep learning framework built in NumPy.
  * Implements **Inverted Dropout** to prevent memory leaks during inference.
  * Features an **Adam Optimizer** with bias correction.
  * Utilizes numerical stability tricks (e.g., max-subtraction in Softmax) to prevent exponential overflow.

### Part III: Framework Mastery & LLMs
* **`06_pytorch_foundations/`**: Translating custom architectures into PyTorch, enforcing strict separation of Datasets, Models, and hardware-agnostic Training loops.
* **`07_computer_vision/`**: CNN architectures with explicit spatial dimension tracking to prevent Out-Of-Memory (OOM) errors at scale.
* **`08_sequence_models/`**: Distinct LSTM architectures for continuous Time-Series Forecasting vs. discrete Sequence Classification.
* **`09_transformers_and_llms/`**: 
  * From-scratch implementation of **Scaled Dot-Product Multi-Head Attention** and the standard Encoder-Decoder Transformer.
  * An autoregressive **Greedy Decoding** engine.
  * Fine-tuning **DistilBERT** (via HuggingFace) for downstream enterprise tasks, specifically unstructured Security Log Classification.

---

## 📐 Mathematical Intuition

Understanding the underlying calculus is critical for debugging complex architectures. Here are a few core concepts implemented in this repository:

**1. Log Loss (Binary Cross-Entropy)**
To navigate elongated loss valleys in binary classification, the Logistic model optimizes:
$$L = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

**2. Scaled Dot-Product Attention**
The Transformer engine prevents softmax saturation by scaling the dot product of Queries and Keys by the square root of their dimension:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**3. Adam Optimizer (First Moment)**
The custom NumPy implementation tracks the exponentially decaying average of past gradients:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad \text{(Bias Correction)}$$

---

## 🚀 Installation & Usage

To explore the implementations or run the training scripts locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/SauravSJK/ml-from-scratch.git](https://github.com/SauravSJK/ml-from-scratch.git)
   cd ml-from-scratch

2. Create a clean virtual environment and install dependencies:

    ```python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt

3. Run a specific module (e.g., the custom Transformer training loop):
   ```bash
   python 09_transformers_and_llms/train_toy_task.py