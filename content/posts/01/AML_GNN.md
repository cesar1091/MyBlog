---
title: "Catching Financial Crime: An AML Classifier Model Using Graph Neural Networks"
description: "Implementing a topology-aware Graph Neural Network (GNN) to detect money laundering activities from synthetic transaction data, designed to perform under extreme real-world class imbalance."
tags: ["GNN", "Imbalanced Classification", "Financial Crime"]
---


This project tackles one of the most challenging problems in financial security: **Anti-Money Laundering (AML)** detection. We implemented a cutting-edge **Graph Neural Network (GNN)**-based classifier to identify potential money laundering (ML) activities using synthetic transaction data.

Our core innovation is modeling the financial ecosystem as a complex graph structure and developing a custom, **topology-aware strategy** to handle the extreme class imbalance inherent in real-world AML datasets ($<0.1\%$ suspicious cases).


## 💡 Why Graphs? Modeling Financial Relationships

Money laundering is fundamentally a crime of **relationships and flow**. Suspicious patterns rarely involve isolated transactions; they emerge from the **connections between clients**, the **direction of funds**, and **behavioral changes over time**.

To capture this relational context, we model the financial ecosystem as an **Attributed Transaction Graph** .

  * **Nodes (Clients):** Represent customers.
  * **Edges (Transactions):** Represent financial interactions or transfers between clients.
  * **Attributes:** Both nodes and edges are enriched with important AML-related features.


## 📈 Data and Graph Feature Engineering

We use **synthetic data** derived from confirmed criminal cases in LATAM, enriched with attributes crucial to AML investigations. These features allow the GNN to learn a comprehensive risk profile.

### **Client (Node) Attributes**

These features capture the intrinsic risk of a client:

  * **Geographic Risk Level:** Risk associated with the client's location.
  * **Economic Activity Risk:** Risk of the client's registered business or profession.
  * **Historical Risk Rating:** Previous internal compliance scores.
  * **Total Transaction Volume:** Summary of their financial activity.

### **Transaction (Edge) Attributes**

These features capture the specific risk of the interaction:

  * **Transaction Amount:** The value of the transfer.
  * **Channel Used:** ATM, online transfer, branch visit, etc.
  * **Number of Interactions:** Frequency of transactions between two parties.


## ⚖️ The Imbalance Challenge: A Topology-Aware Strategy

AML detection is defined by **extreme class imbalance**; less than 1 in 1,000 clients may be suspicious. Traditional methods like naive oversampling can create unrealistic synthetic data, while undersampling loses valuable information.

Our custom **topology-aware mask strategy** addresses this by focusing the model's learning on the **geometric neighborhood** around confirmed suspicious cases, ensuring realism and relevance in every data split.

### **1. Suspicious Node Preservation**

All confirmed suspicious nodes ($y=1$) are first isolated and split into the standard 70/15/15 ratio for Train, Validation, and Test sets. This guarantees that all three sets contain genuine positive examples.

### **2. 2-Hop Neighborhood Extraction (Contextual Negatives)**

This is the critical step. For every suspicious node, we extract all **non-suspicious nodes within 2-hops** in the graph. .

  * **Why 2-Hops?** These nodes often represent **intermediaries**, money mules, or partners who are structurally linked to the criminal activity but might not be labeled as criminal themselves (yet). They serve as the most relevant "contextual negative" examples for the model.

### **3. Realistic Split Assignment**

The extracted pool of 2-hop neighbors (and a small fallback set of random nodes if needed) is then shuffled and split using the **exact same 70/15/15 ratios**.

  * **Result:** Each split (Train, Val, Test) maintains a realistic, high-imbalance ratio, but the non-suspicious samples are not random—they are **structurally adjacent** to the suspicious activities the model must learn to detect.

### **4. Imbalanced Sampling During Training**

Even with the focused splits, batches can be dominated by non-suspicious nodes. We use **`ImbalancedSampler`** within the **`NeighborLoader`** from PyTorch Geometric:

```python
sampler = ImbalancedSampler(data.y, input_nodes=data.train_mask)
loader = NeighborLoader(
    data,
    num_neighbors=[-1, -1], # full neighbors for local subgraph
    batch_size=batch_size,
    sampler=sampler
)
```

This dual approach ensures two key benefits:

1.  **Batch Balancing:** Each training batch contains a sufficient, proportional number of suspicious nodes.
2.  **Subgraph Integrity:** `NeighborLoader` retrieves the full local subgraph around the sampled batch nodes, preserving the crucial **connectivity and attribute structure** that GNNs rely on.

This combination avoids label leakage, maintains real-world imbalance, and focuses the learning on the complex sub-networks where money laundering occurs.


## 🧠 Model Architecture: Graph Attention Network (GAT)

We chose a **Graph Attention Network (GAT)** for the core classifier .

  * **Attention Mechanism:** GATs use a powerful **attention mechanism** to assign varying levels of importance to neighboring nodes and their transactions. This is ideal for financial networks where some relationships are critical (large, high-risk transfers) while others are noise.
  * **Attribute Integration:** The initial layers are customized to seamlessly integrate both the **node (client)** and **edge (transaction)** attributes into the message-passing framework.
  * **Robustness:** The use of **multi-head attention** improves robustness and generalization in noisy, irregular financial graphs.


## 📊 Evaluation: Focusing on Rare Event Detection

Given the extreme imbalance, standard accuracy is misleading. We rely on metrics tailored for rare event detection:

  * **Balanced Accuracy:** The arithmetic mean of sensitivity (True Positive Rate) and specificity (True Negative Rate).
  * **Precision, Recall, and F1-score:** Crucial for managing the trade-off between detecting true positives and minimizing false alerts.
  * **ROC-AUC:** Measures the model's ability to discriminate between positive and negative classes.
  * **Confusion Matrix:** Provides a clear visual breakdown of true/false positives and negatives.

## 🚀 Key Takeaways & Project Features

This project delivers an **end-to-end AML graph learning pipeline** that is uniquely suited for operational risk detection:

  * **Topology-Aware Sampling:** Custom strategy to preserve the relational context around suspicious cases.
  * **Feature-Rich Model:** Seamless integration of detailed node- and edge-level AML attributes.
  * **GAT Architecture:** Optimized for capturing directional and weighted importance in transaction flows.
  * **Real-World Imbalance:** Model training and evaluation preserve realistic class proportions.

Find the complete code and detailed documentation on GitHub: [Link](https://github.com/cesar1091/AML-Graph-Neural-Network-Imbalance-Classification)

