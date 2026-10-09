#### Symbolic Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products whose "Product Name" contains '27in'.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$\max \sum_{i \in \mathcal{I}} A_i x_i$

**Constraints:**
1. Inventory and Demand Bounds:  
$\forall i \in \mathcal{I}: \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

2. Integrality:  
$\forall i \in \mathcal{I}: \quad x_i \in \mathbb{Z}_+$

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv`
    - Index set $\mathcal{I}$: All rows where "Product Name" contains '27in'
    - Parameter $A_i$: Column "Revenue"
    - Parameter $d_i$: Column "Demand"
    - Parameter $I_i$: Column "Initial Inventory"