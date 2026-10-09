#### Symbolic Optimization Model

**Index Sets:**
- $I$: set of all pizza types (indexed by $i$)

**Parameters:**
- $A_i$: revenue per unit of pizza type $i$
- $d_i$: total demand for pizza type $i$ over the sales horizon
- $I_i$: initial inventory available for pizza type $i$

**Decision Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$  
  (number of units of pizza type $i$ to fulfill; non-negative integers)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory and Demand Fulfillment Bounds:**
   \[
   0 \leq x_i \leq \min\{I_i,\, d_i\}, \quad \forall i \in I
   \]
   (i.e., $x_i \leq I_i$ and $x_i \leq d_i$ for all $i$)

2. **Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv`
    - **Product Name** $\rightarrow$ Index set $I$
    - **Revenue** $\rightarrow$ Parameter $A_i$
    - **Demand** $\rightarrow$ Parameter $d_i$
    - **Initial Inventory** $\rightarrow$ Parameter $I_i$