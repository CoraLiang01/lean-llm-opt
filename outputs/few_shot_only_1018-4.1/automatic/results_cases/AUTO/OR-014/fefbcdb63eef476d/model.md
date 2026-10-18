#### Abstract Mathematical Optimization Model

**Index Sets:**
- Let $\mathcal{I}$ be the set of all pizza types, indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit of pizza type $i$ (from column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv).
- $d_i$: Total demand for pizza type $i$ over the sales horizon (from column ‘Demand’).
- $I_i$: Initial inventory available for pizza type $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integer), for all $i \in \mathcal{I}$.

**Objective:**
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory and Demand Fulfillment:**
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in \mathcal{I}
   \]
   (Equivalently, two separate constraints:)
   \[
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   \]
   \[
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   \]
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I}
   \]

---

#### Data Mapping

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv
    - **Index Set $\mathcal{I}$:** All rows in the file (each row is a pizza type, identified by ‘Product Name’).
    - **Parameter $A_i$:** Column ‘Revenue’.
    - **Parameter $d_i$:** Column ‘Demand’.
    - **Parameter $I_i$:** Column ‘Initial Inventory’.