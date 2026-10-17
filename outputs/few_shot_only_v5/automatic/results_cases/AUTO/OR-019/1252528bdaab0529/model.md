#### Abstract Mathematical Optimization Model

**Index Sets:**
- Let $\mathcal{I}$ be the set of all products classified as ‘27in’ (i.e., all products whose name contains ‘27in’ in the source data).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in \mathcal{I}$ (from column ‘Revenue’ in the source file).
- $d_i$: Demand for product $i \in \mathcal{I}$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in \mathcal{I}$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in \mathcal{I}$ to fulfill, where $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

#### Data Mapping

- **Source File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv
- **Index Set $\mathcal{I}$:** All rows where the ‘Product Name’ column contains ‘27in’.
- **Parameter $A_i$:** Value from column ‘Revenue’ for product $i$.
- **Parameter $d_i$:** Value from column ‘Demand’ for product $i$.
- **Parameter $I_i$:** Value from column ‘Initial Inventory’ for product $i$.
- **Decision Variable $x_i$:** Number of units to fulfill for product $i \in \mathcal{I}$.

No literal data values or record counts are included; all data is referenced symbolically as required.