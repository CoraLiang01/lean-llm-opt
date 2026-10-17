#### Abstract Mathematical Optimization Model

**Index Sets:**
- $\mathcal{F}$: Set of all products classified as ‘Fashion’ in the dataset.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in \mathcal{F}$, from column ‘Revenue’.
- $d_i$: Deterministic demand for product $i \in \mathcal{F}$, from column ‘Demand’.
- $I_i$: Initial inventory for product $i \in \mathcal{F}$, from column ‘Initial Inventory’.

**Decision Variables:**
- $x_i$: Number of units of product $i \in \mathcal{F}$ to fulfill, integer, $x_i \geq 0$.

**Objective:**
\[
\max \sum_{i \in \mathcal{F}} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \quad \forall i \in \mathcal{F}
   \]
2. **Inventory limit:** 
   \[
   x_i \leq I_i \quad \forall i \in \mathcal{F}
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{F}
   \]

---

#### Data Mapping

- **File:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv`
- **Index Set $\mathcal{F}$:** All rows where the product is classified as ‘Fashion’ (i.e., product names containing or labeled as ‘Fashion’).
- **Parameter $A_i$:** Column ‘Revenue’.
- **Parameter $d_i$:** Column ‘Demand’.
- **Parameter $I_i$:** Column ‘Initial Inventory’.