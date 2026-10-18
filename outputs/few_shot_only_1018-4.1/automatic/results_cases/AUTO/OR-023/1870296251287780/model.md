#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified as ‘ELE-S’ in the dataset.

**Parameters:**
- $r_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv).
- $d_i$: Deterministic demand for product $i \in I$ (from column ‘Demand’).
- $s_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory limit:** 
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source file:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv
- **Index set $I$:** All rows where the ‘Product_Reference’ column begins with ‘ELE-S’.
- **Parameter $r_i$:** Value from column ‘Revenue’ for each $i \in I$.
- **Parameter $d_i$:** Value from column ‘Demand’ for each $i \in I$.
- **Parameter $s_i$:** Value from column ‘Initial Inventory’ for each $i \in I$.
- **Decision variable $x_i$:** Number of units to fulfill for each $i \in I$.

No additional constraints or eligibility rules are specified in the query. All ‘ELE-S’ products in the file are included.