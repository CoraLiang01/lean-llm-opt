#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$: Set of all products classified as ‘FAUX’ in the dataset.

**Parameters:**

- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv).
- $d_i$: Deterministic demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Demand fulfillment constraint:**
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
2. **Inventory constraint:**
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Source file:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv
- **Index set $I$:** All rows where the product is classified as ‘FAUX’ (as per the user query; if ‘FAUX’ is a substring or category, select accordingly).
- **Parameter $A_i$:** Column ‘Revenue’
- **Parameter $d_i$:** Column ‘Demand’
- **Parameter $I_i$:** Column ‘Initial Inventory’