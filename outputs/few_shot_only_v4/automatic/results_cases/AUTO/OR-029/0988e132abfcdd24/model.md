#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$: Set of all products classified under ‘FAUX’ in the dataset.

**Parameters:**

- $a_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv).
- $d_i$: Deterministic demand for product $i \in I$ (from column ‘Demand’).
- $s_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**

$$
\max \sum_{i \in I} a_i x_i
$$

**Constraints:**

1. **Inventory Constraint:** 
   $$
   x_i \leq s_i \quad \forall i \in I
   $$
2. **Demand Constraint:** 
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Source File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv
- **Index Set $I$:** All rows where the product is classified under ‘FAUX’ (as defined by the user query; the CSVQA tool returns all rows, so the model applies this subset explicitly).
- **Parameter $a_i$:** Column ‘Revenue’
- **Parameter $d_i$:** Column ‘Demand’
- **Parameter $s_i$:** Column ‘Initial Inventory’

No additional constraints or eligibility rules are specified in the query. All FAUX products are eligible for selection. The model maximizes total revenue subject to inventory and demand limits for each product.