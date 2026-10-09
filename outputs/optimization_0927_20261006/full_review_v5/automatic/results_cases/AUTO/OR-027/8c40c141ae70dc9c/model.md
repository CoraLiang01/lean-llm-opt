#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all products classified under ‘Organ’.

**Parameters:**

- $A_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Total deterministic demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory available for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective Function:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:**  
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraint:**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** SupermartGrocerySales-RetailAnalyticsDataset.csv
- **Table ID:** file_0_view_0
- **Columns Used:**
    - ‘Sub Category’ (filtered by prefix ‘Organic’)
    - ‘Revenue’ (parameter $A_i$)
    - ‘Demand’ (parameter $d_i$)
    - ‘Initial Inventory’ (parameter $I_i$)
- **Filter Applied:** Only rows where ‘Sub Category’ starts with ‘Organic’ (as validated and returned by CSVQA).