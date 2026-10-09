#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : set of all products (indexed by $i$)

**Parameters:**

- $A_i$ : revenue per unit of product $i$ (from column ‘Revenue’ in table_id: file_0_view_0)
- $d_i$ : total demand for product $i$ over the sales horizon (from column ‘Demand’ in table_id: file_0_view_0)
- $I_i$ : initial inventory of product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0)

**Decision Variables:**

- $x_i$ : number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Demand fulfillment cannot exceed demand:**
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

2. **Cannot fulfill more than initial inventory:**
   $$
   x_i \leq I_i \quad \forall i \in I
   $$

3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index set $I$**: All product identifiers from column ‘Product Name’ in table_id: file_0_view_0 (WomenClothingEcommerceSalesData.csv)
- **Parameter $A_i$**: ‘Revenue’ column in table_id: file_0_view_0
- **Parameter $d_i$**: ‘Demand’ column in table_id: file_0_view_0
- **Parameter $I_i$**: ‘Initial Inventory’ column in table_id: file_0_view_0

No additional constraints or scenario-specific rules are imposed beyond those described above.