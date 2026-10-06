#### Abstract Mathematical Model

**Index Sets:**

- $I$ : set of all ‘TABLET’ smartphone models (indexed by $i$)

**Parameters:**

- $A_i$ : revenue per unit for model $i$ (from column ‘Revenue’)
- $d_i$ : total demand for model $i$ (from column ‘Demand’)
- $I_i$ : initial inventory for model $i$ (from column ‘Initial Inventory’)

**Decision Variables:**

- $x_i$ : number of units of model $i$ to fulfill, $\forall i \in I$

**Objective:**

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

- **Table ID:** file_0_view_0 (from SmartphoneRetailOutletSalesData.csv)
- **Index Set $I$:** All rows where ‘Product Name’ has prefix ‘TABLET’
- **Parameter $A_i$:** column ‘Revenue’
- **Parameter $d_i$:** column ‘Demand’
- **Parameter $I_i$:** column ‘Initial Inventory’