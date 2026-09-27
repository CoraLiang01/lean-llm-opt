#### Abstract Mathematical Model

**Index Set:**

- $I$ : set of all products (indexed by $i$)

**Parameters:**

- $A_i$ : revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$ : total demand for product $i$ (from column ‘Demand’)
- $I_i$ : initial inventory for product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**

- $x_i$ : number of units of product $i$ to fulfill, $\forall i \in I$

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Demand fulfillment constraint:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

2. **Inventory constraint:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** file_0_view_0 (from SalesDatainBusinesses.csv)
- **Index Set:** $I$ corresponds to all unique values in column ‘Product Name’
- **Parameter $A_i$:** column ‘Revenue’
- **Parameter $d_i$:** column ‘Demand’
- **Parameter $I_i$:** column ‘Initial Inventory’