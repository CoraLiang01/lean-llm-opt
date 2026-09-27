#### Abstract Mathematical Model

**Index Sets:**

- $I$ : set of all baked goods (from column ‘Product Name’ in table_id: file_0_view_0)

**Parameters:**

- $A_i$ : revenue per unit of baked good $i$ (from column ‘Revenue’ in table_id: file_0_view_0)
- $d_i$ : total demand for baked good $i$ (from column ‘Demand’ in table_id: file_0_view_0)
- $I_i$ : initial inventory for baked good $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0)

**Decision Variables:**

- $x_i$ : quantity of baked good $i$ to fulfill, $\forall i \in I$; $x_i \in \mathbb{Z}_+$

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

**Data Mapping:**

- Table: file_0_view_0 (Frenchbakerydailysales.csv)
    - Index set $I$: column ‘Product Name’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $I_i$: column ‘Initial Inventory’