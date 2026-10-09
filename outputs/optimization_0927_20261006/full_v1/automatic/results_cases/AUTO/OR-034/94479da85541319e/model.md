#### Abstract Mathematical Model

**Index Sets:**

- $I$: set of all baked goods (indexed by $i$).

**Parameters:**

- $A_i$: revenue per unit of baked good $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: total demand for baked good $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: initial inventory for baked good $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

**Decision Variables:**

- $x_i$: quantity of baked good $i$ to fulfill, $\forall i \in I$.

**Objective:**

$$
\max \sum_{i \in I} A_i x_i
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
3. **Nonnegativity and Integrality:**
   $$
   x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
   $$

---

**Data Mapping:**

- All parameters $A_i$, $d_i$, $I_i$ are mapped from table_id: file_0_view_0 (Frenchbakerydailysales.csv), columns: ‘Product Name’, ‘Revenue’, ‘Demand’, ‘Initial Inventory’. Each $i \in I$ corresponds to a unique ‘Product Name’ row in this table.