---

**Sets:**

- $I$ : Set of all products, indexed by $i$.

**Parameters:**

- $A_i$ : Revenue per unit for product $i$.  
- $d_i$ : Demand for product $i$.  
- $I_i$ : Initial Inventory for product $i$.

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill, $\forall i \in I$.

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

- Table: `file_0_view_0` (from `RetailSalesDataset.csv`)
    - Product Name: Index set $I$
    - Revenue: Parameter $A_i$
    - Demand: Parameter $d_i$
    - Initial Inventory: Parameter $I_i$

---