#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all products (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit of product $i$.  
  (Data mapping: table_id = file_0_view_0, column = Revenue)
- $d_i$ : Deterministic demand for product $i$ over the sales horizon.  
  (Data mapping: table_id = file_0_view_0, column = Demand)
- $I_i$ : Initial inventory available for product $i$.  
  (Data mapping: table_id = file_0_view_0, column = Initial Inventory)

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill for customer purchases.  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:**  
   $x_i \leq I_i, \quad \forall i \in I$

2. **Demand Constraint:**  
   $x_i \leq d_i, \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

**Data Mapping:**  
All parameters ($A_i$, $d_i$, $I_i$) are mapped from table_id = file_0_view_0 (file: OnlineSalesDataset.csv), columns:  
- Revenue  
- Demand  
- Initial Inventory  
for all products (Product Name).