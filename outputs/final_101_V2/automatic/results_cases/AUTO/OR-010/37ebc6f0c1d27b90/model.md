#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all products, indexed by $i$.

**Parameters:**

- $A_i$ : Revenue per unit for product $i$.  
  (Data Mapping: table_id = "file_0_view_0", column = "Revenue")
- $d_i$ : Expected demand for product $i$ during the sales cycle.  
  (Data Mapping: table_id = "file_0_view_0", column = "Demand")
- $I_i$ : Initial inventory for product $i$.  
  (Data Mapping: table_id = "file_0_view_0", column = "Initial Inventory")

**Decision Variables:**

- $x_i$ : Number of orders fulfilled for product $i$  
  (Domain: $x_i \in \mathbb{Z}_+, \forall i \in I$)

**Objective Function:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraints:**  
   $x_i \leq I_i, \quad \forall i \in I$

2. **Demand Constraints:**  
   $x_i \leq d_i, \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

**Data Mapping:**

- All parameters are mapped from table_id = "file_0_view_0" (MobileSalesDataset.csv), using columns:
    - "Product Name" (for index set $I$)
    - "Revenue" (for $A_i$)
    - "Demand" (for $d_i$)
    - "Initial Inventory" (for $I_i$)