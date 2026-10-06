#### Abstract Mathematical Model

**Index Sets:**

- $I$ : set of all products (indexed by $i$), where each $i$ corresponds to a unique value in `Product Name` from table_id `file_0_view_0`.

**Parameters:**

- $A_i$ : revenue per unit of product $i$ (`Revenue`, table_id `file_0_view_0`)
- $d_i$ : expected demand for product $i$ during the sales cycle (`Demand`, table_id `file_0_view_0`)
- $I_i$ : initial inventory of product $i$ (`Initial Inventory`, table_id `file_0_view_0`)

**Decision Variables:**

- $x_i$ : number of orders fulfilled for product $i$, $x_i \in \mathbb{Z}_+, \forall i \in I$

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

- Table: `MobileSalesDataset.csv`, table_id `file_0_view_0`
    - Product index: `Product Name`
    - Revenue parameter: `Revenue`
    - Demand parameter: `Demand`
    - Initial inventory parameter: `Initial Inventory`