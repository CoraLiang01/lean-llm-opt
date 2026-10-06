#### Index Sets

- $I$: Set of all baked goods, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of baked good $i$.  
  (From column "Revenue" in table_id: file_0_view_0)
- $d_i$: Total demand for baked good $i$.  
  (From column "Demand" in table_id: file_0_view_0)
- $I_i$: Initial inventory for baked good $i$.  
  (From column "Initial Inventory" in table_id: file_0_view_0)

#### Decision Variables

- $x_i$: Quantity of baked good $i$ to fulfill, $x_i \in \mathbb{Z}_+$, for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint:**  
   $x_i \leq I_i \quad \forall i \in I$

2. **Demand Constraint:**  
   $x_i \leq d_i \quad \forall i \in I$

3. **Nonnegativity and Integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- Table: file_0_view_0 (Frenchbakerydailysales.csv)
    - $A_i$: "Revenue"
    - $d_i$: "Demand"
    - $I_i$: "Initial Inventory"
    - $i \in I$: "Product Name"