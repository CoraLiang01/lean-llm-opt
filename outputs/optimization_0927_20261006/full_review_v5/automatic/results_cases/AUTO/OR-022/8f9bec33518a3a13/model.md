#### Index Sets

- $I$: Set of all products classified under ‘27in’.

#### Parameters

- $A_i$: Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory constraint:
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: Salesorders.csv (table_id: file_0_view_0)
    - Filter: Product Name prefix = ‘27in’
    - Columns used:
        - ‘Product Name’ (index set $I$)
        - ‘Revenue’ (parameter $A_i$)
        - ‘Demand’ (parameter $d_i$)
        - ‘Initial Inventory’ (parameter $I_i$)