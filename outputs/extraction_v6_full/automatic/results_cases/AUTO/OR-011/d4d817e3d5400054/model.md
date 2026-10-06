#### Index Sets

- $I$ : Set of all products classified under ‘id999’, indexed by $i$.

#### Parameters

- $r_i$ : Revenue per unit of product $i$.  
- $d_i$ : Demand for product $i$ over the sales horizon.  
- $s_i$ : Initial inventory of product $i$.

#### Decision Variables

- $x_i$ : Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} r_i \cdot x_i
$$

#### Constraints

1. Inventory constraints:
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$
2. Demand constraints:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$ : All rows in table_id file_0_view_0 where column id_number = ‘id999’.
- $r_i$ : file_0_view_0, column Revenue, indexed by id_number.
- $d_i$ : file_0_view_0, column Demand, indexed by id_number.
- $s_i$ : file_0_view_0, column Initial Inventory, indexed by id_number.