#### Index Sets
- $I$: set of all clothing products (indexed by $i$).

#### Parameters
- $A_i$: revenue per unit for product $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: deterministic demand for product $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

#### Decision Variables
- $x_i$: number of units of product $i$ to fulfill, integer, $x_i \geq 0$.

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

- Source: file_0_view_0 (Salesofsummerclothes.csv)
    - Product Name: index set $I$
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$
- All 68 records returned, no filters applied.