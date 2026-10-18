#### Index Sets
- $I$: set of bread types (from products.csv), indexed by $i$

#### Parameters
- $v_i$: expected profit per unit of bread type $i$ (from products.csv, column "Value")
- $w_i$: storage weight per unit of bread type $i$ (from products.csv, column "Weight")
- $C$: total storage capacity (from capacity.csv, column "Capacity")

#### Decision Variables
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of bread type $i$ to order each day

#### Objective
\[
\max \sum_{i \in I} v_i x_i
\]

#### Constraints
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- Table: file_1_view_0 (products.csv)
  - $I$: all rows, column "ProductName"
  - $v_i$: column "Value"
  - $w_i$: column "Weight"
- Table: file_0_view_0 (capacity.csv)
  - $C$: column "Capacity" (single value)