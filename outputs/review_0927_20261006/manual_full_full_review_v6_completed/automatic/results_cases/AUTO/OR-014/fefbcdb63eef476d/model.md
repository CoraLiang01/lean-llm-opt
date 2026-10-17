#### Index Sets

- $I$: set of all pizza types (indexed by $i$).

#### Parameters

- $A_i$: revenue per unit of pizza type $i$ (from column "Revenue").
- $d_i$: total demand for pizza type $i$ (from column "Demand").
- $I_i$: initial inventory for pizza type $i$ (from column "Initial Inventory").

#### Decision Variables

- $x_i$: number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory and Demand Fulfillment:
   $$
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   $$

   (Equivalently, two separate constraints:)
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

2. Integer Nonnegativity:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Source table: PizzaSalesDataset.csv, table_id: file_0_view_0
- Columns used:
    - "Product Name" $\rightarrow$ index set $I$
    - "Revenue" $\rightarrow$ parameter $A_i$
    - "Demand" $\rightarrow$ parameter $d_i$
    - "Initial Inventory" $\rightarrow$ parameter $I_i$
- All records in file_0_view_0 are included (no filter applied).