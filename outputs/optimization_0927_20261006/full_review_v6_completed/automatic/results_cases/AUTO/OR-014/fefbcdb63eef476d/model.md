#### Index Sets

- $I$: set of all pizza types (from column "Product Name" in table_id file_0_view_0).

#### Parameters

- $A_i$: revenue per unit of pizza type $i$ (from column "Revenue" in table_id file_0_view_0).
- $d_i$: total demand for pizza type $i$ (from column "Demand" in table_id file_0_view_0).
- $I_i$: initial inventory available for pizza type $i$ (from column "Initial Inventory" in table_id file_0_view_0).

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

   (Implemented as two separate constraints:)
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

2. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Source table: file_0_view_0 (PizzaSalesDataset.csv)
- Index set $I$: all rows, column "Product Name"
- Parameter $A_i$: column "Revenue"
- Parameter $d_i$: column "Demand"
- Parameter $I_i$: column "Initial Inventory"
- All 100 records returned directly by CSVQA; no additional filters applied.