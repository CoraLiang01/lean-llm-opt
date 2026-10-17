#### Index Sets
- $I$: set of all pizza types (indexed by $i$).

#### Parameters
- $A_i$: revenue per unit of pizza type $i$ (from column "Revenue").
- $d_i$: total demand for pizza type $i$ (from column "Demand").
- $I_i$: initial inventory available for pizza type $i$ (from column "Initial Inventory").

#### Decision Variables
- $x_i$: number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]
   (Equivalently, two constraints per $i$:)
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

2. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Source Table: `file_0_view_0` (from `PizzaSalesDataset.csv`)
- Index Set $I$: all rows in column "Product Name"
- Parameter $A_i$: column "Revenue"
- Parameter $d_i$: column "Demand"
- Parameter $I_i$: column "Initial Inventory"
- All records used: no filters applied; all rows included as returned by CSVQA.