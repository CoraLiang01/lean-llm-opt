#### Symbolic Mathematical Model

Let $I$ be the set of all products classified under ‘Books’ in the dataset.

Parameters:
- $A_i$: Revenue per unit of product $i \in I$ (from column "Revenue")
- $d_i$: Demand for product $i \in I$ (from column "Demand")
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory")

Decision variables:
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Demand and inventory limits:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
$$

#### Data Mapping

- $I$: All rows in table_id file_0_view_0 where Product_Name has prefix "Books"
- $A_i$: file_0_view_0, column "Revenue"
- $d_i$: file_0_view_0, column "Demand"
- $s_i$: file_0_view_0, column "Initial Inventory"
- $x_i$: Decision variable for each $i \in I$