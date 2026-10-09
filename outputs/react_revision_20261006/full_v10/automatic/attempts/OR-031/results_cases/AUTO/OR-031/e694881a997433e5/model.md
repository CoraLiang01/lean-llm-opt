Mathematical Model

Index Sets:
Let $I$ be the set of all dairy products, indexed by $i$.

Parameters:
For each $i \in I$:
- $r_i$: revenue per unit of product $i$ (from column "Revenue")
- $d_i$: demand for product $i$ (from column "Demand")
- $s_i$: initial inventory of product $i$ (from column "Initial Inventory")

Decision Variables:
For each $i \in I$:
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
x_i \leq d_i \quad \forall i \in I
\]
\[
x_i \leq s_i \quad \forall i \in I
\]
\[
x_i \geq 0 \text{ and } x_i \in \mathbb{Z} \quad \forall i \in I
\]

Data Mapping:
- Index set $I$ and all parameters $r_i$, $d_i$, $s_i$ are defined by all rows in table_id file_0_view_0, columns "Full_Product_Name", "Revenue", "Demand", "Initial Inventory" from DairyGoodsSalesDataset.csv.