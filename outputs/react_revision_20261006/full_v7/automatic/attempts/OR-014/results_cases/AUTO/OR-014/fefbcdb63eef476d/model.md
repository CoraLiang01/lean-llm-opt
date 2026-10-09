#### Symbolic Mathematical Model

Let:
- $I$ = set of all pizza types (indexed by $i$), as defined by the "Product Name" column.
- For each $i \in I$:
    - $A_i$ = revenue per unit of pizza type $i$ (from "Revenue")
    - $d_i$ = total demand for pizza type $i$ (from "Demand")
    - $s_i$ = initial inventory for pizza type $i$ (from "Initial Inventory")
    - $x_i$ = number of units of pizza type $i$ to fulfill (decision variable)

The model is:

$$
\begin{align*}
\max_{x_i} \quad & \sum_{i \in I} A_i x_i \\
\text{s.t.} \quad
& 0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
$$

#### Data Mapping

- $I$: All unique values in column "Product Name" from table_id file_0_view_0 (PizzaSalesDataset.csv)
- $A_i$: "Revenue" column, table_id file_0_view_0
- $d_i$: "Demand" column, table_id file_0_view_0
- $s_i$: "Initial Inventory" column, table_id file_0_view_0
- $x_i$: Decision variable for each $i \in I$