## Mathematical Model

Sets:
- $I$: set of items, $I = \{\text{1}, \text{2}, \ldots, \text{140}\}$ (from column "item" in value.csv)

Parameters (from value.csv, table_id: file_0_view_0):
- $v_i$: value of item $i$ ("value")
- $w_i$: weight of item $i$ ("weight")
- $W = 15$: total weight capacity

Decision variables:
- $x_i \in \{0,1\}$: 1 if item $i$ is selected, 0 otherwise

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Constraint:
\[
\sum_{i \in I} w_i x_i \leq W
\]
\[
x_i \in \{0,1\} \quad \forall i \in I
\]

## Data Mapping

- $I$: All rows in value.csv, column "item", table_id: file_0_view_0
- $v_i$: value.csv, column "value", table_id: file_0_view_0
- $w_i$: value.csv, column "weight", table_id: file_0_view_0
- $W$: 15 (from user description)

## Complete Model

\[
\begin{align*}
\max_{x_i} \quad & \sum_{i \in I} v_i x_i \\
\text{s.t.} \quad & \sum_{i \in I} w_i x_i \leq 15 \\
& x_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]