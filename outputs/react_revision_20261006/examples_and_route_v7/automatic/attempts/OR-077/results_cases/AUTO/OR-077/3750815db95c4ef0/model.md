Mathematical Model

Sets:
- $I$: set of items, with each item identified by the column "item" in value.csv.

Parameters:
- $v_i$: value of item $i$, from column "value" in value.csv.
- $w_i$: weight of item $i$, from column "weight" in value.csv.
- $W$: total weight capacity of the merchandise counter, $W = 15$ (from user description).

Decision Variables:
- $x_i \in \{0,1\}$: $x_i = 1$ if item $i$ is selected for display, $0$ otherwise.

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq W
\]
\[
x_i \in \{0,1\} \quad \forall i \in I
\]

Data Mapping

- $I$: All records in value.csv, column "item", table_id file_0_view_0.
- $v_i$: value.csv, column "value", table_id file_0_view_0, keyed by "item".
- $w_i$: value.csv, column "weight", table_id file_0_view_0, keyed by "item".
- $W$: 15 (from user description).