Let $I$ be the set of items, with $I = \{\text{1}, \text{2}, \ldots, \text{140}\}$ from value.csv.

Parameters (from value.csv, table_id: file_0_view_0):
- $v_i$: value of item $i$, column "value"
- $w_i$: weight of item $i$, column "weight"
- $W = 15$: total weight capacity

Decision variables:
- $x_i \in \{0,1\}$ for $i \in I$: 1 if item $i$ is selected, 0 otherwise

Objective:
$\max \sum_{i \in I} v_i x_i$

Subject to:
$\sum_{i \in I} w_i x_i \leq W$

$x_i \in \{0,1\}$ for all $i \in I$

Data Mapping:
- $I$: all "item" entries in value.csv (table_id: file_0_view_0)
- $v_i$: "value" column, value.csv (table_id: file_0_view_0)
- $w_i$: "weight" column, value.csv (table_id: file_0_view_0)
- $W$: merchandise counter capacity, from question

This is a 0-1 knapsack problem with 140 items, maximizing total value under a weight limit.