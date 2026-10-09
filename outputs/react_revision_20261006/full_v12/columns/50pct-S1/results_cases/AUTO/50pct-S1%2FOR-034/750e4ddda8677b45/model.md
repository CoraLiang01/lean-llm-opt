Let $I$ be the set of items in value.csv, with parameters $v_i$ (value) and $w_i$ (weight) for each $i \in I$.

**Decision variables:**
- $x_i \in \{0,1\}$: 1 if item $i$ is selected, 0 otherwise, for all $i \in I$

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraint:**
\[
\sum_{i \in I} w_i x_i \leq 15
\]

**Variable domains:**
\[
x_i \in \{0,1\} \quad \forall i \in I
\]

**Data Mapping:**
- $I$: All rows in value.csv, column "item" (table_id: file_0_view_0)
- $v_i$: value.csv, column "value", for item $i$ (table_id: file_0_view_0)
- $w_i$: value.csv, column "weight", for item $i$ (table_id: file_0_view_0)