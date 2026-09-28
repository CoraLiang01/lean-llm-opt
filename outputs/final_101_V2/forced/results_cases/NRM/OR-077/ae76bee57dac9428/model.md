#### Abstract Mathematical Model

**Index Set:**
- $I$ : set of items (from value.csv, column "item")

**Parameters:**
- $v_i$ : value of item $i$ (from value.csv, column "value")
- $w_i$ : weight of item $i$ (from value.csv, column "weight")
- $W$ : total weight capacity of the merchandise counter ($W = 15$)

**Decision Variables:**
- $x_i \in \{0,1\}$ : $x_i = 1$ if item $i$ is selected for display, $0$ otherwise

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} w_i x_i \leq W
\]
\[
x_i \in \{0,1\} \quad \forall i \in I
\]

---

#### Data Mapping

- Table: value.csv (table_id: file_0_view_0)
    - Index set $I$: column "item"
    - Parameter $v_i$: column "value"
    - Parameter $w_i$: column "weight"