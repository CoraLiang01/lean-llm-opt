Let $I$ be the set of Operations Research courses in courses_42.csv:
$$
I = \{\text{C22},\ \text{C23},\ \text{C24},\ \text{C25},\ \text{C26},\ \text{C27},\ \text{C28}\}
$$

Define for each $i \in I$:
- $c_i$: credits of course $i$ (from column "credits", table_id: file_0_view_0)
- $p_i$: interest points of course $i$ (from column "interest_points", table_id: file_0_view_0)
- $x_i \in \{0,1\}$: 1 if course $i$ is selected, 0 otherwise

**Objective:**
$$
\max \sum_{i \in I} p_i x_i
$$

**Constraint:**
$$
\sum_{i \in I} c_i x_i \leq 20
$$

**Variable domains:**
$$
x_i \in \{0,1\} \qquad \forall i \in I
$$

**Data Mapping:**
- $I$: All rows in courses_42.csv (table_id: file_0_view_0) with discipline = "Operations Research"
- $c_i$: "credits" column, table_id: file_0_view_0, for $i \in I$
- $p_i$: "interest_points" column, table_id: file_0_view_0, for $i \in I$
- $x_i$: binary decision variable for each $i \in I$