Let $I$ be the set of Operations Research courses in courses_42.csv:
$$
I = \{\text{C22}, \text{C23}, \text{C24}, \text{C25}, \text{C26}, \text{C27}, \text{C28}\}
$$

Define parameters for each $i \in I$:
- $c_i$: credits of course $i$ (from column "credits", table_id: file_0_view_0)
- $p_i$: interest points of course $i$ (from column "interest_points", table_id: file_0_view_0)

Decision variables:
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise

Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to:
$$
\sum_{i \in I} c_i x_i \leq 20
$$

$$
x_i \in \{0,1\} \quad \forall i \in I
$$

**Data Mapping:**
- $I$: All rows in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0)
- $c_i$: "credits" column, table_id: file_0_view_0, for each $i \in I$
- $p_i$: "interest_points" column, table_id: file_0_view_0, for each $i \in I$