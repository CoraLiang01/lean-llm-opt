## Mathematical Model

**Sets**
- Let $\mathcal{C}$ be the set of Operations Research courses in courses_42.csv, with elements indexed by $i$ and identified by course_id.

**Parameters** (from table_id: file_0_view_0)
- $a_i$: credits for course $i$ (column: credits)
- $b_i$: interest points for course $i$ (column: interest_points)

**Decision Variables**
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise, for all $i \in \mathcal{C}$

**Objective**
$$
\max \sum_{i \in \mathcal{C}} b_i x_i
$$

**Constraint**
$$
\sum_{i \in \mathcal{C}} a_i x_i \leq 20
$$

$$
x_i \in \{0,1\} \quad \forall i \in \mathcal{C}
$$

---

### Data Mapping

- $\mathcal{C}$: All rows in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0, column: course_id)
- $a_i$: file_0_view_0, column: credits, for each $i$
- $b_i$: file_0_view_0, column: interest_points, for each $i$
- $x_i$: decision variable for each $i \in \mathcal{C}$

**Constraint and objective use only the Operations Research courses as filtered above.**