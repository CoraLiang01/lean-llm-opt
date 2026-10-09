## Mathematical Model

**Sets**
- Let $\mathcal{C}$ be the set of Operations Research courses in courses_42.csv, with elements indexed by $i$ and identified by course_id.

**Parameters** (from table_id: file_0_view_0)
- $c_i$: credits of course $i$ (column: credits)
- $p_i$: interest points of course $i$ (column: interest_points)

**Decision Variables**
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise, for all $i \in \mathcal{C}$

**Objective**
$$
\max \sum_{i \in \mathcal{C}} p_i x_i
$$

**Constraint**
$$
\sum_{i \in \mathcal{C}} c_i x_i \leq 20
$$

$$
x_i \in \{0,1\} \quad \forall i \in \mathcal{C}
$$

---

### Data Mapping

- $\mathcal{C}$: All records in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0, 7 rows, course_id: C22–C28)
- $c_i$: column "credits" in file_0_view_0, for each $i$
- $p_i$: column "interest_points" in file_0_view_0, for each $i$
- $x_i$: binary variable for each $i$ in $\mathcal{C}$

**Constraint and objective use only these 7 Operations Research courses.**