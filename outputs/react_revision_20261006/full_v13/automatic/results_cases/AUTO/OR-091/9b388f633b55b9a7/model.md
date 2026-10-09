## Mathematical Model

**Sets**
- $I$: Set of Operations Research courses (from table_id: file_0_view_0), indexed by $i$

**Parameters** (from table_id: file_0_view_0)
- $c_i$: credits for course $i$
- $p_i$: interest points for course $i$

**Decision Variables**
- $x_i \in \{0,1\}$: $1$ if course $i$ is selected, $0$ otherwise

**Objective**
$$
\max \sum_{i \in I} p_i x_i
$$

**Constraint**
$$
\sum_{i \in I} c_i x_i \leq 20
$$

$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

### Data Mapping

- $I$ = {C22, C23, C24, C25, C26, C27, C28}
- $c_i$ = "credits" column, $p_i$ = "interest_points" column, both from table_id: file_0_view_0, for each $i \in I$
- $x_i$ is the selection variable for course $i$ in $I$