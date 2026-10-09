## Mathematical Model

**Sets**
- $C$: Set of Operations Research courses (from table_id: file_0_view_0), indexed by $i$

**Parameters** (from table_id: file_0_view_0)
- $a_i$: credits of course $i$
- $b_i$: interest points of course $i$

**Decision Variables**
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise

**Objective**
$$
\max \sum_{i \in C} b_i x_i
$$

**Constraint: Total credits**
$$
\sum_{i \in C} a_i x_i \leq 20
$$

**Constraint: Binary selection**
$$
x_i \in \{0,1\} \quad \forall i \in C
$$

---

### Data Mapping

- $C$ = {C22, C23, C24, C25, C26, C27, C28}
- $a_i$ = "credits" column, $b_i$ = "interest_points" column, both from table_id: file_0_view_0, for each $i \in C$.
- $x_i$ is the decision to select course $i$.

**All data and indices are as returned in the Observation from courses_42.csv, filtered to discipline = "Operations Research".**