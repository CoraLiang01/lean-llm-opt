## Mathematical Model

**Sets:**
- Let $\mathcal{C}$ be the set of Operations Research courses in courses_42.csv, with elements indexed by $i$ and identified by course_id.

**Parameters (from Data Mapping):**
- $a_i$: credits for course $i$ (from column "credits", table_id: file_0_view_0)
- $b_i$: interest points for course $i$ (from column "interest_points", table_id: file_0_view_0)

**Decision Variables:**
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise, for all $i \in \mathcal{C}$

**Objective:**
$$
\max \sum_{i \in \mathcal{C}} b_i x_i
$$

**Constraint:**
$$
\sum_{i \in \mathcal{C}} a_i x_i \leq 20
$$

**Variable Domains:**
$$
x_i \in \{0,1\} \qquad \forall i \in \mathcal{C}
$$

---

### Data Mapping

- $\mathcal{C}$: All records in courses_42.csv (table_id: file_0_view_0) with discipline = "Operations Research"
- $a_i$: "credits" column, table_id: file_0_view_0, for each $i$
- $b_i$: "interest_points" column, table_id: file_0_view_0, for each $i$
- $x_i$: binary variable for each $i$ in $\mathcal{C}$

**Course IDs in $\mathcal{C}$:** C22, C23, C24, C25, C26, C27, C28

**All parameters and sets are defined directly from the filtered rows of courses_42.csv as described above.**