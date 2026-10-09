## Mathematical Model

**Sets:**
- $\mathcal{C}$: Set of Operations Research courses (from table_id: file_0_view_0), indexed by $i$.

**Parameters (from Data Mapping):**
- $a_i$: Number of credits for course $i$ (column: credits, table_id: file_0_view_0)
- $b_i$: Interest points for course $i$ (column: interest_points, table_id: file_0_view_0)

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

$$
x_i \in \{0,1\} \quad \forall i \in \mathcal{C}
$$

---

### Data Mapping

- $\mathcal{C}$: All rows in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0, column: discipline)
- $a_i$: credits (table_id: file_0_view_0, column: credits)
- $b_i$: interest_points (table_id: file_0_view_0, column: interest_points)
- $x_i$: binary selection variable for each course $i$ in $\mathcal{C}$

**Courses in $\mathcal{C}$ (with course_id):**
- C22: Operations Research: Linear Programming
- C23: Integer Programming
- C24: Stochastic Processes
- C25: Simulation Modeling
- C26: Network Flows
- C27: Queueing Theory
- C28: Revenue Management

**All parameters are mapped directly from the corresponding columns in table_id: file_0_view_0.**