## Mathematical Model

**Sets**
- Let $\mathcal{C}$ be the set of Operations Research courses in courses_42.csv, with elements indexed by $i$ and table_id = file_0_view_0.

**Parameters** (from table_id = file_0_view_0)
- $c_i$: credits of course $i$ (column: credits)
- $p_i$: interest points of course $i$ (column: interest_points)

**Decision Variables**
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise

**Objective**
$$
\max \sum_{i \in \mathcal{C}} p_i x_i
$$

**Constraint: Total credits**
$$
\sum_{i \in \mathcal{C}} c_i x_i \leq 20
$$

**Constraint: Binary selection**
$$
x_i \in \{0,1\} \qquad \forall i \in \mathcal{C}
$$

---

### Data Mapping

- $\mathcal{C}$: All rows in courses_42.csv with discipline = "Operations Research" (table_id = file_0_view_0, 7 rows: C22–C28)
- $c_i$: column "credits" in file_0_view_0
- $p_i$: column "interest_points" in file_0_view_0
- $x_i$: decision variable for each $i$ in $\mathcal{C}$

---

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, with total credits $\leq 20$, using binary selection variables for each eligible course. All parameters are mapped directly from the specified columns and rows in courses_42.csv.