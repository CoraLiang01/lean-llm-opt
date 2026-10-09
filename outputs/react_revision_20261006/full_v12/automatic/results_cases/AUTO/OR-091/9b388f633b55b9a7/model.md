## Mathematical Model

Let $I$ be the set of Operations Research courses:
$$
I = \{\text{C22},\ \text{C23},\ \text{C24},\ \text{C25},\ \text{C26},\ \text{C27},\ \text{C28}\}
$$

For each $i \in I$:
- Let $c_i$ = credits of course $i$ (from column "credits", table_id: file_0_view_0)
- Let $p_i$ = interest points of course $i$ (from column "interest_points", table_id: file_0_view_0)
- Let $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise

**Objective:**
$$
\max \sum_{i \in I} p_i x_i
$$

**Subject to:**
$$
\sum_{i \in I} c_i x_i \leq 20
$$

$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

### Data Mapping

- $I$ (course set): All rows in courses_42.csv (table_id: file_0_view_0) with discipline = "Operations Research"
- $c_i$: column "credits" for each $i \in I$ (table_id: file_0_view_0)
- $p_i$: column "interest_points" for each $i \in I$ (table_id: file_0_view_0)
- $x_i$: binary decision variable for each $i \in I$

---

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, with total credits $\leq 20$. Each course can be selected at most once. All data is mapped directly from the filtered rows of courses_42.csv.