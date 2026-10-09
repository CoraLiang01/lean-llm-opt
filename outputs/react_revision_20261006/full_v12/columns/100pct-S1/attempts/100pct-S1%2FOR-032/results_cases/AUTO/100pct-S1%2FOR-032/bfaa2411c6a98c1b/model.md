## Mathematical Model

**Sets**
- $\mathcal{C}$: Set of Operations Research courses (from table_id: file_0_view_0, column: course_id)

**Parameters** (from table_id: file_0_view_0)
- $a_c$: credits of course $c$ (column: credits)
- $b_c$: interest points of course $c$ (column: interest_points)

**Decision Variables**
- $x_c \in \{0,1\}$: $1$ if course $c \in \mathcal{C}$ is selected, $0$ otherwise

**Objective**
\[
\max \sum_{c \in \mathcal{C}} b_c\, x_c
\]

**Constraint: Total credits**
\[
\sum_{c \in \mathcal{C}} a_c\, x_c \leq 20
\]

**Constraint: Binary selection**
\[
x_c \in \{0,1\} \qquad \forall c \in \mathcal{C}
\]

---

### Data Mapping

- $\mathcal{C}$: All records in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0, column: course_id)
- $a_c$: courses_42.csv, column: credits, for each $c \in \mathcal{C}$
- $b_c$: courses_42.csv, column: interest_points, for each $c \in \mathcal{C}$

---

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, with total credits $\leq 20$. Each course can be selected at most once. All data is mapped directly from the filtered rows of courses_42.csv.