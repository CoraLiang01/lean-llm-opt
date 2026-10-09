## Mathematical Model

**Sets:**
- Let $\mathcal{C}$ be the set of Operations Research courses in courses_42.csv, with elements indexed by $i$ and identifiers $\texttt{course\_id}_i$.

**Parameters (from table_id = file_0_view_0):**
- $a_i$: credits of course $i$ (column: credits)
- $b_i$: interest points of course $i$ (column: interest_points)

**Decision Variables:**
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise, for all $i \in \mathcal{C}$

**Objective:**
\[
\max \sum_{i \in \mathcal{C}} b_i x_i
\]

**Constraint:**
\[
\sum_{i \in \mathcal{C}} a_i x_i \leq 20
\]

**Variable Domains:**
\[
x_i \in \{0,1\} \qquad \forall i \in \mathcal{C}
\]

---

**Data Mapping:**
- $\mathcal{C}$: All records in courses_42.csv with discipline = "Operations Research" (table_id = file_0_view_0, 7 rows, course_id: C22–C28)
- $a_i$: credits (column: credits, file_0_view_0)
- $b_i$: interest_points (column: interest_points, file_0_view_0)
- $x_i$: binary selection variable for each course $i$ in $\mathcal{C}$

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, with total credits $\leq 20$. Each course can be selected at most once. All data and indices are mapped directly from the filtered table file_0_view_0.