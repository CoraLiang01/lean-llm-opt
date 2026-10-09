## Mathematical Model

**Sets**
- Let $\mathcal{C}$ be the set of Operations Research courses, as listed in table_id: file_0_view_0.

**Parameters** (from table_id: file_0_view_0)
- For each $i \in \mathcal{C}$:
    - $a_i$: number of credits for course $i$ (column: credits)
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

- $\mathcal{C}$: All records in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0)
- $a_i$: column "credits" in file_0_view_0, for each $i$
- $b_i$: column "interest_points" in file_0_view_0, for each $i$
- $x_i$: binary variable for each $i$ in $\mathcal{C}$

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, with total credits $\leq 20$. All data is mapped directly from the specified columns and filtered discipline in courses_42.csv.