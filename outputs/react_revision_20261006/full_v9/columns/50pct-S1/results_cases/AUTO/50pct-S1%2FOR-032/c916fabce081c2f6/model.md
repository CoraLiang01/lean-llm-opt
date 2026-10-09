## Mathematical Model

**Sets**  
Let $\mathcal{C}$ be the set of Operations Research courses, indexed by $i$.  
From Data Mapping (table_id: file_0_view_0):  
$\mathcal{C} = \{\text{C22}, \text{C23}, \text{C24}, \text{C25}, \text{C26}, \text{C27}, \text{C28}\}$

**Parameters**  
For each $i \in \mathcal{C}$:
- $a_i$: credits of course $i$ (column: credits, table_id: file_0_view_0)
- $b_i$: interest points of course $i$ (column: interest_points, table_id: file_0_view_0)

**Decision Variables**  
For each $i \in \mathcal{C}$:
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise

**Objective**  
Maximize total interest points:
$$
\max \sum_{i \in \mathcal{C}} b_i x_i
$$

**Constraint**  
Total credits no more than 20:
$$
\sum_{i \in \mathcal{C}} a_i x_i \leq 20
$$

**Variable Domains**
$$
x_i \in \{0,1\} \quad \forall i \in \mathcal{C}
$$

---

### Data Mapping

- Set $\mathcal{C}$: All records in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0, column: course_id)
- Parameter $a_i$: credits (table_id: file_0_view_0, column: credits)
- Parameter $b_i$: interest_points (table_id: file_0_view_0, column: interest_points)
- Variable $x_i$: selection indicator for course $i$ in $\mathcal{C}$

---

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, with total credits $\leq 20$, using the exact credits and interest points from courses_42.csv, and binary selection variables for each eligible course.