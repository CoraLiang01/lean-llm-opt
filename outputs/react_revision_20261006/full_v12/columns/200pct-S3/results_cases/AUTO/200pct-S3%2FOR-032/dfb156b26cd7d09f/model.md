## Mathematical Model

Let $I$ be the set of Operations Research courses:
$$
I = \{\text{C22}, \text{C23}, \text{C24}, \text{C25}, \text{C26}, \text{C27}, \text{C28}\}
$$

Let $x_i \in \{0,1\}$ indicate whether course $i \in I$ is selected.

Let $c_i$ = credits of course $i$ (from column "credits", table_id: file_0_view_0).

Let $p_i$ = interest points of course $i$ (from column "interest_points", table_id: file_0_view_0).

**Objective:**
$$
\max \sum_{i \in I} p_i x_i
$$

**Constraint:**
$$
\sum_{i \in I} c_i x_i \leq 20
$$

**Variable domains:**
$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

### Data Mapping

- $I$ = set of course_id where discipline = "Operations Research" (from file_0_view_0)
- $c_i$ = "credits" column for course $i$ (file_0_view_0)
- $p_i$ = "interest_points" column for course $i$ (file_0_view_0)
- $x_i$ = binary decision: select course $i$ or not

---

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, with total credits $\leq 20$. Data for $I$, $c_i$, $p_i$ is from table_id: file_0_view_0, columns: course_id, credits, interest_points.