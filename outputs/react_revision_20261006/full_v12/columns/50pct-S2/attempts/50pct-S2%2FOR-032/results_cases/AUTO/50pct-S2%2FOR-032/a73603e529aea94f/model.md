## Mathematical Model

Let $I$ be the set of Operations Research courses in courses_42.csv:
$$
I = \{\text{C22},\ \text{C23},\ \text{C24},\ \text{C25},\ \text{C26},\ \text{C27},\ \text{C28}\}
$$

Let $c_i$ = credits for course $i \in I$ (from column "credits", table_id: file_0_view_0).

Let $p_i$ = interest points for course $i \in I$ (from column "interest_points", table_id: file_0_view_0).

Define decision variables:
$$
x_i \in \{0,1\} \quad \forall i \in I
$$
where $x_i = 1$ if course $i$ is selected, $0$ otherwise.

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

- $I$: All rows in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0, column "course_id")
- $c_i$: column "credits" for $i \in I$ (table_id: file_0_view_0)
- $p_i$: column "interest_points" for $i \in I$ (table_id: file_0_view_0)
- $x_i$: binary variable for each $i \in I$

---

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, subject to a total credit cap of 20. Each course can be selected at most once. All data is mapped directly from the filtered rows of courses_42.csv.