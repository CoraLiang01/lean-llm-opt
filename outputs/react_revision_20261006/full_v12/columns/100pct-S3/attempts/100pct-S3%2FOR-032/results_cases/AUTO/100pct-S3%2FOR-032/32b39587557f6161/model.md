Let $I = \{\text{C22}, \text{C23}, \text{C24}, \text{C25}, \text{C26}, \text{C27}, \text{C28}\}$ be the set of Operations Research courses, indexed by $i$.

Let $c_i$ = credits for course $i$ (from column "credits", table_id: file_0_view_0).

Let $p_i$ = interest points for course $i$ (from column "interest_points", table_id: file_0_view_0).

Let $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise.

Maximize total interest points:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to:
$$
\sum_{i \in I} c_i x_i \leq 20
$$

$$
x_i \in \{0,1\} \quad \forall i \in I
$$

**Data Mapping:**  
- $I$ = set of course_id where discipline = "Operations Research" (table_id: file_0_view_0, column: "course_id")
- $c_i$ = "credits" (table_id: file_0_view_0, column: "credits")
- $p_i$ = "interest_points" (table_id: file_0_view_0, column: "interest_points")