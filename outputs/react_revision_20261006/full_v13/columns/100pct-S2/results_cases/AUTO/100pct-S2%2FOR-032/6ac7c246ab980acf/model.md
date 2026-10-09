Let $I$ be the set of Operations Research courses in courses_42.csv:
$$
I = \{\text{C22}, \text{C23}, \text{C24}, \text{C25}, \text{C26}, \text{C27}, \text{C28}\}
$$

Define parameters (from table_id: file_0_view_0):
- $c_i$: credits for course $i \in I$
- $p_i$: interest points for course $i \in I$

Define decision variables:
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise

Objective:
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

**Data Mapping (from file_0_view_0):**
- $I$ = set of course_id where discipline = "Operations Research"
- $c_i$ = credits column for course $i$
- $p_i$ = interest_points column for course $i$

**Summary:**  
Choose a subset of Operations Research courses to maximize total interest points, with total credits $\leq 20$. Each course can be selected at most once. All data is mapped directly from the filtered rows of courses_42.csv.