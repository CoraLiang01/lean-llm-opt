Let $I$ be the set of Operations Research courses in courses_42.csv:
$$
I = \{\text{C22},\ \text{C23},\ \text{C24},\ \text{C25},\ \text{C26},\ \text{C27},\ \text{C28}\}
$$

Define parameters for each $i \in I$:
- $c_i$: credits of course $i$ (from column "credits", table_id: file_0_view_0)
- $p_i$: interest points of course $i$ (from column "interest_points", table_id: file_0_view_0)

Decision variables:
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

Data Mapping:
- $I$ is the set of course_id where discipline = "Operations Research" in courses_42.csv (table_id: file_0_view_0)
- $c_i$ and $p_i$ are taken from columns "credits" and "interest_points" for each $i \in I$ in table_id: file_0_view_0

This is a 0-1 knapsack problem over the Operations Research courses, maximizing total interest points with a total credit cap of 20.