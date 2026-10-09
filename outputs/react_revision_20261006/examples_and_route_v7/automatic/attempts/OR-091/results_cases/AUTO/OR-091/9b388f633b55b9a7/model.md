Mathematical Model

Index Sets:
- Let $I$ be the set of Operations Research courses, indexed by $i$, with course IDs from column course_id in table_id file_0_view_0.

Parameters:
- $a_i$: number of credits for course $i$ (from column credits, table_id file_0_view_0)
- $p_i$: interest points for course $i$ (from column interest_points, table_id file_0_view_0)

Decision Variables:
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise

Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to:
$$
\sum_{i \in I} a_i x_i \leq 20
$$
$$
x_i \in \{0,1\} \quad \forall i \in I
$$

Data Mapping

- $I$: All records in table_id file_0_view_0, column course_id, where discipline = "Operations Research"
- $a_i$: table_id file_0_view_0, column credits, keyed by course_id
- $p_i$: table_id file_0_view_0, column interest_points, keyed by course_id
- $x_i$: binary variable for each $i \in I$ (course_id in file_0_view_0)