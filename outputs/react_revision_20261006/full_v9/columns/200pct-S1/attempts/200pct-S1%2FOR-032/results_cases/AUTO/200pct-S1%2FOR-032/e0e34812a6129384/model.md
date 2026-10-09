Mathematical Model

Sets:
- Let $\mathcal{C}$ be the set of Operations Research courses, as indexed by their course_id in table_id file_0_view_0.

Parameters (from table_id file_0_view_0):
- For each $c \in \mathcal{C}$:
    - $a_c$: credits of course $c$ (column: credits)
    - $b_c$: interest points of course $c$ (column: interest_points)

Decision Variables:
- $x_c \in \{0,1\}$ for each $c \in \mathcal{C}$: $x_c = 1$ if course $c$ is selected, $0$ otherwise.

Objective:
$$
\max \sum_{c \in \mathcal{C}} b_c x_c
$$

Constraint:
$$
\sum_{c \in \mathcal{C}} a_c x_c \leq 20
$$

Variable domains:
$$
x_c \in \{0,1\} \quad \forall c \in \mathcal{C}
$$

Data Mapping:
- $\mathcal{C}$: All records in file_0_view_0 where discipline = "Operations Research"
- $a_c$: file_0_view_0.credits for course $c$
- $b_c$: file_0_view_0.interest_points for course $c$