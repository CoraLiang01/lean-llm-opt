#### Index Sets

- $C$: set of Operations Research courses (from table_id: file_0_view_0, column: course_id)

#### Parameters

- $a_c$: number of credits for course $c \in C$ (from file_0_view_0, column: credits)
- $p_c$: interest points for course $c \in C$ (from file_0_view_0, column: interest_points)

#### Decision Variables

- $x_c \in \{0,1\}$: $1$ if course $c$ is selected, $0$ otherwise, for all $c \in C$

#### Objective

$$
\max \sum_{c \in C} p_c x_c
$$

#### Constraints

1. Credit limit:
   $$
   \sum_{c \in C} a_c x_c \leq 20
   $$

2. Binary selection:
   $$
   x_c \in \{0,1\} \quad \forall c \in C
   $$

---

#### Data Mapping

- Source: courses_42.csv
- Table ID: file_0_view_0
- Columns used:
    - course_id (index set $C$)
    - credits (parameter $a_c$)
    - interest_points (parameter $p_c$)
    - discipline (filter: "Operations Research")