ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $C$: set of Operations Research courses, indexed by $c$ (from courses_42.csv, where discipline = "Operations Research", column course_id).

Parameters:
- $a_c$: number of credits for course $c$ (courses_42.csv, column credits).
- $p_c$: interest points for course $c$ (courses_42.csv, column interest_points).

Decision Variables:
- $x_c \in \{0,1\}$: $x_c = 1$ if course $c$ is selected, $0$ otherwise.

Objective:
$$
\max \sum_{c \in C} p_c x_c
$$

Subject to:
$$
\sum_{c \in C} a_c x_c \leq 20
$$
$$
x_c \in \{0,1\} \quad \forall c \in C
$$

DATA MAPPING

- $C$: All rows in courses_42.csv where discipline = "Operations Research", using column course_id.
- $a_c$: courses_42.csv, column credits, for each $c \in C$.
- $p_c$: courses_42.csv, column interest_points, for each $c \in C$.

Returned records (in source order):

| table_id         | course_id | course_name                              | discipline           | credits | interest_points |
|------------------|-----------|------------------------------------------|----------------------|---------|----------------|
| file_0_view_0    | C22       | Operations Research: Linear Programming  | Operations Research  | 5       | 95             |
| file_0_view_0    | C23       | Integer Programming                      | Operations Research  | 5       | 92             |
| file_0_view_0    | C24       | Stochastic Processes                     | Operations Research  | 4       | 86             |
| file_0_view_0    | C25       | Simulation Modeling                      | Operations Research  | 4       | 82             |
| file_0_view_0    | C26       | Network Flows                            | Operations Research  | 4       | 85             |
| file_0_view_0    | C27       | Queueing Theory                          | Operations Research  | 4       | 80             |
| file_0_view_0    | C28       | Revenue Management                       | Operations Research  | 4       | 88             |