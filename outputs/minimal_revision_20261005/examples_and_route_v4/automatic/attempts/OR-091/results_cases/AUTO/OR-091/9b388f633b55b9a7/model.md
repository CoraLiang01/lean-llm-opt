**Abstract Mathematical Model**

**Index Sets**
- $C$: Set of Operations Research courses, indexed by $c$ (from all rows in file_0_view_0, column course_id).

**Parameters**
- $a_c$: Credits required for course $c$ (from file_0_view_0, column credits).
- $p_c$: Interest points for course $c$ (from file_0_view_0, column interest_points).
- $B$: Maximum total credits allowed ($B = 20$, from query).

**Decision Variables**
- $x_c \in \{0,1\}$: $1$ if course $c$ is selected, $0$ otherwise.

**Objective**
\[
\max \sum_{c \in C} p_c x_c
\]

**Constraints**
\[
\sum_{c \in C} a_c x_c \leq B
\]
\[
x_c \in \{0,1\} \quad \forall c \in C
\]

---

**Data Mapping**

- $C$: All course_id in file_0_view_0 where discipline = "Operations Research"
- $a_c$: file_0_view_0, column credits, keyed by course_id
- $p_c$: file_0_view_0, column interest_points, keyed by course_id
- $B$: 20 (from query)
- $x_c$: Binary variable for each $c \in C$