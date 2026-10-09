## Mathematical Model

**Sets**
- Let $\mathcal{C}$ be the set of Operations Research courses in courses_42.csv:
  $$
  \mathcal{C} = \{\text{C22},\ \text{C23},\ \text{C24},\ \text{C25},\ \text{C26},\ \text{C27},\ \text{C28}\}
  $$

**Parameters** (from table_id: file_0_view_0)
- $a_c$: credits for course $c \in \mathcal{C}$ (column: credits)
- $b_c$: interest points for course $c \in \mathcal{C}$ (column: interest_points)

**Decision Variables**
- $x_c \in \{0,1\}$: $1$ if course $c$ is selected, $0$ otherwise, for all $c \in \mathcal{C}$

**Objective**
$$
\max \sum_{c \in \mathcal{C}} b_c\, x_c
$$

**Constraint**
$$
\sum_{c \in \mathcal{C}} a_c\, x_c \leq 20
$$

$$
x_c \in \{0,1\} \quad \forall c \in \mathcal{C}
$$

---

### Data Mapping

- $\mathcal{C}$: All rows in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0)
- $a_c$: column "credits" in file_0_view_0, for each $c$
- $b_c$: column "interest_points" in file_0_view_0, for each $c$
- $x_c$: binary variable for each $c \in \mathcal{C}$

**All data and indices are as returned in the Observation.**