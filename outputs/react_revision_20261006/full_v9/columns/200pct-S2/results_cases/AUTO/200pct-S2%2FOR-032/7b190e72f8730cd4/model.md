## Mathematical Model

**Sets:**
- Let $\mathcal{C}$ be the set of Operations Research courses in courses_42.csv:
  $$
  \mathcal{C} = \{\text{C22},\ \text{C23},\ \text{C24},\ \text{C25},\ \text{C26},\ \text{C27},\ \text{C28}\}
  $$

**Parameters (from table_id: file_0_view_0):**
- For each $c \in \mathcal{C}$:
  - $a_c$: number of credits for course $c$ (column: credits)
  - $p_c$: interest points for course $c$ (column: interest_points)

**Decision Variables:**
- $x_c \in \{0,1\}$: $x_c = 1$ if course $c$ is selected, $0$ otherwise, for all $c \in \mathcal{C}$

**Objective:**
$$
\max \sum_{c \in \mathcal{C}} p_c\, x_c
$$

**Constraint:**
$$
\sum_{c \in \mathcal{C}} a_c\, x_c \leq 20
$$

**Variable Domains:**
$$
x_c \in \{0,1\} \qquad \forall c \in \mathcal{C}
$$

---

### Data Mapping

- $\mathcal{C}$: All records in courses_42.csv (table_id: file_0_view_0) with discipline = "Operations Research"
- $a_c$: column "credits" in file_0_view_0, for each $c$
- $p_c$: column "interest_points" in file_0_view_0, for each $c$
- $x_c$: binary variable for each $c \in \mathcal{C}$

---

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, subject to a total credit cap of 20, using the exact credits and interest points from the specified CSV.