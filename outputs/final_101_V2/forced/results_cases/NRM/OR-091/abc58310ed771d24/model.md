#### Abstract Mathematical Model

**Index Set:**

- $C$ : set of all Operations Research courses (from courses_42.csv, where Discipline = "Operations Research").

**Parameters:**

- $a_c$ : number of credits for course $c \in C$ (from column "credits").
- $p_c$ : interest points for course $c \in C$ (from column "interest_points").

**Decision Variables:**

- $x_c \in \{0,1\}$ : $x_c = 1$ if course $c$ is selected, $0$ otherwise, for all $c \in C$.

**Objective:**

$$
\max \sum_{c \in C} p_c x_c
$$

**Constraints:**

1. **Credit Limit:**
   $$
   \sum_{c \in C} a_c x_c \leq 20
   $$

2. **Binary Selection:**
   $$
   x_c \in \{0,1\} \quad \forall c \in C
   $$

---

#### Data Mapping

- **Table:** courses_42.csv
- **Index Set:** $C$ is all rows where column "discipline" = "Operations Research" (table_id: file_0_view_0).
- **Parameter $a_c$:** column "credits" in table_id: file_0_view_0.
- **Parameter $p_c$:** column "interest_points" in table_id: file_0_view_0.