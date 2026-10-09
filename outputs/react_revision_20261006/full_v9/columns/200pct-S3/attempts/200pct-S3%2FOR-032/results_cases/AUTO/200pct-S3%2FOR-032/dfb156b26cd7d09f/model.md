## Mathematical Model

**Sets:**
- Let $\mathcal{C}$ be the set of Operations Research courses, indexed by $i$.
  - $\mathcal{C} = \{\text{C22}, \text{C23}, \text{C24}, \text{C25}, \text{C26}, \text{C27}, \text{C28}\}$

**Parameters (from Data Mapping):**
- $a_i$: credits for course $i$ (from column "credits", table_id: file_0_view_0)
- $b_i$: interest points for course $i$ (from column "interest_points", table_id: file_0_view_0)

**Decision Variables:**
- $x_i \in \{0,1\}$: $x_i = 1$ if course $i$ is selected, $0$ otherwise, for all $i \in \mathcal{C}$

**Objective:**
$$
\max \sum_{i \in \mathcal{C}} b_i x_i
$$

**Constraint:**
$$
\sum_{i \in \mathcal{C}} a_i x_i \leq 20
$$

$$
x_i \in \{0,1\} \quad \forall i \in \mathcal{C}
$$

---

### Data Mapping

- Set $\mathcal{C}$: All rows in courses_42.csv (table_id: file_0_view_0) where discipline = "Operations Research"
- $a_i$: "credits" column, table_id: file_0_view_0, for each $i \in \mathcal{C}$
- $b_i$: "interest_points" column, table_id: file_0_view_0, for each $i \in \mathcal{C}$
- $x_i$: binary variable for each $i \in \mathcal{C}$

---

**Summary:**  
Select a subset of Operations Research courses to maximize total interest points, with total credits $\leq 20$. Each course can be selected at most once. All parameters and sets are mapped directly from the filtered rows of courses_42.csv.