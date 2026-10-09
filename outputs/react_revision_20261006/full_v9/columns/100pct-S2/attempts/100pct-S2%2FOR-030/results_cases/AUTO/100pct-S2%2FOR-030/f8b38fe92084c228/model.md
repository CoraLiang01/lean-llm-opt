## Mathematical Model

**Sets:**
- $I = \{1,2,\ldots,110\}$: Set of all projects, indexed by $i$.

**Parameters (from Data Mapping):**
- $c_i$: Capital required for project $i$ (k$), from column "Capital (k$)" in table_id=file_0_view_0, Project ID $i$.
- $v_i$: Expected NPV for project $i$ (k$), from column "NPV (k$)" in table_id=file_0_view_0, Project ID $i$.

**Decision Variables:**
- $x_i \in \{0,1\}$: $1$ if project $i$ is selected, $0$ otherwise.

**Objective:**
$$
\max \sum_{i \in I} v_i x_i
$$

**Subject to:**

1. **Budget Constraint:**
   $$
   \sum_{i \in I} c_i x_i \leq 1000
   $$

2. **Mutually Exclusive Constraint (Projects 4 & 7):**
   $$
   x_4 + x_7 \leq 1
   $$

3. **Pre-requisite Constraint (Project 6 requires 1):**
   $$
   x_6 \leq x_1
   $$

4. **Contingent Constraint (Project 10 requires 5):**
   $$
   x_{10} \leq x_5
   $$

5. **Binary Variables:**
   $$
   x_i \in \{0,1\} \quad \forall i \in I
   $$

---

### Data Mapping

- $I$: All "Project ID" values in table_id=file_0_view_0 (project.csv), i.e., $I = \{1,2,\ldots,110\}$.
- $c_i$: "Capital (k$)" for project $i$ in table_id=file_0_view_0.
- $v_i$: "NPV (k$)" for project $i$ in table_id=file_0_view_0.
- Constraints reference project IDs as per the "Project ID" column in table_id=file_0_view_0.

---

**Summary:**  
Select a subset of projects to maximize total NPV, subject to the total capital budget, mutual exclusivity of projects 4 and 7, pre-requisite (6 requires 1), contingent (10 requires 5), and binary selection variables. All data is mapped directly from project.csv as described.