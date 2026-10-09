## Mathematical Model

**Sets:**
- $P$: set of all projects, indexed by $i$; $|P| = 110$ (from Project ID in project.csv).

**Parameters (from project.csv, table_id: file_0_view_0):**
- $c_i$: Capital required for project $i$ ("Capital (k$)")
- $v_i$: Expected NPV for project $i$ ("NPV (k$)")

**Decision Variables:**
- $x_i \in \{0,1\}$: 1 if project $i$ is selected, 0 otherwise

**Objective:**
$$
\max \sum_{i \in P} v_i x_i
$$

**Subject to:**

1. **Budget Constraint:**
   $$
   \sum_{i \in P} c_i x_i \leq 1000
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
   x_i \in \{0,1\} \quad \forall i \in P
   $$

---

### Data Mapping

- $P$: All "Project ID" values in project.csv (table_id: file_0_view_0)
- $c_i$: "Capital (k$)" for project $i$ in project.csv (table_id: file_0_view_0)
- $v_i$: "NPV (k$)" for project $i$ in project.csv (table_id: file_0_view_0)
- Constraints 2–4: Use Project IDs as given in the question and project.csv

---

**Summary:**  
Select a subset of projects to maximize total NPV, subject to the total capital budget, mutual exclusivity of Projects 4 & 7, pre-requisite (6 requires 1), contingent (10 requires 5), and binary selection variables. All data is mapped directly from project.csv (table_id: file_0_view_0).