## Mathematical Model

**Sets:**
- $I = \{1,2,\ldots,110\}$: set of all projects (from "Project ID" in project.csv)

**Parameters (from project.csv, table_id: file_0_view_0):**
- $c_i$: Capital required for project $i$ ("Capital (k$)")
- $v_i$: Expected NPV for project $i$ ("NPV (k$)")

**Decision variables:**
- $x_i \in \{0,1\}$: 1 if project $i$ is selected, 0 otherwise

**Objective:**
$$
\max \sum_{i \in I} v_i x_i
$$

**Subject to:**

1. **Budget constraint:**
$$
\sum_{i \in I} c_i x_i \leq 1000
$$

2. **Mutually exclusive constraint (Projects 4 & 7):**
$$
x_4 + x_7 \leq 1
$$

3. **Pre-requisite constraint (Project 6 requires 1):**
$$
x_6 \leq x_1
$$

4. **Contingent constraint (Project 10 requires 5):**
$$
x_{10} \leq x_5
$$

5. **Binary variables:**
$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

### Data Mapping

- $I$: All "Project ID" in project.csv, table_id: file_0_view_0
- $c_i$: "Capital (k$)" for project $i$, table_id: file_0_view_0
- $v_i$: "NPV (k$)" for project $i$, table_id: file_0_view_0
- Constraints 2–4: Use "Project ID" as specified in the question and project.csv

---

**Summary:**  
Select a subset of projects to maximize total NPV, subject to the total capital budget, mutual exclusivity of Projects 4 & 7, pre-requisite (6 requires 1), contingent (10 requires 5), and binary selection variables. All data is mapped directly from project.csv, table_id: file_0_view_0.