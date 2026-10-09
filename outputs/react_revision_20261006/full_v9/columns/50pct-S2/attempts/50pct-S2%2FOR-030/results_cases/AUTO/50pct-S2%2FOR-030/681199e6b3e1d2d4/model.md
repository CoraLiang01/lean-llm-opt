## Mathematical Model

**Sets:**
- $I = \{1,2,\ldots,110\}$: Set of all projects, indexed by $i$.

**Parameters (from project.csv, table_id: file_0_view_0):**
- $c_i$: Capital required for project $i$ ("Capital (k$)")
- $v_i$: Expected NPV for project $i$ ("NPV (k$)")

**Decision Variables:**
- $x_i \in \{0,1\}$: 1 if project $i$ is selected, 0 otherwise, for all $i \in I$

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

- $I$: All "Project ID" values in project.csv (table_id: file_0_view_0)
- $c_i$: "Capital (k$)" for project $i$ in project.csv (table_id: file_0_view_0)
- $v_i$: "NPV (k$)" for project $i$ in project.csv (table_id: file_0_view_0)
- Constraints reference project IDs as per "Project ID" in project.csv (e.g., $x_4$ is "R&D Initiative Alpha", $x_7$ is "Global Expansion Pilot", $x_6$ is "System Automation", $x_1$ is "Infrastructure Upgrade", $x_{10}$ is "Customer Experience Platform", $x_5$ is "Staff Training Program"). 

No other constraints or data are imposed. All parameters and indices are as in the current project.csv.