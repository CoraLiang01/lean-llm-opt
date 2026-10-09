## Mathematical Model

**Sets:**
- $P = \{1,2,\ldots,110\}$: Set of all projects, indexed by $i$.

**Parameters (from project.csv, table_id: file_0_view_0):**
- $c_i$: Capital required for project $i$ ("Capital (k$)")
- $v_i$: Expected NPV for project $i$ ("NPV (k$)")

**Decision Variables:**
- $x_i \in \{0,1\}$: $1$ if project $i$ is selected, $0$ otherwise, for all $i \in P$

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
- Constraints reference project IDs as per "Project ID" in project.csv (e.g., $x_4$ is "R&D Initiative Alpha", $x_7$ is "Global Expansion Pilot", $x_6$ is "System Automation", $x_1$ is "Infrastructure Upgrade", $x_5$ is "Staff Training Program", $x_{10}$ is "Customer Experience Platform").

---

**Summary:**  
Select a subset of projects to maximize total expected NPV, subject to the total capital budget, mutual exclusivity, pre-requisite, and contingent constraints, with all variables binary. All data is mapped directly from project.csv (table_id: file_0_view_0).