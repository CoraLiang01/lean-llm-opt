---
#### Abstract Mathematical Model

**Sets:**
- $P$: Set of projects, indexed by $p$ (from "Project ID" in project.csv).

**Parameters:**
- $c_p$: Capital required for project $p$ ("Capital (k$)", table_id: file_0_view_0).
- $v_p$: Expected NPV for project $p$ ("NPV (k$)", table_id: file_0_view_0).
- $B$: Total available capital budget ($B = 1000$ k$).

**Decision Variables:**
- $x_p \in \{0,1\}$: $1$ if project $p$ is selected, $0$ otherwise, for all $p \in P$.

**Objective:**
\[
\max \sum_{p \in P} v_p x_p
\]

**Constraints:**

1. **Budget Constraint:**
   \[
   \sum_{p \in P} c_p x_p \leq B
   \]

2. **Mutually Exclusive Constraint (Projects 4 & 7):**
   \[
   x_{4} + x_{7} \leq 1
   \]

3. **Pre-requisite Constraint (Project 6 requires 1):**
   \[
   x_{6} \leq x_{1}
   \]

4. **Contingent Constraint (Project 10 requires 5):**
   \[
   x_{10} \leq x_{5}
   \]

5. **Binary Decision Variables:**
   \[
   x_p \in \{0,1\} \quad \forall p \in P
   \]

---

#### Data Mapping

- $P$: All projects in "Project ID" column of project.csv (table_id: file_0_view_0), preserved in original row order.
- $c_p$: "Capital (k$)" column, table_id: file_0_view_0, keyed by "Project ID".
- $v_p$: "NPV (k$)" column, table_id: file_0_view_0, keyed by "Project ID".
- Special project references:
    - Project 1: "Infrastructure Upgrade"
    - Project 4: "R&D Initiative Alpha"
    - Project 5: "Staff Training Program"
    - Project 6: "System Automation"
    - Project 7: "Global Expansion Pilot"
    - Project 10: "Customer Experience Platform"

---

**All data and constraints are mapped directly from project.csv as required.**