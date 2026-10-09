## Mathematical Model

**Sets:**
- $I = \{1,2,\ldots,110\}$: Set of all projects, indexed by $i$.

**Parameters (from Data Mapping):**
- $c_i$: Capital required for project $i$ (k$), from column "Capital (k$)" in table_id=file_0_view_0.
- $v_i$: Expected NPV for project $i$ (k$), from column "NPV (k$)" in table_id=file_0_view_0.

**Decision Variables:**
- $x_i \in \{0,1\}$: $1$ if project $i$ is selected, $0$ otherwise, for all $i \in I$.

---

**Objective:**
$$
\max \sum_{i \in I} v_i x_i
$$

---

**Constraints:**

1. **Budget Constraint:**
   $$
   \sum_{i \in I} c_i x_i \leq 1000
   $$

2. **Mutually Exclusive Constraint (Projects 4 & 7):**
   $$
   x_4 + x_7 \leq 1
   $$
   (Project 4: R&D Initiative Alpha, Project 7: Global Expansion Pilot)

3. **Pre-requisite Constraint (Project 6 requires 1):**
   $$
   x_6 \leq x_1
   $$
   (Project 6: System Automation, Project 1: Infrastructure Upgrade)

4. **Contingent Constraint (Project 10 requires 5):**
   $$
   x_{10} \leq x_5
   $$
   (Project 10: Customer Experience Platform, Project 5: Staff Training Program)

5. **Binary Variables:**
   $$
   x_i \in \{0,1\} \quad \forall i \in I
   $$

---

## Data Mapping

- $I$: All 110 projects, as listed in table_id=file_0_view_0, column "Project ID".
- $c_i$: "Capital (k$)" for project $i$ in table_id=file_0_view_0.
- $v_i$: "NPV (k$)" for project $i$ in table_id=file_0_view_0.
- Project names for constraints:
    - Project 1: Infrastructure Upgrade
    - Project 4: R&D Initiative Alpha
    - Project 5: Staff Training Program
    - Project 6: System Automation
    - Project 7: Global Expansion Pilot
    - Project 10: Customer Experience Platform

---

**Summary:**  
Select a subset of projects to maximize total NPV, subject to the total capital budget, mutual exclusivity, pre-requisite, and contingent selection constraints, with all variables binary. All parameters and project indices are mapped directly from the provided project.csv (table_id=file_0_view_0).