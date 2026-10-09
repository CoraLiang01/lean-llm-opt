## Mathematical Model

**Sets:**
- $P$: set of all projects, indexed by $p$ (from 1 to 110).

**Parameters (from project.csv, table_id: file_0_view_0):**
- $c_p$: Capital required for project $p$ ("Capital (k$)")
- $v_p$: Expected NPV for project $p$ ("NPV (k$)")

**Decision Variables:**
- $x_p \in \{0,1\}$: 1 if project $p$ is selected, 0 otherwise

---

**Objective:**
$$
\max \sum_{p \in P} v_p x_p
$$

**Subject to:**

1. **Budget Constraint:**
   $$
   \sum_{p \in P} c_p x_p \leq 1000
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
   x_p \in \{0,1\} \quad \forall p \in P
   $$

---

**Data Mapping:**

- $P$ = set of all "Project ID" in file_0_view_0 (project.csv), $|P|=110$
- $c_p$ = "Capital (k$)" for project $p$ in file_0_view_0
- $v_p$ = "NPV (k$)" for project $p$ in file_0_view_0

**Special project mappings:**
- Project 1: Infrastructure Upgrade
- Project 4: R&D Initiative Alpha
- Project 5: Staff Training Program
- Project 6: System Automation
- Project 7: Global Expansion Pilot
- Project 10: Customer Experience Platform

---

**Summary:**  
Select a subset of projects to maximize total NPV, subject to the total capital budget, mutual exclusivity of projects 4 and 7, prerequisite of project 1 for project 6, and contingency of project 5 for project 10. All variables are binary. All data is mapped directly from project.csv (file_0_view_0).