**Abstract Mathematical Model**

**Index Sets:**
- $P$: Set of all projects, indexed by $p$. (From all "Project ID" in file_0_view_0)

**Parameters:**
- $c_p$: Capital required for project $p$. (From "Capital (k$)" in file_0_view_0)
- $v_p$: Expected NPV for project $p$. (From "NPV (k$)" in file_0_view_0)
- $B$: Total available capital budget. ($B = 1000$ k$)
- $P_4$: Project with "Project ID" = 4 (R&D Initiative Alpha)
- $P_7$: Project with "Project ID" = 7 (Global Expansion Pilot)
- $P_1$: Project with "Project ID" = 1 (Infrastructure Upgrade)
- $P_6$: Project with "Project ID" = 6 (System Automation)
- $P_5$: Project with "Project ID" = 5 (Staff Training Program)
- $P_{10}$: Project with "Project ID" = 10 (Customer Experience Platform)

**Decision Variables:**
- $x_p \in \{0,1\}$: $1$ if project $p$ is selected, $0$ otherwise.

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
   x_{P_4} + x_{P_7} \leq 1
   \]

3. **Pre-requisite Constraint (Project 6 requires 1):**
   \[
   x_{P_6} \leq x_{P_1}
   \]

4. **Contingent Constraint (Project 10 requires 5):**
   \[
   x_{P_{10}} \leq x_{P_5}
   \]

5. **Binary Selection:**
   \[
   x_p \in \{0,1\} \quad \forall p \in P
   \]

---

**Data Mapping**

- $P$: All "Project ID" in table_id: file_0_view_0, column: "Project ID"
- $c_p$: file_0_view_0, columns: "Project ID", "Capital (k$)"
- $v_p$: file_0_view_0, columns: "Project ID", "NPV (k$)"
- $P_1$, $P_4$, $P_5$, $P_6$, $P_7$, $P_{10}$: Identified by "Project ID" in file_0_view_0
- $B$: Query-defined as $1000$ k$

**Variable Domain Mapping**

- $x_p$: Binary variable for each $p \in P$ (project selection)

**All parameters and index sets are defined directly from the returned data.**