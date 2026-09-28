#### Abstract Project Selection Model

**Index Sets:**
- $P$: Set of all projects (indexed by $p$), as given by column "Project ID" in table_id file_0_view_0.

**Parameters:**
- $c_p$: Capital investment required for project $p$ ("Capital (k$)").
- $n_p$: Expected Net Present Value (NPV) of project $p$ ("NPV (k$)").
- $B$: Total available capital budget ($B = 1000$ k$).
- $P_{4,7}$: Set $\{4, 7\}$, the mutually exclusive project pair.
- $p_{6,1}$: Pre-requisite pair: project 6 requires project 1.
- $p_{10,5}$: Contingent pair: project 10 requires project 5.

**Decision Variables:**
- $x_p \in \{0,1\}$: $x_p = 1$ if project $p$ is selected; $0$ otherwise, for all $p \in P$.

---

**Objective:**
\[
\max \sum_{p \in P} n_p x_p
\]

---

**Constraints:**

1. **Budget Constraint:**
   \[
   \sum_{p \in P} c_p x_p \leq B
   \]

2. **Mutually Exclusive Constraint (Projects 4 & 7):**
   \[
   x_4 + x_7 \leq 1
   \]

3. **Pre-requisite Constraint (Project 6 requires 1):**
   \[
   x_6 \leq x_1
   \]

4. **Contingent Constraint (Project 10 requires 5):**
   \[
   x_{10} \leq x_5
   \]

5. **Variable Domain:**
   \[
   x_p \in \{0,1\} \quad \forall p \in P
   \]

---

#### Data Mapping

- **Table:** project.csv (table_id: file_0_view_0)
- **Columns:**
  - "Project ID" $\rightarrow$ $P$ (project index set)
  - "Project Name" (for reference in constraints)
  - "Capital (k$)" $\rightarrow$ $c_p$ (capital investment parameter)
  - "NPV (k$)" $\rightarrow$ $n_p$ (NPV parameter)

- **Special Project References:**
  - Project 1: "Infrastructure Upgrade"
  - Project 4: "R&D Initiative Alpha"
  - Project 5: "Staff Training Program"
  - Project 6: "System Automation"
  - Project 7: "Global Expansion Pilot"
  - Project 10: "Customer Experience Platform"

---

**All data and project identifiers are as provided in project.csv, table_id file_0_view_0.**