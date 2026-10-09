## Mathematical Model

**Sets**
- Let $P = \{1,2,\ldots,110\}$ be the set of all projects, indexed by $i$.
- Let $C_i$ = Capital (k$) required for project $i$, from column "Capital (k$)" in file_0_view_0.
- Let $V_i$ = NPV (k$) of project $i$, from column "NPV (k$)" in file_0_view_0.

**Decision Variables**
- $x_i \in \{0,1\}$: $x_i = 1$ if project $i$ is selected, $0$ otherwise, for all $i \in P$.

**Objective**
\[
\max \sum_{i \in P} V_i x_i
\]

**Subject to**

1. **Budget Constraint**
\[
\sum_{i \in P} C_i x_i \leq 1000
\]

2. **Mutually Exclusive Constraint (Projects 4 & 7)**
\[
x_4 + x_7 \leq 1
\]

3. **Pre-requisite Constraint (Project 6 requires 1)**
\[
x_6 \leq x_1
\]

4. **Contingent Constraint (Project 10 requires 5)**
\[
x_{10} \leq x_5
\]

5. **Binary Variables**
\[
x_i \in \{0,1\} \quad \forall i \in P
\]

---

**Data Mapping**

- $P$: All "Project ID" values in file_0_view_0 (project.csv), $P = \{1,2,\ldots,110\}$.
- $C_i$: "Capital (k$)" for project $i$ in file_0_view_0.
- $V_i$: "NPV (k$)" for project $i$ in file_0_view_0.
- Constraints reference project IDs as per the "Project ID" column:
    - Project 1: Infrastructure Upgrade
    - Project 4: R&D Initiative Alpha
    - Project 5: Staff Training Program
    - Project 6: System Automation
    - Project 7: Global Expansion Pilot
    - Project 10: Customer Experience Platform

All data is from file_0_view_0 (project.csv), columns: "Project ID", "Project Name", "Capital (k$)", "NPV (k$)".

---

**Summary**

- Maximize total NPV of selected projects.
- Total capital used $\leq$ 1,000 k$.
- At most one of projects 4 or 7 can be selected.
- Project 6 can only be selected if project 1 is selected.
- Project 10 can only be selected if project 5 is selected.
- All $x_i$ are binary.

**End of Model**