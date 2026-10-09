Let $I = \{1,2,\ldots,110\}$ be the set of project indices, with each project $i \in I$ having parameters:
- $c_i$: Capital (k$) required (from "Capital (k$)" in project.csv, table_id: file_0_view_0)
- $v_i$: NPV (k$) (from "NPV (k$)" in project.csv, table_id: file_0_view_0)

Let $x_i \in \{0,1\}$ be a binary variable indicating whether project $i$ is selected.

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**

**Budget constraint:**
\[
\sum_{i \in I} c_i x_i \leq 1000
\]

**Mutually exclusive constraint (Projects 4 & 7):**
\[
x_4 + x_7 \leq 1
\]

**Pre-requisite constraint (Project 6 requires 1):**
\[
x_6 \leq x_1
\]

**Contingent constraint (Project 10 requires 5):**
\[
x_{10} \leq x_5
\]

**Variable domains:**
\[
x_i \in \{0,1\} \quad \forall i \in I
\]

---

**Data Mapping:**
- $I$: All projects with "Project ID" in project.csv (table_id: file_0_view_0), $|I|=110$.
- $c_i$: "Capital (k$)" for project $i$ in project.csv (table_id: file_0_view_0).
- $v_i$: "NPV (k$)" for project $i$ in project.csv (table_id: file_0_view_0).
- $x_i$: Binary selection variable for project $i$.

Special project mappings (from project.csv, table_id: file_0_view_0):
- Project 1: Infrastructure Upgrade
- Project 4: R&D Initiative Alpha
- Project 5: Staff Training Program
- Project 6: System Automation
- Project 7: Global Expansion Pilot
- Project 10: Customer Experience Platform

All other projects indexed by their "Project ID".

---

**Summary:**  
Select a subset of projects to maximize total NPV, subject to the total capital budget, mutual exclusivity of projects 4 and 7, pre-requisite (6 requires 1), contingent (10 requires 5), and binary selection variables. All data is mapped directly from project.csv (table_id: file_0_view_0).