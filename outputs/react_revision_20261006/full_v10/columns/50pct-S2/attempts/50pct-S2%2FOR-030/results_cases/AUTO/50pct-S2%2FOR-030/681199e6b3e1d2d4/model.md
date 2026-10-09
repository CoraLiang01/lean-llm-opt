Let $I = \{1,2,\ldots,110\}$ be the set of project indices, with each project $i \in I$ having parameters:
- $c_i$: Capital (k$) [from "Capital (k$)" in file_0_view_0]
- $v_i$: NPV (k$) [from "NPV (k$)" in file_0_view_0]

Let $x_i \in \{0,1\}$ be a binary variable indicating whether project $i$ is selected.

Maximize
\[
\sum_{i \in I} v_i x_i
\]

Subject to

Budget constraint:
\[
\sum_{i \in I} c_i x_i \leq 1000
\]

Mutually exclusive constraint (Projects 4 & 7):
\[
x_4 + x_7 \leq 1
\]

Pre-requisite constraint (Project 6 requires 1):
\[
x_6 \leq x_1
\]

Contingent constraint (Project 10 requires 5):
\[
x_{10} \leq x_5
\]

Variable domains:
\[
x_i \in \{0,1\} \quad \forall i \in I
\]

Data Mapping:
- $I$: All "Project ID" in file_0_view_0 (project.csv), $|I|=110$
- $c_i$: "Capital (k$)" for project $i$ in file_0_view_0
- $v_i$: "NPV (k$)" for project $i$ in file_0_view_0
- $x_i$: Decision variable for project $i$ (selected or not)

Special project mappings:
- Project 1: Infrastructure Upgrade
- Project 4: R&D Initiative Alpha
- Project 5: Staff Training Program
- Project 6: System Automation
- Project 7: Global Expansion Pilot
- Project 10: Customer Experience Platform

All indices refer to the "Project ID" field in file_0_view_0. All parameters are taken directly from the corresponding columns in project.csv.