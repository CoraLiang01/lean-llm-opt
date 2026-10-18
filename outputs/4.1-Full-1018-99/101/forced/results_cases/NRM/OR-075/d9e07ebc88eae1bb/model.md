#### Index Sets

- $P$: Set of all projects (as indexed in project.csv, with identifiers and names preserved).

#### Parameters

- $c_p$: Capital investment required for project $p \in P$ (from column "Capital (k$)").
- $v_p$: Expected Net Present Value (NPV) for project $p \in P$ (from column "NPV (k$)").

#### Decision Variables

- $x_p \in \{0,1\}$: 1 if project $p$ is selected, 0 otherwise, for all $p \in P$.

#### Objective

$$
\max \sum_{p \in P} v_p \cdot x_p
$$

#### Constraints

1. **Budget Constraint**
   $$
   \sum_{p \in P} c_p \cdot x_p \leq 1000
   $$

2. **Mutually Exclusive Constraint (Projects 4 & 7)**
   $$
   x_{4} + x_{7} \leq 1
   $$
   (Where Project 4 = "R&D Initiative Alpha", Project 7 = "Global Expansion Pilot")

3. **Pre-requisite Constraint (Project 6 requires 1)**
   $$
   x_{6} \leq x_{1}
   $$
   (Where Project 6 = "System Automation", Project 1 = "Infrastructure Upgrade")

4. **Contingent Constraint (Project 10 requires 5)**
   $$
   x_{10} \leq x_{5}
   $$
   (Where Project 10 = "Customer Experience Platform", Project 5 = "Staff Training Program")

5. **Variable Domain**
   $$
   x_p \in \{0,1\} \quad \forall p \in P
   $$

#### Data Mapping

- Source: project.csv
- Table ID: file_0_view_0
- Columns used:
    - "Project ID" (project identifier, used for $p$)
    - "Project Name" (project name, for mapping and documentation)
    - "Capital (k$)" (parameter $c_p$)
    - "NPV (k$)" (parameter $v_p$)
- All 110 records from project.csv are included, in source order, with no additional filtering.

All project indices and names referenced in constraints correspond to their "Project ID" and "Project Name" as given in project.csv.