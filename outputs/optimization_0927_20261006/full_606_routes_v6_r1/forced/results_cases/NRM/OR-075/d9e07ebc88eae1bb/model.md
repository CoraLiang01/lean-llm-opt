#### Index Sets

- $P$: Set of all projects (from "Project ID" in project.csv).

#### Parameters

- $c_p$: Capital investment required for project $p \in P$ (from "Capital (k$)" in project.csv).
- $v_p$: Expected Net Present Value (NPV) for project $p \in P$ (from "NPV (k$)" in project.csv).

#### Decision Variables

- $x_p \in \{0,1\}$: 1 if project $p$ is selected, 0 otherwise, for all $p \in P$.

#### Objective

$$
\max \sum_{p \in P} v_p x_p
$$

#### Constraints

1. **Budget Constraint**
   $$
   \sum_{p \in P} c_p x_p \leq 1000
   $$

2. **Mutually Exclusive Constraint (Projects 4 & 7)**
   $$
   x_4 + x_7 \leq 1
   $$

3. **Pre-requisite Constraint (Project 6 requires 1)**
   $$
   x_6 \leq x_1
   $$

4. **Contingent Constraint (Project 10 requires 5)**
   $$
   x_{10} \leq x_5
   $$

5. **Variable Domain**
   $$
   x_p \in \{0,1\} \quad \forall p \in P
   $$

---

#### Data Mapping

- Source: project.csv
- Table ID: file_0_view_0
- Columns used:
    - "Project ID" (index set $P$)
    - "Capital (k$)" (parameter $c_p$)
    - "NPV (k$)" (parameter $v_p$)
- No filters applied; all 110 records used as returned by CSVQA.