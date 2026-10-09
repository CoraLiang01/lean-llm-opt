#### Mathematical Model

Let:
- $I$ = set of all projects, indexed by $i$ (from all "Project ID" in file_0_view_0)
- For each $i \in I$:
    - $c_i$ = required capital investment for project $i$ ("Capital (k$)")
    - $v_i$ = expected NPV for project $i$ ("NPV (k$)")
    - $x_i \in \{0,1\}$: 1 if project $i$ is selected, 0 otherwise

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:

1. **Budget Constraint:**
$$
\sum_{i \in I} c_i x_i \leq 1000
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

5. **Binary Decision Variables:**
$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$: All "Project ID" in table_id=file_0_view_0, column="Project ID"
- $c_i$: "Capital (k$)" in table_id=file_0_view_0, column="Capital (k$)", for each $i$
- $v_i$: "NPV (k$)" in table_id=file_0_view_0, column="NPV (k$)", for each $i$
- $x_i$: Binary variable for each $i \in I$
- Project 4: "Project ID"=4 (R&D Initiative Alpha)
- Project 7: "Project ID"=7 (Global Expansion Pilot)
- Project 6: "Project ID"=6 (System Automation)
- Project 1: "Project ID"=1 (Infrastructure Upgrade)
- Project 10: "Project ID"=10 (Customer Experience Platform)
- Project 5: "Project ID"=5 (Staff Training Program)