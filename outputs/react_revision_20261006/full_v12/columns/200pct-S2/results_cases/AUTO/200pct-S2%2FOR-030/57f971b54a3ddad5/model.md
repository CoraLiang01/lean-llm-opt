## Mathematical Model

**Sets:**
- $I = \{1,2,\ldots,110\}$: set of projects, indexed by $i$

**Parameters (from project.csv, table_id: file_0_view_0):**
- $c_i$: Capital (k$) required for project $i$
- $v_i$: NPV (k$) of project $i$

**Decision variables:**
- $x_i \in \{0,1\}$: 1 if project $i$ is selected, 0 otherwise

**Objective:**
$$
\max \sum_{i \in I} v_i x_i
$$

**Subject to:**

**Budget constraint:**
$$
\sum_{i \in I} c_i x_i \leq 1000
$$

**Mutually exclusive constraint (Projects 4 & 7):**
$$
x_4 + x_7 \leq 1
$$

**Pre-requisite constraint (Project 6 requires 1):**
$$
x_6 \leq x_1
$$

**Contingent constraint (Project 10 requires 5):**
$$
x_{10} \leq x_5
$$

**Variable domains:**
$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

### Data Mapping

- $I$: All 110 projects, as listed in project.csv, table_id: file_0_view_0, column "Project ID"
- $c_i$: "Capital (k$)" for project $i$ (file_0_view_0)
- $v_i$: "NPV (k$)" for project $i$ (file_0_view_0)
- $x_i$: binary selection variable for project $i$

Special constraints reference:
- $x_4$: Project 4 (R&D Initiative Alpha)
- $x_7$: Project 7 (Global Expansion Pilot)
- $x_6$: Project 6 (System Automation)
- $x_1$: Project 1 (Infrastructure Upgrade)
- $x_{10}$: Project 10 (Customer Experience Platform)
- $x_5$: Project 5 (Staff Training Program)

All data is mapped directly from project.csv, table_id: file_0_view_0, columns "Project ID", "Project Name", "Capital (k$)", "NPV (k$)".