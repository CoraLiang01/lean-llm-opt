## Mathematical Model

**Sets:**
- $P = \{1,2,\ldots,110\}$: set of projects, indexed by $i$.

**Parameters (from project.csv, table_id: file_0_view_0):**
- $c_i$: Capital required for project $i$ ("Capital (k$)")
- $v_i$: Expected NPV for project $i$ ("NPV (k$)")

**Decision Variables:**
- $x_i \in \{0,1\}$: $1$ if project $i$ is selected, $0$ otherwise

**Objective:**
\[
\max \sum_{i \in P} v_i x_i
\]

**Subject to:**

**Budget constraint:**
\[
\sum_{i \in P} c_i x_i \leq 1000
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
x_i \in \{0,1\} \quad \forall i \in P
\]

---

### Data Mapping

- $P$: All "Project ID" in project.csv (file_0_view_0), $1$ to $110$
- $c_i$: "Capital (k$)" for project $i$ in project.csv (file_0_view_0)
- $v_i$: "NPV (k$)" for project $i$ in project.csv (file_0_view_0)
- $x_i$: Binary variable for project $i$ selection

**Special constraints reference:**
- Mutually exclusive: $i=4$ ("R&D Initiative Alpha"), $i=7$ ("Global Expansion Pilot")
- Pre-requisite: $i=6$ ("System Automation"), $i=1$ ("Infrastructure Upgrade")
- Contingent: $i=10$ ("Customer Experience Platform"), $i=5$ ("Staff Training Program")

---

**All data and constraints are mapped directly from project.csv (file_0_view_0) and the problem description.**