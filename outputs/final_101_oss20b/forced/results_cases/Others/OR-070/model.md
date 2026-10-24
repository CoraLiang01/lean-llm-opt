## Mathematical Model

**Index sets**

- \(M=\{m\mid m\text{ is a manager}\}\)  
- \(P=\{p\mid p\text{ is a project}\}\)

**Parameters**

- \(c_{mp}\) – cost for manager \(m\) to complete project \(p\).

**Decision variables**

- \(x_{mp}\in\{0,1\}\) – 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.

**Objective**

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

**Constraints**

1. **Each manager at most one project**

\[
\sum_{p\in P} x_{mp}\;\le\;1 \qquad \forall m\in M
\]

2. **Each project assigned to exactly one manager**

\[
\sum_{m\in M} x_{mp}\;=\;1 \qquad \forall p\in P
\]

3. **Binary variables**

\[
x_{mp}\in\{0,1\}\qquad \forall m\in M,\;p\in P
\]

---

## Data Mapping

| Symbol | Source | Mapping Details |
|--------|--------|-----------------|
| \(M\) | `file_0_view_0` | Set of values in column **"Manager"** (rows 0–6). |
| \(P\) | `file_0_view_0` | Set of project indices derived from column names **"Project 1 Cost"** … **"Project 7 Cost"**. |
| \(c_{mp}\) | `file_0_view_0` | For each row where **"Manager"** = \(m\) and column **"Project p Cost"** = \(c_{mp}\). |
| \(x_{mp}\) | Decision variable | Binary variable defined for every pair \((m,p)\) with \(m\in M,\;p\in P\). |

All parameters and sets are taken directly from the returned rows of the CSV; no additional values are introduced. The model is a standard linear assignment problem with binary decision variables.