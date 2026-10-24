**Mathematical Model**

Let  

* \(M=\{1,\dots ,7\}\) – set of managers (index \(m\)).  
* \(P=\{1,\dots ,7\}\) – set of projects (index \(p\)).  

**Parameters**

* \(c_{mp}\) – cost for manager \(m\) to complete project \(p\).  
  \(c_{mp}\) is taken from the table **file_0_view_0** (column *Project \(p\) Cost* for row *Manager \(m\)*).

**Decision Variables**

* \(x_{mp}\in\{0,1\}\) – equals 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.

**Objective**

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

**Constraints**

1. **Each manager is assigned to at most one project**

\[
\sum_{p\in P} x_{mp}\;\le\;1 \qquad \forall m\in M
\]

2. **Each project is assigned to exactly one manager**

\[
\sum_{m\in M} x_{mp}\;=\;1 \qquad \forall p\in P
\]

3. **Binary decision variables**

\[
x_{mp}\in\{0,1\}\qquad \forall m\in M,\;p\in P
\]

---

### Data Mapping

| Element | Source | Mapping |
|---------|--------|---------|
| \(M\) | Managers in **file_0_view_0** | Index set derived from the *Manager* column (7 rows). |
| \(P\) | Projects in **file_0_view_0** | Index set derived from the column names *Project 1 Cost* … *Project 7 Cost* (7 columns). |
| \(c_{mp}\) | **file_0_view_0** | For each row *Manager \(m\)* and column *Project \(p\) Cost*, assign the numeric value to \(c_{mp}\). |
| \(x_{mp}\) | Decision variables | Binary variable for each manager‑project pair. |

No other data or constraints are introduced beyond those explicitly stated in the problem description.