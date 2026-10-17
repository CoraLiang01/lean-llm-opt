**Mathematical Model**

Let  

* \(M=\{m\}\) be the set of managers (rows of *manager_project_costs.csv*).  
* \(P=\{p\}\) be the set of projects (columns \(P1,P2,P3\)).  

Decision variables  

\[
x_{mp}\in\{0,1\}\quad\forall m\in M,\;p\in P
\]

*\(x_{mp}=1\)* if manager \(m\) is assigned to project \(p\), otherwise 0.

Parameters  

\[
c_{mp}\ge 0 \quad\text{(cost of manager }m\text{ on project }p)
\]

**Objective**

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

**Constraints**

1. **Manager assignment** – each manager is assigned to exactly one project  

\[
\sum_{p\in P} x_{mp}=1 \qquad \forall m\in M
\]

2. **Project assignment** – each project is managed by exactly one manager  

\[
\sum_{m\in M} x_{mp}=1 \qquad \forall p\in P
\]

3. **Binary variables**  

\[
x_{mp}\in\{0,1\}\qquad \forall m\in M,\;p\in P
\]

---

### Data Mapping

| Parameter / Index | Source | Table ID | Column | Mapping Rule |
|-------------------|--------|----------|--------|--------------|
| \(M\) | Manager identifiers | `file_0_view_0` | `Unnamed: 0` | \(M=\{\text{value of } \text{Unnamed: 0}\}\) |
| \(P\) | Project identifiers | `file_0_view_0` | `P1`, `P2`, `P3` | \(P=\{\text{column names except } \text{Unnamed: 0}\}\) |
| \(c_{mp}\) | Cost matrix | `file_0_view_0` | `P1`, `P2`, `P3` | For each row where `Unnamed: 0 = m` and column `p`, set \(c_{mp}=\text{value}\) |

All numerical values are taken directly from the CSV rows; no additional processing or hard‑coding of numbers is performed.