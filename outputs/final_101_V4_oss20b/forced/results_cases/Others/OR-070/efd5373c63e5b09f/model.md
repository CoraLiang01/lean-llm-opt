**Mathematical Model**

Let  

- \(M=\{m_1,\dots ,m_7\}\) : set of managers (taken from the *Manager* column)  
- \(P=\{p_1,\dots ,p_7\}\) : set of projects (taken from the column names *Project 1 Cost*, …, *Project 7 Cost*)

Parameter  

\[
c_{mp}\;=\;\text{cost for manager }m\text{ to complete project }p
\]

Decision variable  

\[
x_{mp}\in\{0,1\}\quad\forall m\in M,\;p\in P
\]

---

### Objective  

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

### Constraints  

1. **Each project is assigned to exactly one manager**  

\[
\sum_{m\in M} x_{mp}=1 \qquad \forall p\in P
\]

2. **Each manager is assigned to at most one project**  

\[
\sum_{p\in P} x_{mp}\le 1 \qquad \forall m\in M
\]

3. **Binary decision variables**  

\[
x_{mp}\in\{0,1\}\qquad \forall m\in M,\;p\in P
\]

---

## Data Mapping

| Parameter / Variable | Source | Table ID | Column(s) | Mapping |
|----------------------|--------|----------|-----------|---------|
| \(c_{mp}\) | Cost data | `file_0_view_0` | `Project 1 Cost`, …, `Project 7 Cost` | For each manager row \(m\), \(c_{mp}\) equals the value in the column corresponding to project \(p\). |
| \(M\) | Manager list | `file_0_view_0` | `Manager` | Each distinct value in the *Manager* column defines an element of \(M\). |
| \(P\) | Project list | `file_0_view_0` | `Project 1 Cost`, …, `Project 7 Cost` | Each column name (excluding *Manager*) defines an element of \(P\). |
| \(x_{mp}\) | Decision variable | – | – | Binary variable indicating assignment of manager \(m\) to project \(p\). |

All indices, parameters, and constraints are defined solely from the data in `manager_project_costs.csv` and the problem statement. No additional data or assumptions are introduced.