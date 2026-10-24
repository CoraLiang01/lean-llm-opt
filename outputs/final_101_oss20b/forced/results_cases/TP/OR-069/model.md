**Mathematical Model**

Let  

- \(M=\{m_1,\dots ,m_{11}\}\) (Managers)  
- \(P=\{p_1,\dots ,p_{11}\}\) (Projects)  

Decision variable  

\[
x_{ij}\in\{0,1\}\quad\forall i\in M,\;j\in P
\]

\[
\text{(1)}\qquad x_{ij}=1 \;\Longleftrightarrow\; \text{manager }i\text{ is assigned to project }j
\]

Objective  

\[
\min \;\sum_{i\in M}\sum_{j\in P} c_{ij}\,x_{ij}
\]

Assignment constraints  

\[
\sum_{j\in P} x_{ij}=1 \qquad \forall i\in M \tag{2}
\]

\[
\sum_{i\in M} x_{ij}=1 \qquad \forall j\in P \tag{3}
\]

All other constraints are implicit in the binary domain of \(x_{ij}\).

---

**Data Mapping Rules**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(c_{ij}\) | `file_0_view_0` | `c[Manager][Project k Cost]` → `c_{ij}` where `Manager` = \(i\) and `Project k Cost` = \(j\) |
| \(M\) | `file_0_view_0` | `Manager` column values → set \(M\) |
| \(P\) | `file_0_view_0` | `Project 1 Cost`, `Project 2 Cost`, …, `Project 11 Cost` column names → set \(P\) |

*All indices and parameter values are taken directly from the CSV file `manager_project_costs.csv` as presented in the observation.*