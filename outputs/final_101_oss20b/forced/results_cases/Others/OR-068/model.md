## Mathematical Model

**Index sets**  
\(M=\{i\mid i\text{ is a manager in the table}\}\)  \(P=\{j\mid j\text{ is a project column in the table}\}\)

**Parameters**  
\(c_{ij}\) – assignment cost of manager \(i\) to project \(j\) (taken from the CSV)

**Decision variables**  
\(x_{ij}\in\{0,1\}\) – 1 if manager \(i\) is assigned to project \(j\), 0 otherwise

**Objective**  
\[
\min \;\sum_{i\in M}\sum_{j\in P} c_{ij}\,x_{ij}
\]

**Constraints**  

1. **Manager assignment**  
\[
\sum_{j\in P} x_{ij}=1 \qquad \forall i\in M
\]

2. **Project assignment**  
\[
\sum_{i\in M} x_{ij}=1 \qquad \forall j\in P
\]

---

## Data Mapping

| Element | Source | Mapping |
|---------|--------|---------|
| \(M\) | `file_0_view_0` – column **Unnamed: 0** | Set of distinct values in this column (MA, MB, MC, MD, ME, MF) |
| \(P\) | `file_0_view_0` – columns **P1**–**P6** | Set of column names (P1, P2, P3, P4, P5, P6) |
| \(c_{ij}\) | `file_0_view_0` – row where **Unnamed: 0** = \(i\), column = \(j\) | Numeric value in that cell |
| \(x_{ij}\) | Decision variable | Binary variable defined for every \((i,j)\in M\times P\) |

All constraints and the objective use only the mapped parameters and variables defined above. No additional data or assumptions are introduced.