**Mathematical Model**

Let  

- \(I=\{\,i\,|\,i\text{ is a manager}\,\}\)  
- \(J=\{\,j\,|\,j\text{ is a project}\,\}\)  

Decision variables  

\[
x_{ij}\in\{0,1\}\quad\forall i\in I,\;j\in J
\]

Parameter  

\[
c_{ij}\ge 0 \quad\text{(cost of assigning manager }i\text{ to project }j)
\]

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

\[
\begin{aligned}
&\sum_{j\in J} x_{ij}=1 &&\forall i\in I \quad\text{(each manager gets one project)}\\[4pt]
&\sum_{i\in I} x_{ij}=1 &&\forall j\in J \quad\text{(each project gets one manager)}
\end{aligned}
\]

---

**Data Mapping Rules**

| Symbol | Source | Mapping |
|--------|--------|---------|
| \(I\) | `file_0_view_0` | Set of values in column **`Unnamed: 0`** (manager identifiers) |
| \(J\) | `file_0_view_0` | Set of column names **`P1`, `P2`, `P3`** (project identifiers) |
| \(c_{ij}\) | `file_0_view_0` | Value in the cell at row where `Unnamed: 0 = i` and column `Pj = j` |
| \(x_{ij}\) | Decision variable | Binary variable defined for each pair \((i,j)\) |

All numerical values for \(c_{ij}\) are taken directly from the corresponding table cells; no additional processing or enumeration is performed.