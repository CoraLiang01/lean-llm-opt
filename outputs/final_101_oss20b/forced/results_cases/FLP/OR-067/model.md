**Mathematical Model**

Let  

- \(I=\{MA,MB,MC\}\) (Managers)  
- \(J=\{P1,P2,P3\}\) (Projects)  

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
&\sum_{j\in J} x_{ij}=1 &&\forall i\in I \quad\text{(each manager to one project)}\\
&\sum_{i\in I} x_{ij}=1 &&\forall j\in J \quad\text{(each project to one manager)}
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Mapping |
|----------|--------|---------|
| `file_0_view_0` | `Unnamed: 0` | Manager identifier \(i\) |
| `file_0_view_0` | `P1`, `P2`, `P3` | Project identifiers \(j\) |
| `file_0_view_0` | cell \((i,j)\) | Parameter \(c_{ij}\) (numeric cost) |

Thus \(c_{ij}\) is obtained directly from the cell in row corresponding to manager \(i\) and column corresponding to project \(j\) in the CSV file `manager_project_costs.csv`. No other data are required.