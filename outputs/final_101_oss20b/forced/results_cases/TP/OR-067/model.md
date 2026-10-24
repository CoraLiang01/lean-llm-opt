**Mathematical Model**

Let  

- \(I=\{MA,MB,MC\}\) – set of managers  
- \(J=\{P1,P2,P3\}\) – set of projects  

Parameters  

- \(c_{ij}\) – cost for manager \(i\) to manage project \(j\)  

Decision variables  

- \(x_{ij}\in\{0,1\}\) – 1 if manager \(i\) is assigned to project \(j\), 0 otherwise  

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. }&\sum_{j\in J} x_{ij}=1 &&\forall i\in I \quad\text{(each manager gets one project)}\\
&\sum_{i\in I} x_{ij}=1 &&\forall j\in J \quad\text{(each project gets one manager)}\\
&x_{ij}\in\{0,1\} &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Table ID | Source File | Column | Mapping |
|----------|-------------|--------|---------|
| `file_0_view_0` | `manager_project_costs.csv` | `Unnamed: 0` | Manager identifier → index \(i\in I\) |
| `file_0_view_0` | `manager_project_costs.csv` | `P1` | Cost for project P1 → \(c_{i,P1}\) |
| `file_0_view_0` | `manager_project_costs.csv` | `P2` | Cost for project P2 → \(c_{i,P2}\) |
| `file_0_view_0` | `manager_project_costs.csv` | `P3` | Cost for project P3 → \(c_{i,P3}\) |

**Rule Description**

For each row \(r\) in `file_0_view_0`:

1. Let \(i =\) value of column `Unnamed: 0` in row \(r\).  
2. For each project column \(j \in \{P1,P2,P3\}\), set  
   \[
   c_{ij} = \text{numeric value in column } j \text{ of row } r.
   \]

The index sets \(I\) and \(J\) are derived from the distinct values in the `Unnamed: 0` column and the project columns, respectively. No additional data or constraints are introduced beyond those specified.