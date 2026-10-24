**Mathematical Model**

\[
\begin{aligned}
\min \quad & \sum_{i\in M}\sum_{j\in T} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\quad
& \sum_{j\in T} x_{ij}=1 && \forall i\in M \quad\text{(each machine assigned to one task)}\\
& \sum_{i\in M} x_{ij}=1 && \forall j\in T \quad\text{(each task assigned to one machine)}\\
& x_{ij}\in\{0,1\} && \forall i\in M,\; j\in T
\end{aligned}
\]

**Index Sets**

- \(M=\{M1,M2,\dots ,M12\}\) (machines, taken from the *Machine* column of the data table)  
- \(T=\{A,B,C,D,E,F,G,H,I,J,K,L\}\) (tasks, taken from the column headers of the data table)

**Parameters**

- \(c_{ij}\) – machining cost of assigning machine \(i\) to task \(j\) (taken from the table entry where *Machine* = \(i\) and column = \(j\))  

**Data Mapping Rules**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(c_{ij}\) | `file_0_view_0` | For each row where `Machine = i` and column header = `j`, set \(c_{ij}\) to the numeric value in that cell. |
| \(M\) | `file_0_view_0` | Set of distinct values in the `Machine` column. |
| \(T\) | `file_0_view_0` | Set of column names excluding the `Machine` column. |

All other model components are defined symbolically; no numerical values are enumerated.