**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{subject to}\quad 
& \sum_{j\in J} x_{ij}=1 &&\forall i\in I \quad\text{(each machine assigned once)}\\
& \sum_{i\in I} x_{ij}=1 &&\forall j\in J \quad\text{(each task assigned once)}\\
& x_{ij}\in\{0,1\} &&\forall i\in I,\; j\in J
\end{aligned}
\]

**Index Sets**

- \(I=\{M1,M2,\dots ,M12\}\)  (machines)  
- \(J=\{A,B,C,D,E,F,G,H,I,J,K,L\}\)  (tasks)

**Parameters**

- \(c_{ij}\) – machining cost for assigning machine \(i\) to task \(j\).

**Data Mapping**

| Parameter | Source Table | Row Key | Column Key |
|-----------|--------------|---------|------------|
| \(c_{ij}\) | `file_0_view_0` | `Machine` = \(i\) | column name = \(j\) |

*The sets \(I\) and \(J\) are derived from the distinct values in the `Machine` column and the column names `A`–`L` of the same table, respectively.*