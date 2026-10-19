**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad & \sum_{i\in I}\sum_{j\in J} \text{Cost}_{ij}\;x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} x_{ij}=1, &&\forall j\in J \qquad\text{(each job assigned once)}\\[4pt]
& \sum_{j\in J} \text{Resource}_{ij}\;x_{ij}\le \text{Capacity}_{i}, &&\forall i\in I \qquad\text{(team capacity)}\\[4pt]
& x_{ij}\in\{0,1\}, &&\forall i\in I,\;j\in J
\end{aligned}
\]

**Index Sets**

- \(I=\{M1,M2,M3,M4\}\) (technician teams)  
- \(J=\{J1,J2,J3,J4,J5,J6,J7,J8\}\) (jobs)

**Parameters**

| Symbol | Description | Data Source |
|--------|-------------|-------------|
| \(\text{Capacity}_{i}\) | Available capacity of team \(i\) | `file_0_view_0`, column `Capacity`, key `Machine` |
| \(\text{Cost}_{ij}\) | Assignment cost of job \(j\) to team \(i\) | `file_1_view_0`, column `Jj`, key `Machine` |
| \(\text{Resource}_{ij}\) | Capacity consumed by assigning job \(j\) to team \(i\) | `file_2_view_0`, column `Jj`, key `Machine` |

**Data Mapping Rules**

- For each \(i\in I\):  
  \(\text{Capacity}_{i} \leftarrow\) value in `file_0_view_0` where `Machine = i`.
- For each \(i\in I,\; j\in J\):  
  \(\text{Cost}_{ij} \leftarrow\) value in `file_1_view_0` where `Machine = i` and column `Jj`.  
  \(\text{Resource}_{ij} \leftarrow\) value in `file_2_view_0` where `Machine = i` and column `Jj`.

No numerical values are hard‑coded; all coefficients are obtained from the specified tables.