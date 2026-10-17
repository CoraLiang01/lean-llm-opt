**Mathematical Model**

Let  

- \(M\) – set of managers (indices \(i\)).  
- \(P\) – set of projects (indices \(j\)).  

Parameters  

- \(c_{ij}\) – assignment cost of manager \(i\) to project \(j\).  

Decision variables  

- \(x_{ij}\in\{0,1\}\) – equals 1 if manager \(i\) is assigned to project \(j\), 0 otherwise.  

\[
\begin{aligned}
\min \quad & \sum_{i\in M}\sum_{j\in P} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\quad & \sum_{j\in P} x_{ij}=1 && \forall i\in M \quad\text{(each manager assigned once)}\\
& \sum_{i\in M} x_{ij}=1 && \forall j\in P \quad\text{(each project assigned once)}\\
& x_{ij}\in\{0,1\} && \forall i\in M,\; j\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(M\) | `file_0_view_0` | Set of distinct values in column **Manager** |
| \(P\) | `file_0_view_0` | Set \(\{1,2,\dots,11\}\) corresponding to columns **Project 1 Cost** … **Project 11 Cost** |
| \(c_{ij}\) | `file_0_view_0` | For each row where **Manager** = \(i\) and column **Project \(j\) Cost** = value, assign that value to \(c_{ij}\) |

All other data (indices, variable domains, objective sense, and constraints) are defined solely by the problem statement and do not require additional source references.