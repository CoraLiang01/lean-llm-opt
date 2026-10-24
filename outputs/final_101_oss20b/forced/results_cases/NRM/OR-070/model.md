**Mathematical Model**

Let  

- \(M=\{1,\dots,7\}\) index set of managers  
- \(P=\{1,\dots,7\}\) index set of projects  

Parameters  

- \(c_{m,p}\) cost for manager \(m\) to complete project \(p\)  

Decision variables  

- \(x_{m,p}\in\{0,1\}\) equals 1 if manager \(m\) is assigned to project \(p\), 0 otherwise  

Objective  

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p}
\]

Constraints  

1. **Manager assignment** each manager is assigned to exactly one project  

\[
\sum_{p\in P} x_{m,p}=1,\qquad \forall m\in M
\]

2. **Project assignment** each project is assigned to exactly one manager  

\[
\sum_{m\in M} x_{m,p}=1,\qquad \forall p\in P
\]

3. **Binary variables**  

\[
x_{m,p}\in\{0,1\},\qquad \forall m\in M,\;p\in P
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(c_{m,p}\) | Cost value | `file_0_view_0` | `Project p Cost` (where `p` = 1…7) |
| Manager index \(m\) | Manager name | `file_0_view_0` | `Manager` |
| Project index \(p\) | Project number | `file_0_view_0` | implicit (Project 1 Cost … Project 7 Cost) |

The mapping rule: for each row in `file_0_view_0`, the value in column `Project p Cost` becomes \(c_{m,p}\) where \(m\) is the row’s `Manager` identifier and \(p\) is the project number corresponding to that column. No other data are required.