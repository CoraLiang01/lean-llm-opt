**Mathematical Model**

Let  

- \(I=\{\,i\mid i\text{ is a workstation}\,\}\)  
- \(J=\{\,j\mid j\text{ is a job}\,\}\)

Parameters  

- \(c_i\) – capacity of workstation \(i\)  
- \(a_{ij}\) – assignment cost of job \(j\) to workstation \(i\)  
- \(r_{ij}\) – resource consumption of job \(j\) on workstation \(i\)

Decision variables  

- \(x_{ij}\in\{0,1\}\) – 1 if job \(j\) is assigned to workstation \(i\), 0 otherwise  

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} a_{ij}\,x_{ij}
\]

Constraints  

1. **Exactly one workstation per job**  
\[
\sum_{i\in I} x_{ij}=1,\qquad \forall j\in J
\]

2. **Workstation capacity**  
\[
\sum_{j\in J} r_{ij}\,x_{ij}\le c_i,\qquad \forall i\in I
\]

3. **Binary variables**  
\[
x_{ij}\in\{0,1\},\qquad \forall i\in I,\;j\in J
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Indexing |
|-----------|----------|-----------|----------|
| \(I\) | `file_0_view_0` | `Workstation` | set of distinct `Workstation` values |
| \(J\) | `file_1_view_0` | `J1`–`J9` | set of column names excluding `Workstation` |
| \(c_i\) | `file_0_view_0` | `Capacity` | indexed by `Workstation` |
| \(a_{ij}\) | `file_1_view_0` | `J1`–`J9` | indexed by `Workstation` (row) and job column |
| \(r_{ij}\) | `file_2_view_0` | `J1`–`J9` | indexed by `Workstation` (row) and job column |

All numeric values are taken directly from the corresponding CSV files; no additional data are introduced.