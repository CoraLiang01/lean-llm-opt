**Mathematical Model**

Let  

- \(I=\{1,\dots,7\}\) index set of managers  
- \(J=\{1,\dots,7\}\) index set of projects  

Parameters  

- \(c_{ij}\) cost for manager \(i\) to complete project \(j\)  

Decision variables  

- \(x_{ij}\in\{0,1\}\) equals 1 if manager \(i\) is assigned to project \(j\), 0 otherwise  

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Manager assignment** each manager is assigned to exactly one project  

\[
\sum_{j\in J} x_{ij}=1,\qquad \forall i\in I
\]

2. **Project assignment** each project is assigned to exactly one manager  

\[
\sum_{i\in I} x_{ij}=1,\qquad \forall j\in J
\]

3. **Binary variables**  

\[
x_{ij}\in\{0,1\},\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) |
|-----------|--------|----------|-----------|
| \(c_{ij}\) | Cost value | `file_0_view_0` | `Project 1 Cost`, `Project 2 Cost`, …, `Project 7 Cost` (column index \(j\)) |
| Manager index \(i\) | Manager name | `file_0_view_0` | `Manager` |
| Project index \(j\) | Project number | `file_0_view_0` | implicit from column position (1–7) |

*No numerical values are listed; the mapping specifies how each parameter is extracted from the CSV file.*