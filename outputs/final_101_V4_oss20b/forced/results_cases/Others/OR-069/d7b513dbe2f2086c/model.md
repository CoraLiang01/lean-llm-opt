## Mathematical Model

**Sets**  
\(M=\{1,\dots ,11\}\) – managers (row indices)  
\(P=\{1,\dots ,11\}\) – projects (column indices)

**Parameters**  
\(c_{ij}\) – assignment cost of manager \(i\) to project \(j\) (from the CSV)

**Decision Variables**  
\(x_{ij}\in\{0,1\}\) – 1 if manager \(i\) is assigned to project \(j\), 0 otherwise

**Objective**  
\[
\min \;\sum_{i\in M}\sum_{j\in P} c_{ij}\,x_{ij}
\]

**Constraints**  

1. **Manager assignment**  
\[
\sum_{j\in P} x_{ij}=1 \quad \forall i\in M
\]

2. **Project assignment**  
\[
\sum_{i\in M} x_{ij}=1 \quad \forall j\in P
\]

3. **Binary variables**  
\[
x_{ij}\in\{0,1\}\quad \forall i\in M,\; j\in P
\]

---

## Data Mapping

| Table ID | Column | Mapping Rule |
|----------|--------|--------------|
| `file_0_view_0` | `Manager` | Index \(i\) (1‑11) – each distinct manager row |
| `file_0_view_0` | `Project 1 Cost` | \(c_{i1}\) – cost for manager \(i\) to project 1 |
| `file_0_view_0` | `Project 2 Cost` | \(c_{i2}\) – cost for manager \(i\) to project 2 |
| `file_0_view_0` | `Project 3 Cost` | \(c_{i3}\) – cost for manager \(i\) to project 3 |
| `file_0_view_0` | `Project 4 Cost` | \(c_{i4}\) – cost for manager \(i\) to project 4 |
| `file_0_view_0` | `Project 5 Cost` | \(c_{i5}\) – cost for manager \(i\) to project 5 |
| `file_0_view_0` | `Project 6 Cost` | \(c_{i6}\) – cost for manager \(i\) to project 6 |
| `file_0_view_0` | `Project 7 Cost` | \(c_{i7}\) – cost for manager \(i\) to project 7 |
| `file_0_view_0` | `Project 8 Cost` | \(c_{i8}\) – cost for manager \(i\) to project 8 |
| `file_0_view_0` | `Project 9 Cost` | \(c_{i9}\) – cost for manager \(i\) to project 9 |
| `file_0_view_0` | `Project 10 Cost` | \(c_{i10}\) – cost for manager \(i\) to project 10 |
| `file_0_view_0` | `Project 11 Cost` | \(c_{i11}\) – cost for manager \(i\) to project 11 |

All indices \(i\) and \(j\) are taken from the full set of rows and columns present in the CSV; no additional filtering is applied. The model uses the exact cost values as provided.