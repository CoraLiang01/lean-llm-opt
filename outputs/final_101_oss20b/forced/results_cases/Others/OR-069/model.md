## Mathematical Model

**Sets**  
\(M=\{\text{Manager 1},\dots ,\text{Manager 11}\}\)  (Managers)  
\(P=\{\text{Project 1},\dots ,\text{Project 11}\}\)  (Projects)

**Parameters**  
\(c_{ij}\) – assignment cost of Manager \(i\) to Project \(j\)  
\(c_{ij}\) is taken from the table *file_0_view_0* (see Data Mapping).

**Decision Variables**  
\(x_{ij}\in\{0,1\}\)  (1 if Manager \(i\) is assigned to Project \(j\), 0 otherwise)

**Objective**  
\[
\min \;\sum_{i\in M}\sum_{j\in P} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Manager assignment**  
\[
\sum_{j\in P} x_{ij}=1 \qquad \forall i\in M
\]

2. **Project assignment**  
\[
\sum_{i\in M} x_{ij}=1 \qquad \forall j\in P
\]

3. **Binary variables**  
\[
x_{ij}\in\{0,1\}\qquad \forall i\in M,\; j\in P
\]

---

## Data Mapping

| Table ID | Column Name | Mapping |
|----------|-------------|---------|
| file_0_view_0 | Manager | Index \(i\) in set \(M\) (row order) |
| file_0_view_0 | Project 1 Cost | Parameter \(c_{i1}\) for all \(i\in M\) |
| file_0_view_0 | Project 2 Cost | Parameter \(c_{i2}\) for all \(i\in M\) |
| file_0_view_0 | Project 3 Cost | Parameter \(c_{i3}\) for all \(i\in M\) |
| file_0_view_0 | Project 4 Cost | Parameter \(c_{i4}\) for all \(i\in M\) |
| file_0_view_0 | Project 5 Cost | Parameter \(c_{i5}\) for all \(i\in M\) |
| file_0_view_0 | Project 6 Cost | Parameter \(c_{i6}\) for all \(i\in M\) |
| file_0_view_0 | Project 7 Cost | Parameter \(c_{i7}\) for all \(i\in M\) |
| file_0_view_0 | Project 8 Cost | Parameter \(c_{i8}\) for all \(i\in M\) |
| file_0_view_0 | Project 9 Cost | Parameter \(c_{i9}\) for all \(i\in M\) |
| file_0_view_0 | Project 10 Cost | Parameter \(c_{i10}\) for all \(i\in M\) |
| file_0_view_0 | Project 11 Cost | Parameter \(c_{i11}\) for all \(i\in M\) |

All other data (e.g., row/column names) are used solely for indexing and are not part of the optimization parameters.