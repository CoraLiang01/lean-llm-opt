## Mathematical Model

**Indices**  
\(W=\{1,\dots ,12\}\) – workers (row numbers in 15.csv)  
\(T=\{A,B,\dots ,J\}\) – tasks (column names in 15.csv)

**Parameters**  
\(t_{ij}\) – time (hours) required for worker \(i\) to complete task \(j\)  
 \(t_{ij}\ge 0\) (see Data Mapping)

**Decision Variables**  
\(x_{ij}\in\{0,1\}\) = 1 if worker \(i\) is assigned to task \(j\), 0 otherwise

**Objective**  
\[
\min \;\sum_{i\in W}\sum_{j\in T} t_{ij}\,x_{ij}
\]

**Constraints**

1. **Task assignment** – each task is performed by exactly one worker  
\[
\sum_{i\in W} x_{ij}=1 \qquad \forall j\in T
\]

2. **Worker capacity** – each worker is assigned to at most one task  
\[
\sum_{j\in T} x_{ij}\le 1 \qquad \forall i\in W
\]

(An implicit consequence of the above is that exactly 10 workers are used.)

---

## Data Mapping

| Table ID | Column | Mapping to Parameter |
|----------|--------|-----------------------|
| `file_0_view_0` | `Task Time Required` (row header “Worker”) | row index \(i\) (1‑12) |
| `file_0_view_0` | `A` | \(t_{iA}\) |
| `file_0_view_0` | `B` | \(t_{iB}\) |
| `file_0_view_0` | `C` | \(t_{iC}\) |
| `file_0_view_0` | `D` | \(t_{iD}\) |
| `file_0_view_0` | `E` | \(t_{iE}\) |
| `file_0_view_0` | `F` | \(t_{iF}\) |
| `file_0_view_0` | `G` | \(t_{iG}\) |
| `file_0_view_0` | `H` | \(t_{iH}\) |
| `file_0_view_0` | `I` | \(t_{iI}\) |
| `file_0_view_0` | `J` | \(t_{iJ}\) |

*Each \(t_{ij}\) is taken directly from the cell at row \(i\) and column \(j\) of the CSV.*