## Mathematical Model

**Sets**  
\(M=\{M1,\dots ,M12\}\) (machines)  
\(T=\{A,B,\dots ,L\}\) (tasks)

**Parameters**  
\(c_{ij}\) – machining cost of assigning machine \(i\in M\) to task \(j\in T\)  
(Values taken from *cost_12x12.csv* – table_id `file_0_view_0`)

**Decision Variables**  
\(x_{ij}\in\{0,1\}\) (1 if machine \(i\) is assigned to task \(j\), 0 otherwise)

**Objective**  
\[
\min \;\sum_{i\in M}\sum_{j\in T} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Each machine assigned to exactly one task**  
\[
\sum_{j\in T} x_{ij}=1 \quad \forall i\in M
\]

2. **Each task assigned to exactly one machine**  
\[
\sum_{i\in M} x_{ij}=1 \quad \forall j\in T
\]

---

## Data Mapping

| Element | Source | Mapping |
|---------|--------|---------|
| **Set \(M\)** | Rows of *cost_12x12.csv* (column `Machine`) | `M = {row.Machine for row in table}` |
| **Set \(T\)** | Columns of *cost_12x12.csv* except `Machine` | `T = {col for col in table.columns if col != 'Machine'}` |
| **Parameter \(c_{ij}\)** | Cell at row `i`, column `j` of *cost_12x12.csv* | `c[i,j] = float(row[j])` |
| **Variable \(x_{ij}\)** | Decision variable defined for all \(i\in M, j\in T\) | Binary domain |

All constraints and the objective use only the indices and parameters defined above. No additional data or assumptions are introduced.