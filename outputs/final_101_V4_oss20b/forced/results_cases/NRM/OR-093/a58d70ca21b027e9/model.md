**Mathematical Model**

Let  

- \(I=\{M1,\dots ,M12\}\) (12 machines)  
- \(J=\{A,\dots ,L\}\) (12 tasks)  

Parameters  

- \(c_{ij}\) – machining cost of assigning machine \(i\) to task \(j\)  
- \(c_{ij}\in \mathbb{R}_{+}\)

Decision variables  

- \(x_{ij}\in\{0,1\}\) (1 if machine \(i\) is assigned to task \(j\), 0 otherwise)

Objective  

\[
\min \;\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Machine‑to‑task assignment**  
   \[
   \sum_{j\in J} x_{ij}=1,\qquad \forall i\in I
   \]

2. **Task‑to‑machine assignment**  
   \[
   \sum_{i\in I} x_{ij}=1,\qquad \forall j\in J
   \]

3. **Binary domain**  
   \[
   x_{ij}\in\{0,1\},\qquad \forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Symbol | Source | Table ID | Column(s) | Mapping |
|--------|--------|----------|-----------|---------|
| \(I\) | Machine list | `file_0_view_0` | `Machine` | Each distinct `Machine` value → index in \(I\) |
| \(J\) | Task list | `file_0_view_0` | `A`–`L` | Each column header `A`–`L` → index in \(J\) |
| \(c_{ij}\) | Cost values | `file_0_view_0` | `A`–`L` | For row with `Machine = i`, column `j` gives \(c_{ij}\) |
| \(x_{ij}\) | Decision variable | – | – | Binary variable for each \((i,j)\) pair |

All parameter values are taken directly from the rows of `cost_12x12.csv`; no additional data or transformations are introduced.