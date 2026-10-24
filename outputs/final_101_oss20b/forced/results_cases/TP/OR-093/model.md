**Mathematical Model**

Let  

- \(I=\{M1,M2,\dots ,M12\}\) (machines)  
- \(J=\{A,B,C,D,E,F,G,H,I,J,K,L\}\) (tasks)

Decision variables  

\[
x_{ij}\in\{0,1\}\quad\forall i\in I,\;j\in J
\]

where \(x_{ij}=1\) iff machine \(i\) is assigned to task \(j\).

Parameters  

\[
c_{ij}\ge 0 \quad\text{(machining cost of assigning machine \(i\) to task \(j\))}
\]

**Objective**

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Machine assignment**  
   \[
   \sum_{j\in J} x_{ij}=1 \quad\forall i\in I
   \]

2. **Task assignment**  
   \[
   \sum_{i\in I} x_{ij}=1 \quad\forall j\in J
   \]

---

**Data Mapping Rules**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(c_{ij}\) | Table **file_0_view_0** | `c[i,j]` ← value in column `j` of row where `Machine` = `i` |
| Index set \(I\) | Table **file_0_view_0** | `I` ← distinct values of column `Machine` |
| Index set \(J\) | Table **file_0_view_0** | `J` ← column names `A` through `L` |

All numeric values for \(c_{ij}\) are taken directly from the corresponding cells in **cost_12x12.csv** (table id `file_0_view_0`). No additional data or constraints are introduced beyond those specified.