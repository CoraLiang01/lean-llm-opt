**Mathematical Model**

Let  

* \(I=\{M1,M2,\dots ,M12\}\) – set of machines  
* \(J=\{A,B,C,D,E,F,G,H,I,J,K,L\}\) – set of tasks  

Decision variables  

\[
x_{ij}\in\{0,1\}\qquad \forall i\in I,\;j\in J
\]

where \(x_{ij}=1\) if machine \(i\) is assigned to task \(j\).

Parameters  

\[
c_{ij}\ge 0 \qquad \forall i\in I,\;j\in J
\]

(cost of assigning machine \(i\) to task \(j\)).

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Machine‑to‑task assignment**  
   \[
   \sum_{j\in J} x_{ij}=1 \qquad \forall i\in I
   \]

2. **Task‑to‑machine assignment**  
   \[
   \sum_{i\in I} x_{ij}=1 \qquad \forall j\in J
   \]

3. **Binary variables**  
   \[
   x_{ij}\in\{0,1\}\qquad \forall i\in I,\;j\in J
   \]

---

### Data Mapping

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(c_{ij}\) | `file_0_view_0` (cost_12x12.csv) | For each row where `Machine = i` and column `j`, assign the numeric value to \(c_{ij}\). |
| Index set \(I\) | `file_0_view_0` | All distinct values in the `Machine` column. |
| Index set \(J\) | `file_0_view_0` | All column names except `Machine`. |

No other data are required. The model uses only the cost matrix provided; all other constraints are derived from the assignment problem structure.