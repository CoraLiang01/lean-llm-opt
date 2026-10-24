**Mathematical Model**

Let  

- \(M=\{M_1,\dots ,M_{12}\}\) machines  
- \(T=\{A,B,\dots ,L\}\) tasks  

Parameters  

- \(c_{m t}\) machining cost for assigning machine \(m\) to task \(t\)  

Decision variables  

- \(x_{m t}\in\{0,1\}\) 1 if machine \(m\) is assigned to task \(t\), 0 otherwise  

Objective  

\[
\min \sum_{m\in M}\sum_{t\in T} c_{m t}\,x_{m t}
\]

Constraints  

\[
\sum_{t\in T} x_{m t}=1 \quad \forall m\in M \qquad\text{(each machine to one task)}
\]

\[
\sum_{m\in M} x_{m t}=1 \quad \forall t\in T \qquad\text{(each task to one machine)}
\]

\[
x_{m t}\in\{0,1\}\quad \forall m\in M,\; t\in T
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{m t}\) | `file_0_view_0` | For each row where `Machine = m` and column header `t`, assign the numeric value to \(c_{m t}\). |
| Index set \(M\) | `file_0_view_0` | All distinct values in the `Machine` column. |
| Index set \(T\) | `file_0_view_0` | All column headers except `Machine`. |

No numerical values are listed; the model references the data via the mapping rules above.