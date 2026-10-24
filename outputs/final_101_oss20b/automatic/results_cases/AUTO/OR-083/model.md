**Mathematical Model**

Let  

* \(W=\{1,\dots ,12\}\) – set of workers (row index of *15.csv*)  
* \(T=\{A,B,\dots ,J\}\) – set of tasks (column names of *15.csv*)  

Parameters  

* \(c_{i t}\) – time required for worker \(i\) to complete task \(t\)  
  (value taken from the cell in row \(i\), column \(t\) of *15.csv*).  

Decision variables  

* \(x_{i t}\in\{0,1\}\) – equals 1 if worker \(i\) is assigned to task \(t\), 0 otherwise.  

Objective  

\[
\min \;\sum_{i\in W}\sum_{t\in T} c_{i t}\,x_{i t}
\]

Constraints  

1. **Task assignment** – each task is performed by exactly one worker  

\[
\sum_{i\in W} x_{i t}=1 \quad \forall\, t\in T
\]

2. **Worker capacity** – each worker can perform at most one task (two workers will remain idle)  

\[
\sum_{t\in T} x_{i t}\le 1 \quad \forall\, i\in W
\]

3. **Binary variables**  

\[
x_{i t}\in\{0,1\}\quad \forall\, i\in W,\; t\in T
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{i t}\) | *15.csv* (table_id: `file_0_view_0`) | For each worker row \(i\) (identified by the value in column **Task Time Required**) and each task column \(t\) (A–J), assign \(c_{i t}\) to the numeric value in that cell. |

No other data or constraints are required. The model above fully captures the assignment of 10 workers to 10 tasks while minimizing total working hours.