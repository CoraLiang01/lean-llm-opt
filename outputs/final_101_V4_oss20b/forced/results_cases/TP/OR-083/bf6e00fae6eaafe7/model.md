**Mathematical model**

Let  

* \(W=\{1,2,\dots ,12\}\) ( workers, indexed by the row number in *15.csv* )  
* \(T=\{A,B,C,D,E,F,G,H,I,J\}\) ( tasks, indexed by the column names in *15.csv* )  

Parameters  

* \(t_{w\,t}\) – time (hours) required for worker \(w\) to complete task \(t\)  
  (taken from the cell in row \(w\), column \(t\) of *15.csv*).

Decision variables  

* \(x_{w\,t}\in\{0,1\}\) – equals 1 if worker \(w\) is assigned to task \(t\), 0 otherwise.

Objective  

\[
\min \;\sum_{w\in W}\;\sum_{t\in T}\; t_{w\,t}\;x_{w\,t}
\]

Constraints  

1. **One worker per task**  
   \[
   \sum_{w\in W} x_{w\,t}=1 \qquad \forall\,t\in T
   \]

2. **At most one task per worker**  
   \[
   \sum_{t\in T} x_{w\,t}\le 1 \qquad \forall\,w\in W
   \]

3. **Binary assignment**  
   \[
   x_{w\,t}\in\{0,1\}\qquad \forall\,w\in W,\;t\in T
   \]

Because each task receives exactly one worker and each worker can be assigned to at most one task, the total number of assignments is automatically 10, satisfying the requirement to use 10 of the 12 workers.

---

**Data‑mapping rules**

| Parameter | Source table | Source columns | Mapping rule |
|-----------|--------------|----------------|--------------|
| \(t_{w\,t}\) | `file_0_view_0` | `Task Time Required` (row header), `A`–`J` (columns) | For each row \(w\) (1‑12) and column \(t\) in \{A,…,J\}, set \(t_{w\,t}\) to the numeric value in that cell. |

*The table `file_0_view_0` is the view of `15.csv` provided in the observation. No additional data or transformations are required.*