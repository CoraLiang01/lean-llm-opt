**Mathematical model**

Let  

* \(W=\{1,\dots ,12\}\) – set of workers  
* \(T=\{A,B,C,D,E,F,G,H,I,J\}\) – set of tasks  

Parameters  

* \(c_{w t}\) – time (hours) required for worker \(w\) to complete task \(t\)  

Decision variables  

* \(x_{w t}\in\{0,1\}\) – 1 if worker \(w\) is assigned to task \(t\), 0 otherwise  

Objective  

\[
\min \;\sum_{w\in W}\sum_{t\in T} c_{w t}\,x_{w t}
\]

Constraints  

1. **Task assignment** – each task is performed by exactly one worker  

\[
\sum_{w\in W} x_{w t}=1 \qquad \forall\,t\in T
\]

2. **Worker capacity** – each worker is assigned to at most one task (10 workers are chosen)  

\[
\sum_{t\in T} x_{w t}\le 1 \qquad \forall\,w\in W
\]

3. **Binary variables**  

\[
x_{w t}\in\{0,1\}\qquad \forall\,w\in W,\;t\in T
\]

---

**Data mapping**

| Parameter | Source table | Source column | Source row key | Mapping rule |
|-----------|--------------|---------------|----------------|--------------|
| \(c_{w t}\) | `file_0_view_0` | `t` (one of A–J) | `Task Time Required` = `w` | `c_{w t} = value in column t of the row where Task Time Required = w` |

*The table `file_0_view_0` contains 13 rows: the first header row and 12 data rows, one per worker. The column `Task Time Required` holds the worker identifier (1–12). Each of the columns A–J holds the corresponding time value.*