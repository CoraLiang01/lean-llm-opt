**Mathematical Model**

Let  

* \(W=\{w\}\) be the set of 12 workers (identified by the values in the column **Task Time Required**).  
* \(T=\{t\}\) be the set of 10 tasks (identified by the column names **A**–**J**).  

Parameters  

* \(c_{wt}\) – time (hours) required for worker \(w\) to complete task \(t\).  

Decision variables  

* \(x_{wt}\in\{0,1\}\) – equals 1 if worker \(w\) is assigned to task \(t\), 0 otherwise.  

Objective  

\[
\min \;\sum_{w\in W}\sum_{t\in T} c_{wt}\,x_{wt}
\]

Constraints  

1. **Task assignment** – each task is performed by exactly one worker  

\[
\sum_{w\in W} x_{wt}=1 \quad \forall\, t\in T
\]

2. **Worker capacity** – each worker performs at most one task (only 10 of the 12 workers will be used)  

\[
\sum_{t\in T} x_{wt}\le 1 \quad \forall\, w\in W
\]

3. **Binary variables**  

\[
x_{wt}\in\{0,1\}\quad \forall\, w\in W,\; t\in T
\]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "row_key": "Task Time Required",
  "columns": {
    "A": "Task_A",
    "B": "Task_B",
    "C": "Task_C",
    "D": "Task_D",
    "E": "Task_E",
    "F": "Task_F",
    "G": "Task_G",
    "H": "Task_H",
    "I": "Task_I",
    "J": "Task_J"
  },
  "parameter": "c",
  "parameter_mapping": {
    "row_key": "w",
    "columns": "t",
    "value": "c_{wt}"
  }
}
```

* `row_key` identifies each worker \(w\).  
* Each column name maps to a task \(t\).  
* The value in the cell gives the parameter \(c_{wt}\).