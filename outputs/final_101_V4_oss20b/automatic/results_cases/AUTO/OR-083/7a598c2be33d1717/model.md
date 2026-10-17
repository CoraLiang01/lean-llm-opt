**Mathematical Model**

Let  

* \(W=\{1,\dots ,12\}\) – set of workers  
* \(T=\{A,B,C,D,E,F,G,H,I,J\}\) – set of tasks  

Parameters  

* \(c_{w t}\) – time (hours) required for worker \(w\) to complete task \(t\)  

Decision variables  

* \(x_{w t}\in\{0,1\}\) – equals 1 if worker \(w\) is assigned to task \(t\), 0 otherwise  

---

### Objective
\[
\min \;\sum_{w\in W}\sum_{t\in T} c_{w t}\,x_{w t}
\]

### Constraints
1. **Task assignment** – each task is handled by exactly one worker  
   \[
   \sum_{w\in W} x_{w t}=1 \quad \forall\, t\in T
   \]

2. **Worker capacity** – each worker is assigned to at most one task  
   \[
   \sum_{t\in T} x_{w t}\le 1 \quad \forall\, w\in W
   \]

3. **Binary variables**  
   \[
   x_{w t}\in\{0,1\}\quad \forall\, w\in W,\; t\in T
   \]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{w t}\) | `file_0_view_0` (15.csv) | For each worker \(w\) (row where **Task Time Required** = \(w\)) and each task \(t\) (column \(t\)), set \(c_{w t}\) to the cell value in that row‑column pair. |
| \(W\) | `file_0_view_0` | All distinct non‑header values in the **Task Time Required** column (1–12). |
| \(T\) | `file_0_view_0` | All column names except **Task Time Required** (A–J). |

No numerical values are enumerated; the model references the data through the mapping rules above.