**Mathematical Model**

Let  

* \(W=\{1,\dots ,12\}\) – set of workers  
* \(T=\{A,B,\dots ,J\}\) – set of tasks  

Decision variables  

\[
x_{w t}\in\{0,1\}\quad\forall w\in W,\;t\in T
\]

*\(x_{w t}=1\) if worker \(w\) is assigned to task \(t\).*

Parameters  

\[
t_{w t}\ge 0 \quad\forall w\in W,\;t\in T
\]

*time required for worker \(w\) to complete task \(t\).*

**Objective**

\[
\min \;\sum_{w\in W}\sum_{t\in T} t_{w t}\,x_{w t}
\]

**Constraints**

1. **Task assignment** – each task is performed by exactly one worker  

   \[
   \sum_{w\in W} x_{w t}=1 \qquad \forall t\in T
   \]

2. **Worker capacity** – a worker can perform at most one task  

   \[
   \sum_{t\in T} x_{w t}\le 1 \qquad \forall w\in W
   \]

(Constraint 1 already guarantees that exactly 10 workers are used.)

---

### Data Mapping

| Parameter | Source | Table ID | Column(s) | Mapping |
|-----------|--------|----------|-----------|---------|
| \(t_{w t}\) | Time matrix | `file_0_view_0` | `Task Time Required` (row header), `A`–`J` (columns) | \(t_{w t} =\) value in row *w* and column *t* of the table |

*No other parameters are required.*