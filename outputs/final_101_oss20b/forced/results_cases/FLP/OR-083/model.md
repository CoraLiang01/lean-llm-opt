**Mathematical Model**

Let  

* \(W=\{1,\dots ,12\}\) – workers  
* \(T=\{A,B,\dots ,J\}\) – tasks  

Decision variables  

\[
x_{wt}\in\{0,1\}\quad\forall w\in W,\;t\in T
\]

( \(x_{wt}=1\) if worker \(w\) is assigned to task \(t\) ).

Parameters  

\[
c_{wt}\ge 0 \quad\text{(time required for worker }w\text{ to perform task }t)
\]

---

**Objective**

\[
\min \;\sum_{w\in W}\sum_{t\in T} c_{wt}\,x_{wt}
\]

---

**Constraints**

1. **Task coverage** – each task is performed by exactly one worker  

\[
\sum_{w\in W} x_{wt}=1 \qquad \forall t\in T
\]

2. **Worker capacity** – a worker can perform at most one task  

\[
\sum_{t\in T} x_{wt}\le 1 \qquad \forall w\in W
\]

3. **Number of workers used** – exactly ten workers are employed  

\[
\sum_{w\in W}\sum_{t\in T} x_{wt}=10
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(c_{wt}\) | `file_0_view_0` | Row with `Task Time Required` = worker \(w\); column `A`–`J` gives \(c_{wA},\dots ,c_{wJ}\). |
| Sets \(W,T\) | `file_0_view_0` | Workers: values of `Task Time Required` (1–12). Tasks: column headers `A`–`J`. |

No numerical values are listed; the model references the table directly.