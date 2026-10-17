**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & Z = \sum_{w\in W}\sum_{t\in T} \; t_{w,t}\,x_{w,t} \\[4pt]
\text{subject to}\quad
&\sum_{w\in W} x_{w,t}=1 \qquad &&\forall\,t\in T \quad\text{(one worker per task)}\\[4pt]
&\sum_{t\in T} x_{w,t}\le 1 \qquad &&\forall\,w\in W \quad\text{(each worker at most one task)}\\[4pt]
&x_{w,t}\in\{0,1\} \qquad &&\forall\,w\in W,\;t\in T
\end{aligned}
\]

**Index Sets**

| Symbol | Meaning | Elements |
|--------|---------|----------|
| \(W\)  | Workers | \(\{A,B,C,D,E,F,G,H,I,J\}\) |
| \(T\)  | Tasks   | \(\{1,2,\dots,10\}\) |

**Parameters**

| Symbol | Meaning | Data source |
|--------|---------|-------------|
| \(t_{w,t}\) | Time required for worker \(w\) to complete task \(t\) | Table **file_0_view_0** (column \(w\), row where *Task Time Required* = \(t\)) |

**Data Mapping Rules**

1. **Parameter \(t_{w,t}\)**  
   - **Table ID**: `file_0_view_0`  
   - **Row key**: `Task Time Required` = \(t\) (integer 1–10)  
   - **Column key**: worker identifier \(w\) (A–J)  
   - **Value**: numeric entry in that cell (converted to a real number).

All other rows (tasks 11–12) and any unused workers are ignored by the model. The objective is to minimize the total working hours while assigning exactly one of the 10 selected workers to each of the 10 tasks.