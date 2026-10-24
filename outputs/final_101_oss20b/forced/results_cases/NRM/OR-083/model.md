**Symbolic Model**

\[
\begin{aligned}
\text{Sets} \quad & W=\{A,B,C,D,E,F,G,H,I,J\} \\
& T=\{1,2,\dots ,10\} \\[4pt]

\text{Parameters} \quad & \text{time}_{w,t}\;\; (w\in W,\;t\in T) \\[4pt]

\text{Decision Variables} \quad & x_{w,t}\in\{0,1\}\;\; (w\in W,\;t\in T) \\[4pt]

\text{Objective} \quad & \min \;\sum_{w\in W}\sum_{t\in T}\text{time}_{w,t}\,x_{w,t} \\[6pt]

\text{Constraints} \\
& \sum_{w\in W}x_{w,t}=1 \qquad \forall\,t\in T \quad\text{(each task gets one worker)}\\[4pt]
& \sum_{t\in T}x_{w,t}\le 1 \qquad \forall\,w\in W \quad\text{(each worker does at most one task)}\\[4pt]
& \sum_{w\in W}\sum_{t\in T}x_{w,t}=10 \qquad\text{(exactly ten assignments)}\\[4pt]
& x_{w,t}\in\{0,1\}\qquad \forall\,w\in W,\;t\in T
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| `time_{w,t}` | Table `file_0_view_0` (15.csv) | For each worker `w` (column `A`–`J`) and task `t` (row where `Task Time Required = t`), assign the numeric value in that cell to `time_{w,t}`. |

*No numeric values are listed; the mapping references the exact table ID and column names as required.*