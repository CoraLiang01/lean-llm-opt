**Mathematical Model**

\[
\begin{aligned}
\text{Sets} \quad & W=\{1,\dots,12\}, \qquad T=\{A,B,C,D,E,F,G,H,I,J\} \\[4pt]
\text{Parameters} \quad & \text{time}[w,t] \;\ge 0 \qquad \forall w\in W,\;t\in T \\[4pt]
\text{Decision Variables} \quad & x[w,t]\in\{0,1\} \qquad \forall w\in W,\;t\in T \\[4pt]
\text{Objective} \quad & \min Z = \sum_{w\in W}\sum_{t\in T}\text{time}[w,t]\;x[w,t] \\[6pt]
\text{Constraints} \\
& \sum_{w\in W} x[w,t] = 1 \qquad \forall t\in T \quad\text{(each task gets one worker)}\\[4pt]
& \sum_{t\in T} x[w,t] \le 1 \qquad \forall w\in W \quad\text{(each worker does at most one task)}\\[4pt]
& \sum_{w\in W}\sum_{t\in T} x[w,t] = 10 \qquad\text{(exactly ten workers are used)}\\
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Mapping Rule |
|-----------|-------------|--------------|
| `time[w,t]` | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv` | For each worker `w` (row where the column **Worker** equals `w`) and each task `t` (column header `t` among `A`–`J`), assign `time[w,t]` the numeric value in that cell. |

*The CSV layout:*  
- First column header: **Task Time Required** (row index).  
- Second column header: **Worker** (contains values 1–12).  
- Subsequent columns: **A, B, C, D, E, F, G, H, I, J** (task identifiers).  

No numerical values are listed here; the mapping rule references the exact file path and column names.