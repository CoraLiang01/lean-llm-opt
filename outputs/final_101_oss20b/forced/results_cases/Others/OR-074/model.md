**Mathematical Model**

Let  

* \(T=\{0,1,\dots ,47\}\) – 48 consecutive 30‑minute time slots covering 24 h.  
* \(R_t\) – required number of waitstaff in slot \(t\) (taken from *44.csv*).  
* \(x_t\) – number of waitstaff whose shift starts at slot \(t\).  
  (Each shift lasts 8 h = 16 slots.)

\[
\begin{aligned}
\min \quad & \sum_{t\in T} x_t \\[4pt]
\text{s.t.}\quad
& \sum_{\substack{t\in T\\ t\le s < t+16 \pmod{48}}} x_t \;\ge\; R_s,
   && \forall\, s\in T \quad\text{(coverage)}\\[4pt]
& x_t \;\in\; \mathbb{Z}_{\ge 0},
   && \forall\, t\in T .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(T\) | implicit (48 slots) | \(t=0,\dots ,47\) |
| \(R_t\) | *44.csv* (table_id `file_0_view_0`) | `Requirement` column, row `data_row_number = t` |
| Shift length | problem statement | 16 slots (8 h) |
| Objective | problem statement | minimize total staff |

All constraints and variables are defined solely from the data in *44.csv* and the
problem description. No additional data or assumptions are introduced.