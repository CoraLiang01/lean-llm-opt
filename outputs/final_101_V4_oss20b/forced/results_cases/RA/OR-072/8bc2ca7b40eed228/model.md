**Mathematical Model**

Let  

* \(T=\{0,1,\dots ,23\}\) be the set of hourly periods (period 0 = 0:00‑1:00, …, period 23 = 23:00‑0:00).  
* \(r_t\) = number of drivers/crew required during period \(t\).  
* \(x_t\) = number of drivers/crew who start a 4‑hour shift at the beginning of period \(t\).

All variables are non‑negative integers.

\[
\begin{aligned}
\min \quad & \sum_{t\in T} x_t \\[4pt]
\text{s.t.}\quad
& \sum_{k=0}^{3} x_{(t-k)\bmod 24} \;\ge\; r_t,
   && \forall\, t\in T, \\[4pt]
& x_t \;\in\; \mathbb{Z}_{\ge 0},
   && \forall\, t\in T .
\end{aligned}
\]

The constraint for period \(t\) ensures that the total number of drivers/crew working during that hour (those who started in the current period or in any of the previous three periods) meets or exceeds the required number \(r_t\).  
The modulo operation implements the wrap‑around at midnight.

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(r_t\) | Required number of drivers/crew | `file_0_view_0` | `Number Required` |
| \(x_t\) | Decision variable | – | – |

*The index \(t\) corresponds to the `Time` column in `file_0_view_0`; the mapping is one‑to‑one with the 24 rows of that table.*