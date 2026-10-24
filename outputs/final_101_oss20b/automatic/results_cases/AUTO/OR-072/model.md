**Mathematical Model**

Let  

* \(T=\{1,2,\dots ,24\}\) – the 24 one‑hour shifts of the day.  
* \(r_t\) – number of drivers (or crew members) required during shift \(t\).  
* \(x_t\) – number of drivers that start their 4‑hour shift at the beginning of shift \(t\).

The 4‑hour shift that starts at shift \(t\) covers the four consecutive shifts  
\(t,\;t+1,\;t+2,\;t+3\) (indices taken modulo 24, i.e. after 24 comes 1).

\[
\begin{aligned}
\min \quad & \sum_{t\in T} x_t \\[4pt]
\text{s.t.}\quad
& \sum_{\substack{t\in T\\ (s-t)\bmod 24 \in \{0,1,2,3\}}} x_t \;\ge\; r_s,
&& \forall\, s\in T \quad\text{(coverage)}\\[4pt]
& x_t \;\in\; \mathbb{Z}_{\ge 0},
&& \forall\, t\in T \quad\text{(integer, non‑negative)} .
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Source | Column(s) | Mapping |
|----------------------|--------|-----------|---------|
| \(r_t\) | `file_0_view_0` | `Shift`, `Number Required` | \(r_t =\) value of **Number Required** where **Shift** = \(t\) |
| \(x_t\) | – | – | Decision variable, integer, non‑negative |

*The set \(T\) is derived from the distinct values of **Shift** in `file_0_view_0` (1–24).*

This model yields the minimum total number of drivers (or crew members) that must be scheduled so that every one‑hour shift meets its required staffing level, given that each driver works a continuous 4‑hour block starting at the beginning of a shift.