**Symbolic Linear Programming Model**

Let  

* \(T=\{1,2,\dots ,24\}\) – the 24 one‑hour shifts.  
* \(d_t\) – required number of drivers/crew members during shift \(t\).  
* \(x_t\in \mathbb{Z}_{\ge 0}\) – number of workers who start work at the beginning of shift \(t\).  

A worker who starts at shift \(t\) works continuously for 4 hours, i.e. he/she covers shifts  
\(t,\;t+1,\;t+2,\;t+3\) (indices taken modulo 24).  

\[
\begin{aligned}
\min \quad & \sum_{t\in T} x_t \\[4pt]
\text{s.t.}\quad & \sum_{\substack{t\in T\\ (t\le s\le t+3)\bmod 24}} x_t \;\ge\; d_s,
   && \forall\, s\in T, \\[4pt]
& x_t \in \mathbb{Z}_{\ge 0}, && \forall\, t\in T .
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(d_t\) | Demand per shift | `file_0_view_0` | `Number Required` |
| \(t\) | Shift index | `file_0_view_0` | `Shift` |

The mapping rule is: for each record in `file_0_view_0`, set \(d_{\text{Shift}} = \text{Number Required}\). No other data are required.