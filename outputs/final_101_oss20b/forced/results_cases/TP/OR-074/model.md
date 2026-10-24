**Linear programming model**

Let  

* \(P=\{1,\dots ,48\}\) – the 48 consecutive half‑hour periods of the day  
* \(T=\{1,\dots ,33\}\) – the periods at which a new 8‑hour shift may start  
  (a shift that starts at period \(t\) covers periods \(t,\dots ,t+15\))  

Parameters  

* \(r_p\) – required number of waitstaff in period \(p\) (obtained from *44.csv*)  

Decision variables  

* \(x_t\in\{0,1\}\) – 1 if a shift starts at period \(t\), 0 otherwise  

Objective  

\[
\min \sum_{t\in T} x_t
\]

Constraints  

\[
\sum_{\substack{t\in T\\ t\le p\le t+15}} x_t \;\ge\; r_p ,
\qquad \forall\, p\in P
\]

(Each period must be covered by at least the required number of staff.)

---

**Data‑mapping rules**

| Index | Source | Mapping |
|-------|--------|---------|
| \(p\in P\) | *file_0_view_0* – row \(p-1\) (0‑based) | \(r_p =\) integer value of column **Requirement** in that row |
| \(t\in T\) | *file_0_view_0* – row \(t-1\) (0‑based) | Used only to define the coverage set \(\{p: t\le p\le t+15\}\) |

All other data (period labels, times) are not needed for the LP formulation.