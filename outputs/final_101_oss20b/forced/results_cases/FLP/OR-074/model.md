**Symbolic Model**

Let  

* \(T=\{1,\dots ,48\}\) – the 48 consecutive half‑hour periods of the day.  
* \(R_j\) – required number of waitstaff in period \(j\) (data from *44.csv*).  
* \(y_t\in \mathbb{Z}_{+}\) – number of waitstaff whose shift starts at the beginning of period \(t\).

A waitstaff works continuously for 8 h = 16 half‑hour periods, so a staff starting at period \(t\) works in periods  
\(t,\,t+1,\,\dots ,\,t+15\) (indices truncated to the set \(T\)).

\[
\begin{aligned}
\min\;&\sum_{t\in T} y_t \\[4pt]
\text{s.t.}\;&\sum_{\substack{t\in T\\ t\le j\le t+15}} y_t \;\ge\; R_j,
\qquad \forall\, j\in T, \\[4pt]
&y_t\in\mathbb{Z}_{+},\qquad \forall\, t\in T .
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(R_j\) | Requirement | `file_0_view_0` | `Requirement` |
| \(T\) | Index set | – | – |
| \(y_t\) | Decision variable | – | – |

*The table `file_0_view_0` contains 48 rows, each with a `Requirement` value for the corresponding half‑hour period.*