**Linear programming model**

Let  

* \(P=\{1,2,\dots ,48\}\) – the 48 consecutive half‑hour periods of the day.  
* \(R_p\) – required number of waitstaff in period \(p\) (obtained from *file_0_view_0*).  
* \(S_t\) – number of waitstaff whose shift starts at the beginning of period \(t\) (integer, \(S_t\ge 0\)).  

Each waitstaff works a continuous 8‑hour shift, i.e. 16 consecutive half‑hour periods.  
Therefore a staff starting at period \(t\) is present in periods \(t,t+1,\dots ,t+15\) (up to period 48).

\[
\begin{aligned}
\min\;&\sum_{t\in P} S_t \\[4pt]
\text{s.t. } &\sum_{t=\max(1,p-15)}^{p} S_t \;\ge\; R_p , \qquad \forall\,p\in P \\[4pt]
& S_t \in \mathbb{Z}_{\ge 0}, \qquad \forall\,t\in P .
\end{aligned}
\]

---

### Data‑mapping rules

| Symbol | Source | Table ID | Column name | Notes |
|--------|--------|----------|-------------|-------|
| \(R_p\) | Requirement | `file_0_view_0` | `Requirement` | Indexed by the row order (1 → first row, …, 48 → last row). |
| \(S_t\) | Decision variable | – | – | Integer, non‑negative. |

The model uses the full set of 48 periods; no aggregation or additional constraints are introduced beyond those required by the problem statement.