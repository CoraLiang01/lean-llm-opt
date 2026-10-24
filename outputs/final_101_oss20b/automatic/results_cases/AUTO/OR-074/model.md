**Mathematical Model**

Let  

* \(T=\{1,\dots ,48\}\) – the 48 consecutive 30‑minute time slots that cover the 24‑hour day.  
* \(R_t\) – the required number of waitstaff in slot \(t\) (taken from *file_0_view_0*).  
* \(S_t\) – the number of waitstaff whose shift starts at the beginning of slot \(t\).  

A waitstaff works continuously for 8 h, i.e., 16 consecutive slots.  
Thus a staff starting at slot \(k\) works in slots  

\[
\{k,\,k+1,\,\dots ,\,k+15\}\pmod{48}.
\]

Define the binary coverage matrix  

\[
A_{t,k}= 
\begin{cases}
1 & \text{if } (t-k+48)\bmod 48 \le 15,\\
0 & \text{otherwise.}
\end{cases}
\]

The model is

\[
\begin{aligned}
\min \quad & \sum_{t\in T} S_t \\[4pt]
\text{s.t.}\quad
& \sum_{k\in T} A_{t,k}\,S_k \;\ge\; R_t, \qquad \forall\,t\in T, \\[4pt]
& S_t \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall\,t\in T .
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(R_t\) | Requirement per slot | `file_0_view_0` | `Requirement` |
| \(A_{t,k}\) | Coverage indicator | derived from `file_0_view_0` | computed as above |
| \(S_t\) | Decision variable | – | – |

The index set \(T\) is obtained from the row order of *file_0_view_0* (1 = first row, 48 = last row). No other data are required.