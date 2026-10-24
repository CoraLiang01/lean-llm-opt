**Mathematical model**

\[
\begin{aligned}
\text{Minimize}\quad & Z = \sum_{s\in S} x_s \\[4pt]
\text{subject to}\quad
& \sum_{k=0}^{3} x_{\;((t-k-1)\bmod 24)+1}\;\;\ge\;\; R_t ,
&& \forall\,t\in S, \\[4pt]
& x_s \in \mathbb{Z}_{\ge 0},
&& \forall\,s\in S .
\end{aligned}
\]

*Index set*  
\(S=\{1,2,\dots ,24\}\) – the 24 hourly shifts.

*Parameters*  
\(R_t\) – number of staff required in shift \(t\).  
\(R_t\) is obtained from the table **file_0_view_0** (column *Number Required*) with key *Shift* = \(t\).

*Decision variables*  
\(x_s\) – number of drivers (and crew members) who start working at the beginning of shift \(s\).

*Explanation of the coverage constraint*  
A driver who starts at shift \(s\) works the four consecutive shifts  
\(s,\;s+1,\;s+2,\;s+3\) (indices taken modulo 24).  
Thus, for any shift \(t\) the set of drivers that are on duty is  
\(\{\,s\in S : t\in\{s,s+1,s+2,s+3\}\,\}\).  
The constraint written above enumerates these four starting shifts for each \(t\).

---

**Data‑mapping rules**

| Parameter | Source table | Key column | Value column |
|-----------|--------------|------------|--------------|
| \(R_t\)   | `file_0_view_0` | `Shift` (string `"1"`–`"24"`) | `Number Required` |

All other data (shift times, etc.) are not used in the model.