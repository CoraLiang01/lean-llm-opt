**Model**

Let  

* \(S=\{1,\dots ,24\}\) – set of hourly shifts (Shift values in *42.csv*).  
* \(r_s\) – required number of staff (drivers + crew) for shift \(s\).  
* \(x_s\) – number of staff who start working at the beginning of shift \(s\).  

Staff work continuously for 4 consecutive hours, so a staff member who starts at shift \(s\) works shifts  
\(s,\;s+1,\;s+2,\;s+3\) (indices taken modulo 24).

\[
\begin{aligned}
\min \quad & \sum_{s\in S} x_s \\[4pt]
\text{s.t.}\quad
& \sum_{k=0}^{3} x_{\;((s-k-1)\bmod 24)+1}\;\;\ge\;\; r_s,
   && \forall\, s\in S \quad\text{(coverage)}\\[4pt]
& x_s \;\in\; \mathbb{Z}_{\ge 0},
   && \forall\, s\in S \quad\text{(integer, non‑negative)} .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Column |
|-----------|--------|--------|
| \(r_s\) | `file_0_view_0` | `Number Required` |
| \(S\) | `file_0_view_0` | `Shift` |

The index set \(S\) is obtained from the distinct values of the `Shift` column in *42.csv*.  
All other data (shift times, etc.) are not needed for the LP formulation.