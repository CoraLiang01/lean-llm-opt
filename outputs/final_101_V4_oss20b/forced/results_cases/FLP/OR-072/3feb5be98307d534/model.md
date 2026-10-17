**Symbolic Linear Programming Model**

Let  

* \(S=\{1,2,\dots ,24\}\) be the set of hourly shifts.  
* \(r_s\) be the required number of drivers/crew for shift \(s\).  
* \(x_s\) be the number of workers who start their 4‑hour shift at the beginning of shift \(s\).

\[
\begin{aligned}
\min \quad & \sum_{s\in S} x_s \\[4pt]
\text{s.t.}\quad & \sum_{k=0}^{3} x_{\;((s-k-1)\bmod 24)+1}\;\;\ge\;\; r_s,
   && \forall\, s\in S \\[4pt]
& x_s \;\ge\; 0,\;\; x_s \in \mathbb{Z},
   && \forall\, s\in S .
\end{aligned}
\]

*The first constraint ensures that the workers whose 4‑hour shifts cover shift \(s\) meet or exceed the required number \(r_s\).  The modulo operation implements the wrap‑around from shift 24 back to shift 1.*

---

**Data Mapping Rules**

| Parameter / Variable | Source | Mapping |
|----------------------|--------|---------|
| \(S\) | table_id `file_0_view_0` | indices `Shift` (values 1–24) |
| \(r_s\) | table_id `file_0_view_0` | column `Number Required` where `Shift = s` |
| \(x_s\) | decision variable | no external data; decision variable indexed by `Shift` |

*Only the `Number Required` column is used; the `Time` column is informational and not part of the model.*