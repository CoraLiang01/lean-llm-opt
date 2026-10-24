**Linear Programming Model**

Let  

* \(S=\{1,2,\dots ,24\}\) – index of the hourly shifts.  
* \(r_i\) – number of drivers/crew members required in shift \(i\).  
* \(x_i\) – number of workers who start their 4‑hour shift at the beginning of shift \(i\).

The workers that start at shift \(s\) cover the four consecutive shifts  
\(s,\;s+1,\;s+2,\;s+3\).  
Hence shift \(i\) is covered by workers that started at shifts  
\(s\in\{\,\max(1,i-3),\dots ,i\,\}\).

\[
\begin{aligned}
\min\;&\sum_{i\in S} x_i \\[4pt]
\text{s.t. } &\sum_{s=\max(1,i-3)}^{i} x_s \;\ge\; r_i
\qquad \forall\, i\in S \\[4pt]
&x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall\, i\in S
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(r_i\) | `file_0_view_0` | `Number Required` column, indexed by `Shift` (1–24) |

The matrix of coefficients \(a_{i,s}\) used in the constraints is implicitly defined by the 4‑hour coverage rule:  
\(a_{i,s}=1\) if \(s\in[\max(1,i-3),\,i]\), otherwise \(a_{i,s}=0\).