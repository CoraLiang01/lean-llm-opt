**Mathematical Model**

Let  

* \(T=\{1,\dots ,24\}\) – the 24 one‑hour periods.  
* \(r_t\) – number of staff required in period \(t\) (taken from the column **Number Required** in *42.csv*).  
* \(x_t\) – number of staff who start working at the beginning of period \(t\).  

All variables are non‑negative integers.

\[
\begin{aligned}
\min \quad & \sum_{t\in T} x_t \\[4pt]
\text{s.t.}\quad 
& \sum_{k=0}^{3} x_{\;((t-k-1)\bmod 24)+1}\;\;\ge\;\; r_t,
   && \forall\, t\in T, \\[4pt]
& x_t \in \mathbb{Z}_{\ge 0},
   && \forall\, t\in T .
\end{aligned}
\]

The constraint for period \(t\) counts all staff who started in the four periods that cover \(t\) (including wrap‑around from the end of the day to the beginning).

---

**Data Mapping**

| Parameter | Source File | Source Column |
|-----------|-------------|---------------|
| \(r_t\)   | `Test_Dataset/Large-scale-or/Others_example/Others1/42.csv` | `Number Required` |
| Index \(t\) | same file | `Shift` |

No other parameters are required. The model uses only the data supplied in the observation.