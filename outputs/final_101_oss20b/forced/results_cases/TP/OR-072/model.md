**Linear‑programming model**

Let  

* \(I=\{1,2,\dots ,24\}\) – index of the 24 one‑hour time periods (shifts).  
* \(r_i\) – number of drivers/crew members required in shift \(i\) (data from *42.csv*).  
* \(x_i\) – number of drivers/crew members who start their 4‑hour shift at the beginning of shift \(i\).  

All variables are non‑negative integers (continuous variables may be used in an LP relaxation).

\[
\begin{aligned}
\min \quad & \sum_{i\in I} x_i \\[4pt]
\text{s.t.}\quad
& \sum_{k=0}^{3} x_{\;((i-k-1)\bmod 24)+1}\;\;\ge\;\; r_i ,
   && \forall\, i\in I \quad\text{(coverage constraints)}\\[4pt]
& x_i \;\ge\; 0 , && \forall\, i\in I .
\end{aligned}
\]

*The coverage constraint* states that the drivers who start in the four preceding shifts (including the current one) are the only ones who can cover shift \(i\).  
The modulo operation implements the wrap‑around from shift 24 to shift 1.

---

### Data‑mapping table

| Table ID | Column name | Parameter | Index | Mapping rule |
|----------|-------------|-----------|-------|--------------|
| file_0_view_0 | Shift | \(r_i\) | \(i\in I\) | \(r_i =\) *Number Required* where *Shift* = \(i\) |
| file_0_view_0 | Shift | \(x_i\) | \(i\in I\) | Decision variable \(x_i\) corresponds to *Shift* = \(i\) |

All other data (time strings) are not used in the model. The model uses only the required numbers and the shift indices.