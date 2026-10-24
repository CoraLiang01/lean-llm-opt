**Mathematical Model**

Let  

* \(I=\{A,B,C,D,E,F,G,H,I,J\}\) (heavy‑machinery devices)  
* \(J=\{P1,P2,\dots ,P111\}\) (products)

Parameters  

* \(p_j\) – unit profit of product \(j\) \((j\in J)\)  
* \(t_{ij}\) – processing time of product \(j\) on device \(i\) \((i\in I,\;j\in J)\)  
* \(c_i\) – monthly operating capacity of device \(i\) \((i\in I)\)

Decision variable  

* \(x_j\ge 0\) – quantity of product \(j\) produced in the month.

---

\[
\begin{aligned}
\max\;&\sum_{j\in J} p_j\,x_j \\[4pt]
\text{s.t.}\;&\sum_{j\in J} t_{ij}\,x_j \;\le\; c_i \qquad \forall\, i\in I \\[4pt]
&x_j \;\ge\; 0 \qquad \forall\, j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(p_j\)   | `file_2_view_0` | `Product` | `Unit_Profit` |
| \(t_{ij}\)| `file_0_view_0` | `Device`  | `P1` … `P111` |
| \(c_i\)   | `file_1_view_0` | `Device`  | `Monthly_Capacity` |

All indices (`Device`, `Product`) are taken directly from the corresponding tables; no additional filtering or transformation is applied.