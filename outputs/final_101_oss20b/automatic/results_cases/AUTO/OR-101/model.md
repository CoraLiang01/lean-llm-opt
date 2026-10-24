**Mathematical Model**

Let  

* \(D=\{A,B,C,D,E,F,G,H,I,J\}\) – set of devices  
* \(P=\{P1,P2,\dots ,P111\}\) – set of products  

Decision variable  
\[
x_p \;\;(\text{units of product }p), \qquad p\in P
\]

Parameters  

\[
\begin{aligned}
\text{profit}_p &\;=\;\text{Unit\_Profit of product }p, \\
\text{time}_{d,p} &\;=\;\text{processing time of product }p\text{ on device }d, \\
\text{capacity}_d &\;=\;\text{Monthly\_Capacity of device }d .
\end{aligned}
\]

---

\[
\begin{aligned}
\max\;&\sum_{p\in P}\text{profit}_p\,x_p \\[4pt]
\text{s.t. } &\sum_{p\in P}\text{time}_{d,p}\,x_p \;\le\; \text{capacity}_d ,
\qquad \forall\, d\in D \\[4pt]
&x_p \;\ge\; 0 , \qquad \forall\, p\in P .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| `profit_p` | `file_2_view_0` | `Unit_Profit` |
| `time_{d,p}` | `file_0_view_0` | matrix entry at row `Device` = *d*, column `Product` = *p* |
| `capacity_d` | `file_1_view_0` | `Monthly_Capacity` |

*All indices (`d`, `p`) are taken from the identifiers present in the corresponding source tables.*