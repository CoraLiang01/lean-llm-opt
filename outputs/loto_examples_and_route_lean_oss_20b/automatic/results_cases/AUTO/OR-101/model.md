**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{p\in P} \; \text{profit}_p \; x_p \\[4pt]
\text{subject to}\quad & \sum_{p\in P} \; \text{time}_{d,p}\; x_p \;\le\; \text{capacity}_d ,
\qquad \forall d\in D_{\text{cap}} \\[4pt]
& x_p \;\ge\; 0 , \qquad \forall p\in P
\end{aligned}
\]

*Indices*  
- \(P=\{P1,\dots ,P111\}\) – product identifiers  
- \(D=\{A,B,C,D,E,F,G,H,I,J\}\) – device identifiers  
- \(D_{\text{cap}}\subseteq D\) – devices for which a monthly capacity is supplied  
  (here \(D_{\text{cap}}=\{A,B,I,J\}\))

*Parameters*  
- \(\text{profit}_p\) – unit profit of product \(p\)  
- \(\text{time}_{d,p}\) – processing time of product \(p\) on device \(d\)  
- \(\text{capacity}_d\) – monthly operating capacity of device \(d\)

*Decision variables*  
- \(x_p\) – production quantity of product \(p\) (continuous, \(\ge 0\))

---

### Data Mapping

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(\text{profit}_p\) | `file_2_view_0` | `Unit_Profit` | Indexed by `Product` |
| \(\text{time}_{d,p}\) | `file_0_view_0` | `P1`–`P111` | Matrix with `Device` as row ID |
| \(\text{capacity}_d\) | `file_1_view_0` | `Monthly_Capacity` | Indexed by `Device` (only rows A, B, I, J are present) |

All indices and parameters are taken directly from the runtime records supplied in the observation. No numerical values are hard‑coded in the model.