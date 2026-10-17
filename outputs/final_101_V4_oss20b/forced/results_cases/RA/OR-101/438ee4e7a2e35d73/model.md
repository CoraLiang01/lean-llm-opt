**Mathematical Model**

\[
\begin{aligned}
\max_{x}\;&\sum_{p\in P} \; \text{profit}_p \; x_p \\[4pt]
\text{s.t.}\;&\sum_{p\in P} \; \text{time}_{d,p}\; x_p \;\le\; \text{cap}_d ,
\qquad \forall d\in D \\[4pt]
&x_p \;\ge\; 0 , \qquad \forall p\in P
\end{aligned}
\]

*Sets*  
\(P\) – set of products \(\{P1,\dots ,P111\}\)  
\(D\) – set of devices with capacity data \(\{A,B,I,J\}\)

*Parameters*  
\(\text{profit}_p\) – unit profit of product \(p\)  
\(\text{time}_{d,p}\) – processing time of product \(p\) on device \(d\)  
\(\text{cap}_d\) – monthly operating capacity of device \(d\)

*Decision variables*  
\(x_p\) – production quantity of product \(p\) (continuous, \(\ge 0\))

---

### Data Mapping

| Parameter | Table ID | Key column(s) | Value column(s) | Notes |
|-----------|----------|---------------|-----------------|-------|
| \(\text{profit}_p\) | `file_2_view_0` | `Product` | `Unit_Profit` | One‑to‑one mapping from product to profit |
| \(\text{time}_{d,p}\) | `file_0_view_0` | Row: `Device` ; Column: `Product` | Numeric matrix entries | Full 10×111 matrix; only rows in \(D\) are used in constraints |
| \(\text{cap}_d\) | `file_1_view_0` | `Device` | `Monthly_Capacity` | Only devices present in this table (A, B, I, J) appear in \(D\) |

All indices \(p\) and \(d\) are derived directly from the identifiers in the corresponding tables; no additional enumeration or sorting is performed. The model uses the exact numeric values supplied in the CSV files.