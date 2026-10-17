**Mathematical Model**

Let  

- \(S\) – set of store sections (indexed by the values in *SectionID*).  
- \(P\) – set of products (indexed by the values in *ProductName*).  

Parameters (all taken directly from the source tables)

- \(C_s\) – display‑space capacity of section \(s\)  \(\;(s\in S)\).  
- \(v_p\) – revenue (price) per unit of product \(p\)  \(\;(p\in P)\).  
- \(w_p\) – shelf‑space requirement per unit of product \(p\)  \(\;(p\in P)\).  

Decision variables

- \(x_{sp}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(p\) stocked in section \(s\).

---

\[
\begin{aligned}
\max\;& \sum_{s\in S}\sum_{p\in P} v_p\,x_{sp} \\[4pt]
\text{s.t. } & \sum_{p\in P} w_p\,x_{sp} \;\le\; C_s , && \forall s\in S \\[4pt]
& x_{sp}\;\in\;\mathbb{Z}_{\ge 0}, && \forall s\in S,\;p\in P .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns |
|-----------|--------------|----------------|
| \(C_s\)   | `file_0_view_0` | `SectionID`, `Capacity` |
| \(v_p\)   | `file_1_view_0` | `ProductName`, `Value` |
| \(w_p\)   | `file_1_view_0` | `ProductName`, `Weight` |

All indices \(s\) and \(p\) are taken from the distinct values of the corresponding source columns. No other data are introduced.