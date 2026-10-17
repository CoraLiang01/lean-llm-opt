**Mathematical Model**

Let  

- \(S\) be the set of store sections (indexed by the values in column **SectionID** of *capacity.csv*).  
- \(P\) be the set of products (indexed by the values in column **ProductName** of *products.csv*).  

Parameters  

- \(c_s\) – available display capacity of section \(s\) (units of space).  
- \(v_p\) – revenue per unit of product \(p\).  
- \(w_p\) – shelf‑space requirement per unit of product \(p\).  

Decision variables  

- \(x_{sp}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(p\) stocked in section \(s\).

\[
\begin{aligned}
\max\;& \sum_{s\in S}\sum_{p\in P} v_p\,x_{sp} \\[4pt]
\text{s.t. } & \sum_{p\in P} w_p\,x_{sp} \;\le\; c_s \qquad \forall s\in S \\[4pt]
& x_{sp}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| \(c_s\) | `file_0_view_0` | `SectionID`, `Capacity` | \(c_s = \text{Capacity}\) for the row where `SectionID = s` |
| \(v_p\) | `file_1_view_0` | `ProductName`, `Value` | \(v_p = \text{Value}\) for the row where `ProductName = p` |
| \(w_p\) | `file_1_view_0` | `ProductName`, `Weight` | \(w_p = \text{Weight}\) for the row where `ProductName = p` |

The index sets \(S\) and \(P\) are the distinct values of **SectionID** and **ProductName** respectively, as extracted from the two tables. No additional constraints or parameters are introduced beyond those specified.