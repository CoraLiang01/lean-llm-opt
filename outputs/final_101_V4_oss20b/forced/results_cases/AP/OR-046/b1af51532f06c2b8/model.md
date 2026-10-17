**Mathematical Model**

Let  

- \(P\) be the set of products indexed by the values in the column **ProductName** of *products.csv*.  
- \(C\) be the overall stock capacity (units of weight) from *capacity.csv*.  
- \(w_p\) be the weight per unit of product \(p\in P\) (column **Weight**).  
- \(v_p\) be the benefit (value) per unit of product \(p\in P\) (column **Value**).  
- \(x_p\) be the number of units of product \(p\) ordered each day.

\[
\begin{aligned}
\max\;& \sum_{p\in P} v_p\,x_p \\[4pt]
\text{s.t.}\;& \sum_{p\in P} w_p\,x_p \;\le\; C \\[4pt]
& x_p \;\ge\; 0,\;\; x_p \in \mathbb{Z}\quad \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| \(C\) | `file_0_view_0` | `Capacity` | `C = Capacity` (single value) |
| \(w_p\) | `file_1_view_0` | `Weight` | `w_p = Weight` for each product \(p\) |
| \(v_p\) | `file_1_view_0` | `Value` | `v_p = Value` for each product \(p\) |
| \(P\) | `file_1_view_0` | `ProductName` | `P = {ProductName}` |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.