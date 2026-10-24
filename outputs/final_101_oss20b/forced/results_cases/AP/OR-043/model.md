**Mathematical Model**

Let  

- \(P\) be the set of drug products.  
- \(V_p\) be the benefit per unit of product \(p\).  
- \(W_p\) be the weight (stock space) per unit of product \(p\).  
- \(C\) be the overall stock capacity.  
- \(x_p\) be the number of units of product \(p\) ordered each day.

\[
\begin{aligned}
\max\;& \sum_{p\in P} V_p\,x_p \\[4pt]
\text{s.t.}\;& \sum_{p\in P} W_p\,x_p \;\le\; C \\[4pt]
& x_p \;\in\; \mathbb{Z}_{\ge 0}\quad \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(C\) | Capacity | `file_0_view_0` | `Capacity` |
| \(V_p\) | Benefit | `file_1_view_0` | `Value` |
| \(W_p\) | Weight | `file_1_view_0` | `Weight` |
| \(P\) | Products | `file_1_view_0` | `ProductName` |

All indices, parameters, and variables are defined exactly as above, with no additional enumeration of numerical values.