**Mathematical Model**

Let  

- \(I\) be the set of bread types (product names).  
- \(p_i\) be the expected profit per unit of product \(i\).  
- \(w_i\) be the storage weight per unit of product \(i\).  
- \(C\) be the total storage capacity.  
- \(x_i\) be the number of units of product \(i\) to order (integer, \(x_i\ge 0\)).

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(p_i\) | Expected profit | `file_1_view_0` | `Value` |
| \(w_i\) | Storage weight | `file_1_view_0` | `Weight` |
| \(C\)   | Storage capacity | `file_0_view_0` | `Capacity` |

All index sets, parameters, and variables are defined exactly as above. No numerical values are enumerated.