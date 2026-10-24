**Mathematical Model**

Let  

- \(I\) be the set of produce types.  
- \(C\) be the overall inventory capacity.  
- \(w_i\) be the weight of one unit of type \(i\).  
- \(v_i\) be the benefit (value) of one unit of type \(i\).  
- \(x_i\) be the number of units of type \(i\) ordered daily.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(I\) | `file_1_view_0` | `ProductName` |

All indices and parameters are defined solely from the supplied tables; no numerical values are enumerated.