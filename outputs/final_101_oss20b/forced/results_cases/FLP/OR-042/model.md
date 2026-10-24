**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of drug types (product names).  
- \(v_i\) = benefit coefficient (value) of product \(i\).  
- \(w_i\) = weight of one unit of product \(i\).  
- \(C\) = overall inventory capacity.  
- \(x_i\) = number of units of product \(i\) ordered daily.

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
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(C\)   | `file_0_view_0` | `Capacity` |

*Index set \(I\) is derived from the distinct `ProductName` values in `file_1_view_0`.*