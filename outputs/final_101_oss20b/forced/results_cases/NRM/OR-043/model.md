**Mathematical Model**

Let  

- \(I\) – index set of drug products.  
- \(v_i\) – benefit (value) of product \(i\).  
- \(w_i\) – weight (inventory consumption) of product \(i\).  
- \(C\) – total stock capacity.  
- \(x_i\) – number of units of product \(i\) ordered each day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(C\) | Capacity | `file_0_view_0` | `Capacity` |
| \(I\) | Product names | `file_1_view_0` | `ProductName` |
| \(v_i\) | Benefit | `file_1_view_0` | `Value` |
| \(w_i\) | Weight | `file_1_view_0` | `Weight` |

All parameters are taken directly from the specified columns; no additional data or enumeration is introduced.