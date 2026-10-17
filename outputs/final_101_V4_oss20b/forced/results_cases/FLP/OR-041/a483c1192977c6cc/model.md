**Mathematical Model**

Let  

- \(I\) be the set of real‑estate areas (ProductName).  
- \(V_i\) be the benefit per unit of development in area \(i\).  
- \(W_i\) be the capacity consumption per unit of development in area \(i\).  
- \(C\) be the overall development capacity.  
- \(x_i\) be the scale of development per day in area \(i\) (continuous, \(x_i\ge 0\)).

\[
\begin{aligned}
\max\;& \sum_{i\in I} V_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} W_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 ,\qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(V_i\) | benefit per unit | `file_1_view_0` | `Value` |
| \(W_i\) | capacity consumption per unit | `file_1_view_0` | `Weight` |
| \(C\) | overall capacity | `file_0_view_0` | `Capacity` |
| \(I\) | set of areas | `file_1_view_0` | `ProductName` |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.