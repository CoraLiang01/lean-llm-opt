**Mathematical Model**

Let  

- \(I\) be the set of produce types (rows of *products.csv*).  
- For each \(i\in I\):  
  - \(w_i\) – weight of one unit (kg) – from column **Weight**.  
  - \(v_i\) – benefit per unit – from column **Value**.  
- \(C\) – total inventory capacity (kg) – from column **Capacity** of *capacity.csv*.  
- Decision variable \(x_i\in\mathbb{Z}_{\ge 0}\) – number of units of type \(i\) ordered daily.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(C\) | Capacity | `file_0_view_0` | `Capacity` |
| \(w_i\) | Weight | `file_1_view_0` | `Weight` |
| \(v_i\) | Value | `file_1_view_0` | `Value` |
| \(I\) | Product names | `file_1_view_0` | `ProductName` |

All indices, parameters, and variables are defined solely from the returned rows of the specified tables. No additional data or assumptions are introduced.