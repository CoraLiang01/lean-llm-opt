**Mathematical Model**

Let  

- \(I=\{\text{Spinach},\text{Shiitake Mushrooms},\text{Apples},\text{Carrots},\text{Basil},\text{Potatoes},\text{Green Beans},\text{Blueberries},\text{Oranges},\text{Watermelons}\}\)  
- \(x_i\in \mathbb{Z}_{\ge 0}\) quantity of product \(i\) ordered daily  
- \(w_i\) weight of one unit of product \(i\) (\(w_i\ge 0\))  
- \(v_i\) benefit (value) of one unit of product \(i\) (\(v_i\ge 0\))  
- \(C\) overall inventory capacity (\(C\ge 0\))

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(v_i\) | `file_1_view_0` | `Value` |
| Index set \(I\) | `file_1_view_0` | `ProductName` |

All values are taken directly from the corresponding CSV files; no additional processing or aggregation is performed.