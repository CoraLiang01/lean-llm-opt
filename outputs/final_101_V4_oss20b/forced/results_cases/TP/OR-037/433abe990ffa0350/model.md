**Mathematical model**

Let  

* \(I=\{\text{ProductName}\}\) – the set of all vehicle types (25 items).  
* \(p_i\) – profit (value) of one unit of product \(i\).  
* \(w_i\) – weight (space) required by one unit of product \(i\).  
* \(C\) – total inventory capacity.  
* \(x_i\) – number of units of product \(i\) ordered per day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0,\qquad x_i \in \mathbb{Z}\quad \forall i\in I
\end{aligned}
\]

---

**Data mapping**

| Data element | Source table | Column(s) | Mapping |
|--------------|--------------|-----------|---------|
| \(I\) | `file_1_view_0` | `ProductName` | index set |
| \(p_i\) | `file_1_view_0` | `Value` | parameter `profit` |
| \(w_i\) | `file_1_view_0` | `Weight` | parameter `weight` |
| \(C\) | `file_0_view_0` | `Capacity` | parameter `capacity` |
| \(x_i\) | – | – | decision variable |

*Table identifiers*  
- `file_0_view_0` – capacity data (single row).  
- `file_1_view_0` – product data (25 rows).  

All parameters are taken directly from the corresponding columns; no aggregation or transformation is applied. The variable domain is integer non‑negative as the decision is the number of vehicles. The objective sense is maximization, and the only constraint is the capacity limit.