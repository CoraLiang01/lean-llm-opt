**Mathematical Model**

Let  

- \(I\) be the set of all product rows in *RetailSalesDataset.csv* (35 products).  
- For each \(i\in I\):  
  - \(r_i\) – revenue per unit (column **Revenue**).  
  - \(d_i\) – demand quantity (column **Demand**).  
  - \(s_i\) – initial inventory (column **Initial Inventory**).  

Decision variable  

- \(x_i\) – quantity of product \(i\) to sell (continuous, non‑negative).

\[
\begin{aligned}
\max\;& Z = \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le \min\{d_i,\; s_i\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Parameter | Mapping |
|----------|--------|-----------|---------|
| file_0_view_0 | Product Name | – | Identifier for index \(i\) |
| file_0_view_0 | Revenue | \(r_i\) | \(r_i =\) value in **Revenue** column |
| file_0_view_0 | Demand | \(d_i\) | \(d_i =\) value in **Demand** column |
| file_0_view_0 | Initial Inventory | \(s_i\) | \(s_i =\) value in **Initial Inventory** column |

No additional constraints or parameters are required beyond those listed. The model uses the full set of products returned by the observation.