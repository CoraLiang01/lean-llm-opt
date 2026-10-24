**Mathematical Model**

Let  

* \(I=\{i\mid \text{SKU}_i \text{ starts with “ZZ”}\}\) – set of ZZ‑products.  
* \(r_i\) – revenue per unit of product \(i\) (from column **Revenue**).  
* \(d_i\) – demand for product \(i\) (from column **Demand**).  
* \(s_i\) – initial inventory of product \(i\) (from column **Initial Inventory**).  
* \(x_i\) – number of units of product \(i\) to sell (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le s_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column | Filter |
|-----------|--------|----------|--------|--------|
| \(r_i\) | Revenue | file_0_view_0 | Revenue | SKU prefix “ZZ” |
| \(d_i\) | Demand | file_0_view_0 | Demand | SKU prefix “ZZ” |
| \(s_i\) | Initial Inventory | file_0_view_0 | Initial Inventory | SKU prefix “ZZ” |
| \(x_i\) | Decision variable | – | – | – |

*The filter condition “SKU prefix ‘ZZ’” selects the 5 rows shown in the observation. No other rows are included in the index set \(I\).*