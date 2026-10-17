## Mathematical Model

Let  

* \(I\) be the set of Fashion products obtained from the filtered rows of  
  **SupermarketSales.csv**.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (RMB)  (from column **Revenue**)  
  * \(d_i\) – demand (units)        (from column **Demand**)  
  * \(s_i\) – initial inventory (units) (from column **Initial Inventory**)  

Decision variable  

* \(x_i\) – number of units of product \(i\) to sell (integer, \(x_i\ge 0\)).

\[
\begin{aligned}
\max\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } &x_i \;\le\; d_i &&\forall i\in I \\[2pt]
             &x_i \;\le\; s_i &&\forall i\in I \\[2pt]
             &x_i \in \mathbb{Z}_+ &&\forall i\in I
\end{aligned}
\]

---

## Data Mapping Rules

| Parameter | Source | Column | Filter |
|-----------|--------|--------|--------|
| \(I\) | **SupermarketSales.csv** | *Product Name* | `contains "Fashion"` |
| \(r_i\) | **SupermarketSales.csv** | **Revenue** | same filter as above |
| \(d_i\) | **SupermarketSales.csv** | **Demand** | same filter as above |
| \(s_i\) | **SupermarketSales.csv** | **Initial Inventory** | same filter as above |

*Only the rows that satisfy the filter (`Product Name` contains “Fashion”) are used to define the index set \(I\) and the corresponding parameter values.*