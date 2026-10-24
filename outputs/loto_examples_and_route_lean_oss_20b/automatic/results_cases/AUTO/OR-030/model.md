**Mathematical Model**

Let  

- \(I\) be the set of car models that satisfy the filter  
  (`Product Name` starts with **FDK57**) in the view **file_0_view_0**.  

Parameters (indexed by \(i\in I\))  

- \(\text{Revenue}_i\) – revenue per unit of model \(i\)  
  (column **Revenue** of **file_0_view_0**).  
- \(\text{Demand}_i\) – maximum demand for model \(i\)  
  (column **Demand** of **file_0_view_0**).  
- \(\text{Inventory}_i\) – initial inventory of model \(i\)  
  (column **Initial Inventory** of **file_0_view_0**).  

Decision variables  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – units of model \(i\) to sell.  

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{Revenue}_i \, x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le \text{Demand}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{Inventory}_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(\text{Revenue}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\) | `file_0_view_0` | `Demand` |
| \(\text{Inventory}_i\) | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is implicitly defined as the set of row identifiers in **file_0_view_0** that satisfy the filter condition on `Product Name`.