**Mathematical Model**

Let  

* \(I\) be the set of products whose *Sub Category* contains the string “Organ”.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**)  
  * \(d_i\) – demand quantity (from column **Demand**)  
  * \(s_i\) – initial inventory (from column **Initial Inventory**)  

Decision variable  

* \(x_i\) – number of units of product \(i\) to be sold (fulfilled).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le s_i && \forall i\in I \\[2pt]
& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column | Filter |
|-----------|--------|----------|--------|--------|
| \(r_i\) | Revenue | file_0_view_0 | Revenue | Sub Category contains “Organ” |
| \(d_i\) | Demand | file_0_view_0 | Demand | Sub Category contains “Organ” |
| \(s_i\) | Initial Inventory | file_0_view_0 | Initial Inventory | Sub Category contains “Organ” |

The index set \(I\) is defined by the rows that satisfy the filter condition on *Sub Category*. No other rows are included.