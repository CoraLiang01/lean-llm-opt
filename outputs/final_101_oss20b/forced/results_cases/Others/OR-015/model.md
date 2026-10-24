**Mathematical Model**

Let  

* \(I\) be the set of Aalop products selected by the filter  
  \[
  I=\{\,i \mid \text{Product Name}_i \text{ starts with “Aalop”}\,\}.
  \]

* Parameters (obtained from *RestaurantSalesreport.csv* after filtering)  

  * \(r_i\) – revenue per unit of product \(i\)  \((\text{Revenue})\)  
  * \(d_i\) – deterministic demand for product \(i\)  \((\text{Demand})\)  
  * \(s_i\) – initial inventory of product \(i\)  \((\text{Initial Inventory})\)

* Decision variables  

  * \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to sell

The objective is to maximize total revenue:

\[
\max_{x}\;\;\sum_{i\in I} r_i\,x_i
\]

subject to

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I &&\text{(cannot sell more than demand)}\\
x_i &\le s_i &&\forall i\in I &&\text{(cannot sell more than initial stock)}\\
x_i &\ge 0 &&\forall i\in I &&\text{(non‑negativity)}\\
x_i &\in \mathbb{Z} &&\forall i\in I &&\text{(integer units)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | file_0_view_0 | Revenue |
| \(d_i\) | Demand | file_0_view_0 | Demand |
| \(s_i\) | Initial Inventory | file_0_view_0 | Initial Inventory |

*The index set \(I\) is defined by the filter condition: “Product Name” starts with the prefix “Aalop”. Only rows satisfying this condition are included in the model.*