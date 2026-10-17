**Mathematical Model**

Let  

\[
I=\{1,\dots ,8\}
\]

be the set of the eight *FDK57* car models that appear in the filtered view of **BigMartSales.csv**.  
For each \(i\in I\) define

* \(x_i\) – quantity of model \(i\) to sell (decision variable).  
* \(p_i\) – revenue per unit of model \(i\) (parameter).  
* \(d_i\) – maximum demand for model \(i\) (parameter).  
* \(s_i\) – initial inventory of model \(i\) (parameter).

All variables are non‑negative integers.

\[
\begin{aligned}
\max\;&\sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
             & 0 \le x_i \le s_i &&\forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column | Filter |
|-----------|--------|----------|--------|--------|
| \(p_i\) | Revenue | file_0_view_0 | Revenue | Product Name starts with “FDK57” |
| \(d_i\) | Demand | file_0_view_0 | Demand | Product Name starts with “FDK57” |
| \(s_i\) | Initial Inventory | file_0_view_0 | Initial Inventory | Product Name starts with “FDK57” |

The index set \(I\) is derived from the 8 rows returned by the filter on **Product Name**. Each row corresponds to a distinct \(i\in I\). No additional rows or entities are introduced.