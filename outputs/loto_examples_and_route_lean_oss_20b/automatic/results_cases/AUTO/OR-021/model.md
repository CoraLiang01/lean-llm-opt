**Mathematical Model**

Let  

* \(I\) – set of products (indexed by the values in the column **Product Name**).  
* \(x_i\) – number of units of product \(i\) that will be fulfilled.  

Parameters (all indexed by \(i\))  

* \(\text{Revenue}_i\) – revenue per unit of product \(i\).  
* \(\text{Demand}_i\) – deterministic demand for product \(i\).  
* \(\text{Inv}_i\) – initial inventory available for product \(i\).  

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le \text{Demand}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{Inv}_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(\text{Revenue}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\) | `file_0_view_0` | `Demand` |
| \(\text{Inv}_i\) | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | `file_0_view_0` | `Product Name` |

All values are taken directly from the runtime records of `Salesofsummerclothes.csv`.