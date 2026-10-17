**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{subject to}\quad
& 0 \le x_i \le \text{dem}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{inv}_i && \forall i\in I \\[4pt]
& x_i \;\text{continuous} && \forall i\in I
\end{aligned}
\]

*Index set*  
\(I\) – set of all products whose *Product Name* starts with “Baby” (filtered view).

*Parameters*  
\(\text{rev}_i\) – revenue per unit of product \(i\) (from column **Revenue**).  
\(\text{dem}_i\) – deterministic demand for product \(i\) (from column **Demand**).  
\(\text{inv}_i\) – initial inventory of product \(i\) (from column **Initial Inventory**).

*Decision variables*  
\(x_i\) – number of units of product \(i\) to be sold (continuous, non‑negative).

---

**Data Mapping Rules**

| Table ID          | Column Name          | Parameter Name |
|-------------------|----------------------|----------------|
| file_0_view_0     | Revenue              | rev_i          |
| file_0_view_0     | Demand               | dem_i          |
| file_0_view_0     | Initial Inventory    | inv_i          |

*Note:* The table `file_0_view_0` is the filtered view of `Salesdata.csv` where `Product Name` starts with “Baby”. Each row in this view corresponds to one element of the index set \(I\). The mapping above binds each column to the corresponding model parameter.