**Mathematical Model**

Let  

* \(I=\{S1,S2,S3\}\) – set of warehouses (index taken from *fixed_cost.csv* column **Unnamed: 0**).  
* \(J=\{C1,C2,C3\}\) – set of customers (index taken from *demand.csv* column **customer**).  

Parameters (all taken directly from the supplied tables)

| Parameter | Source table | Column name | Description |
|-----------|--------------|-------------|-------------|
| \(f_i\) | *fixed_cost.csv* | **fixed_costs** | Fixed opening cost of warehouse \(i\) |
| \(d_j\) | *demand.csv* | **demand** | Demand of customer \(j\) |
| \(c_{ij}\) | *transportation_costs.csv* | **C1**, **C2**, **C3** | Transportation cost per unit from warehouse \(i\) to customer \(j\) |

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if warehouse \(i\) is opened, 0 otherwise.  
* \(x_{ij} \ge 0\) – quantity shipped from warehouse \(i\) to customer \(j\).

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} \;\ge\; d_j , && \forall j\in J \quad\text{(meet demand)}\\[4pt]
& x_{ij} \;\le\; d_j\,y_i , && \forall i\in I,\; j\in J \quad\text{(no shipment if warehouse closed)}\\[4pt]
& y_i \in \{0,1\}, && \forall i\in I\\
& x_{ij} \ge 0 , && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Symbol | Table ID | Column ID | Notes |
|--------|----------|-----------|-------|
| \(I\) | *file_1_view_0* | **Unnamed: 0** | Warehouse identifiers |
| \(J\) | *file_0_view_0* | **customer** | Customer identifiers |
| \(f_i\) | *file_1_view_0* | **fixed_costs** | Fixed cost per warehouse |
| \(d_j\) | *file_0_view_0* | **demand** | Demand per customer |
| \(c_{ij}\) | *file_2_view_0* | **C1**, **C2**, **C3** | Transportation cost matrix |

All parameters are mapped exactly to the columns shown; no values are hard‑coded in the model.