**Mathematical Model**

Let  

- \(I\) be the set of all dairy products (identified by the value in column **Full_Product_Name**).  

Parameters (indexed by \(i\in I\))  

\[
\begin{aligned}
\text{Revenue}_i &\;=\; \text{value in column } \text{Revenue},\\
\text{Demand}_i  &\;=\; \text{value in column } \text{Demand},\\
\text{Inventory}_i &\;=\; \text{value in column } \text{Initial Inventory}.
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} \text{Revenue}_i \; x_i
\]

Constraints  

\[
\begin{aligned}
x_i &\;\le\; \text{Demand}_i &&\forall i\in I,\\
x_i &\;\le\; \text{Inventory}_i &&\forall i\in I,\\
x_i &\;\ge\; 0 &&\forall i\in I.
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Set | Table ID | Key Column | Value Column |
|-----------------|----------|------------|--------------|
| \(I\) (products) | `file_0_view_0` | `Full_Product_Name` | – |
| \(\text{Revenue}_i\) | `file_0_view_0` | `Full_Product_Name` | `Revenue` |
| \(\text{Demand}_i\) | `file_0_view_0` | `Full_Product_Name` | `Demand` |
| \(\text{Inventory}_i\) | `file_0_view_0` | `Full_Product_Name` | `Initial Inventory` |

All other model components are defined symbolically and do not require additional data mappings.