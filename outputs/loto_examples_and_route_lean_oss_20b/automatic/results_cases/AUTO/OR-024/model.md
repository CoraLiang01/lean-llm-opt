**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* starts with **S700_**.  
- For each \(i\in I\) let  

\[
\begin{aligned}
\text{Revenue}_i &\;=\; \text{value in column “Revenue” of table } \texttt{file\_0\_view\_0} \text{ for product } i,\\
\text{Demand}_i   &\;=\; \text{value in column “Demand” of table } \texttt{file\_0\_view\_0} \text{ for product } i,\\
\text{Inv}_i      &\;=\; \text{value in column “Initial Inventory” of table } \texttt{file\_0\_view\_0} \text{ for product } i.
\end{aligned}
\]

Decision variable  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

(Units of product \(i\) to fulfill).

---

**Objective**

\[
\max \;\sum_{i\in I} \text{Revenue}_i \; x_i
\]

---

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; \text{Demand}_i &&\forall i\in I \quad (\text{cannot exceed demand})\\[4pt]
x_i &\;\le\; \text{Inv}_i     &&\forall i\in I \quad (\text{cannot exceed inventory})\\[4pt]
x_i &\;\ge\; 0                &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Index | Table ID          | Column Name          |
|-------------------|-------------------|----------------------|
| \(I\) (index)     | file_0_view_0     | Product Name         |
| Revenue\(_i\)     | file_0_view_0     | Revenue              |
| Demand\(_i\)      | file_0_view_0     | Demand               |
| InitialInventory\(_i\) | file_0_view_0 | Initial Inventory    |

All numerical values are taken directly from the rows returned by the filter on *Product Name* with prefix **S700_**.