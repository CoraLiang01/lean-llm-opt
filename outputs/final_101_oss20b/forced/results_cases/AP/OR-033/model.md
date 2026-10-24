**Mathematical Model**

Let  

* \(I\) be the set of product identifiers whose *Product Name* contains the word **“Baby”**.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**)  
  * \(d_i\) – demand quantity (from column **Demand**)  
  * \(s_i\) – initial inventory (from column **Initial Inventory**)  

Decision variable  

* \(x_i\) – number of units of product \(i\) to be sold (fulfilled).

---

### Objective
\[
\max \;\; \sum_{i\in I} r_i\,x_i
\]

### Constraints
\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; d_i &&\forall i\in I \quad &&\text{(cannot exceed demand)}\\[4pt]
&0 \;\le\; x_i \;\le\; s_i &&\forall i\in I \quad &&\text{(cannot exceed inventory)}\\[4pt]
&x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I \quad &&\text{(integer units)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Table ID | Column Name |
|-----------|--------------|----------|-------------|
| \(r_i\)   | Revenue      | file_0_view_0 | Revenue |
| \(d_i\)   | Demand       | file_0_view_0 | Demand |
| \(s_i\)   | Initial Inventory | file_0_view_0 | Initial Inventory |
| \(I\)     | Product Name | file_0_view_0 | Product Name (filter: contains “Baby”) |

All parameters are taken directly from the CSV file **EuropeSalesRecords.csv** after applying the filter that selects rows where *Product Name* contains the substring “Baby”. No numerical values are enumerated in the model; the mapping above links each symbolic parameter to its source column.