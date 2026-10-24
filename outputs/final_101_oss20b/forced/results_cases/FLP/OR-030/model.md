**Mathematical Model**

Let  

* \(I\) – set of distinct *FDK57* car‑model records that satisfy the filter  
  (each record is identified by its source row number in the view).  

Parameters (all indexed by \(i\in I\))  

* \(r_i\) – revenue per unit of model \(i\) (from column **Revenue**)  
* \(d_i\) – demand quantity for model \(i\) (from column **Demand**)  
* \(inv_i\) – initial inventory of model \(i\) (from column **Initial Inventory**)  

Decision variable  

* \(x_i \ge 0\) – quantity of model \(i\) that the dealership plans to sell  

Objective  

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

Constraints  

1. **Inventory limits**  
   \[
   x_i \le inv_i , \qquad \forall i\in I
   \]

2. **Demand limits**  
   \[
   x_i \le d_i , \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_i \ge 0 , \qquad \forall i\in I
   \]

---

**Data Mapping Rules**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(r_i\) | `BigMartSales.csv` view `file_0_view_0` | `r_i = Revenue` (value in the row that satisfies the filter) |
| \(d_i\) | `BigMartSales.csv` view `file_0_view_0` | `d_i = Demand` (value in the row that satisfies the filter) |
| \(inv_i\) | `BigMartSales.csv` view `file_0_view_0` | `inv_i = Initial Inventory` (value in the row that satisfies the filter) |
| \(x_i\) | Decision variable | \(x_i\) is defined for each row \(i\) in the filtered view |

*Index set \(I\)* is the set of all source rows returned by the filter (8 rows in the current observation). Each row is uniquely identified by its `data_row_number` value; that identifier is used as the index \(i\). No other files or tables contribute data to this model.