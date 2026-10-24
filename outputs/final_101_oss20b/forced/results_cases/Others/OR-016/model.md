**Mathematical Model**

Let  

* \(I=\{1,\dots ,35\}\) be the set of product records in *RetailSalesDataset.csv*.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (RMB)  (from column **Revenue**)  
  * \(d_i\) – demand quantity (units) (from column **Demand**)  
  * \(v_i\) – available inventory (units) (from column **Initial Inventory**)  

Decision variable  

* \(x_i\) – quantity of product \(i\) to sell (continuous, units).

---

**Objective**

\[
\max \; Z = \sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

1. **Demand satisfaction**  
   \[
   0 \le x_i \le d_i \qquad \forall i\in I
   \]

2. **Inventory availability**  
   \[
   0 \le x_i \le v_i \qquad \forall i\in I
   \]

(Constraints 1 and 2 together imply \(0\le x_i \le \min\{d_i,v_i\}\).)

---

**Variable Domains**

\[
x_i \ge 0 \quad \text{and continuous} \qquad \forall i\in I
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(r_i\) | *RetailSalesDataset.csv* (view `file_0_view_0`) | `Revenue` | `r_i = Revenue` for record \(i\) |
| \(d_i\) | *RetailSalesDataset.csv* (view `file_0_view_0`) | `Demand` | `d_i = Demand` for record \(i\) |
| \(v_i\) | *RetailSalesDataset.csv* (view `file_0_view_0`) | `Initial Inventory` | `v_i = Initial Inventory` for record \(i\) |
| \(x_i\) | Decision variable | – | `x_i` is the quantity sold of product \(i\) |

*No additional filtering or aggregation is applied; all 35 product records are included in the index set \(I\).*

---

**Notes**

* All variables are continuous; integer or binary decisions are not required by the problem statement.  
* The model respects each product’s individual demand and inventory limits, ensuring that no product is sold beyond what is demanded or available.  
* The objective is a simple revenue maximization, consistent with the retailer’s goal of maximizing total revenue.