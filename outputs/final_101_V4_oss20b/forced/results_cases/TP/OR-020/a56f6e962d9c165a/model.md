**Mathematical Model**

Let  

* \(I\) – set of all products (identified by *Product Name*).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – maximum demand (from column **Demand**).  
  * \(s_i\) – available initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad\text{(units of product }i\text{ to sell)}
\]

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

1. **Demand limit**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 \qquad \forall i\in I
   \]

---

**Data Mapping Rules**

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(I\) (index set) | `file_0_view_0` | `Product Name` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

*All values are taken directly from the corresponding columns of the table `file_0_view_0` without any aggregation or transformation.*