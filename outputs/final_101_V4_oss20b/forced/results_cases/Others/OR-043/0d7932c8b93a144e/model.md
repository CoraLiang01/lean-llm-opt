**Mathematical Model**

Let  

* \(I=\{1,\dots ,20\}\) be the set of drug products.  
* \(x_i\) – number of units of product \(i\) ordered each day.  

Parameters (obtained from the supplied tables)

| Parameter | Definition | Source |
|-----------|------------|--------|
| \(v_i\) | Benefit per unit of product \(i\) | `products.csv` column **Value** |
| \(w_i\) | Weight per unit of product \(i\) | `products.csv` column **Weight** |
| \(C\) | Total stock capacity | `capacity.csv` column **Capacity** |

The optimization problem is

\[
\begin{aligned}
\max_{x}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0,\;\; x_i \in \mathbb{Z}\quad \forall i\in I .
\end{aligned}
\]

---

**Data‑Mapping Rules**

* **Table `capacity.csv`** (`file_0_view_0`):  
  * Column **Capacity** → parameter \(C\).  
  * No filtering; the single row provides the capacity value.

* **Table `products.csv`** (`file_1_view_0`):  
  * Columns **ProductName**, **Value**, **Weight** → parameters \(v_i\), \(w_i\).  
  * Each row corresponds to an index \(i\in I\).  
  * No filtering; all 20 rows are included in the decision set.

* **Decision set**:  
  * Index set \(I\) is the set of all rows returned from `products.csv`.  
  * Variables \(x_i\) are defined for every \(i\in I\).

* **Variable domains**:  
  * \(x_i\) are non‑negative integers.  
  * No upper bounds are imposed beyond the capacity constraint.

* **Objective and constraints**:  
  * Objective uses the benefit parameters \(v_i\).  
  * Capacity constraint uses the weight parameters \(w_i\) and the capacity \(C\).

This formulation fully captures the pharmacy chain’s daily ordering decision problem as described.