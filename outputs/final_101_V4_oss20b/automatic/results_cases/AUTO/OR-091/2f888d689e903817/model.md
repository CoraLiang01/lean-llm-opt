**Mathematical Model**

Let  

* \(I=\{i\mid\text{course }i\text{ is in the Operations Research discipline}\}\)  
  (the 7 courses returned in the observation).  

* Parameters (all taken from *courses_42.csv* – table_id `file_0_view_0`):  

  * \(c_i\) – credits of course \(i\) (column **credits**)  
  * \(p_i\) – interest‑points of course \(i\) (column **interest_points**)  

* Decision variables:  

  * \(x_i\in\{0,1\}\) – 1 if course \(i\) is selected, 0 otherwise.  

**Objective**

\[
\max \sum_{i\in I} p_i\,x_i
\]

**Constraints**

\[
\sum_{i\in I} c_i\,x_i \;\le\; 20
\]

\[
x_i \in \{0,1\}\qquad \forall i\in I
\]

---

**Data Mapping**

| Index | Parameter | Source Table | Source Column | Mapping |
|-------|-----------|--------------|---------------|---------|
| \(i\) | \(c_i\) | `file_0_view_0` | `credits` | `c_i = value of column credits for course i` |
| \(i\) | \(p_i\) | `file_0_view_0` | `interest_points` | `p_i = value of column interest_points for course i` |
| \(i\) | \(x_i\) | – | – | binary decision variable |

*The index set \(I\) is defined by the filter condition `discipline = 'Operations Research'` applied to `file_0_view_0`.*