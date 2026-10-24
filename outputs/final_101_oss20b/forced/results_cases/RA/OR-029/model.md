**Mathematical Model**

Let  

* \(I\) – index set of all products whose *Product Name* contains “FAUX”.  
* For each \(i\in I\)  
  * \(p_i\) – revenue per unit (from column **Revenue**)  
  * \(I_i\) – initial inventory (from column **Initial Inventory**)  
  * \(D_i\) – demand (from column **Demand**)  

Decision variable  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

### Objective  

\[
\max \;\; \sum_{i\in I} p_i\,x_i
\]

---

### Constraints  

1. **Inventory limit**  
   \[
   x_i \;\le\; I_i \qquad \forall i\in I
   \]

2. **Demand limit**  
   \[
   x_i \;\le\; D_i \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 \qquad \forall i\in I
   \]

---

**Data Mapping**

| Index / Parameter | Source Table | Source Column |
|-------------------|--------------|---------------|
| \(I\) (product names) | `file_0_view_0` | `Product Name` |
| \(p_i\) | `file_0_view_0` | `Revenue` |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` |
| \(D_i\) | `file_0_view_0` | `Demand` |

All numerical values are taken directly from the corresponding rows of the filtered view `file_0_view_0`. No additional data or assumptions are introduced.