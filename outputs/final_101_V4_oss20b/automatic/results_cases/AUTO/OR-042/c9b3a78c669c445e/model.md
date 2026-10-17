**Mathematical Model**

Let  

* \(I\) – index set of drug types (rows of *products.csv*).  
* \(x_i\) – integer decision variable, number of units of drug \(i\) ordered daily.  

Parameters (obtained from the CSV tables):

* \(v_i\) – benefit coefficient of drug \(i\) (column **Value** in *products.csv*).  
* \(w_i\) – weight of one unit of drug \(i\) (column **Weight** in *products.csv*).  
* \(C\) – overall inventory capacity (column **Capacity** in *capacity.csv*).  

The model is

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| Index set \(I\) | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding rows of the CSV files; no additional data are introduced.