**Mathematical Model**

Let  

* \(I\) – index set of vehicle types (one element per row of *products.csv*).  
* \(C\) – total inventory capacity (value from *capacity.csv*).  

Parameters  

* \(v_i\) – benefit coefficient of vehicle type \(i\) (column **Value**).  
* \(w_i\) – space/weight requirement of vehicle type \(i\) (column **Weight**).  

Decision variables  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of vehicle type \(i\) ordered daily.  

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` | `v[ProductName] = Value` |
| \(w_i\) | `file_1_view_0` | `Weight` | `w[ProductName] = Weight` |
| Index set \(I\) | `file_1_view_0` | `ProductName` | `I = {ProductName}` |

No other parameters or constraints are required.