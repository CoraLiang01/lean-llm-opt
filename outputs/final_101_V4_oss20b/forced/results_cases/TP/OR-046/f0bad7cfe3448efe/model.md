**Mathematical Model**

Let  

* \(I=\{p_1,\dots ,p_{10}\}\) be the set of product names (rows of *products.csv*).  
* \(c\) be the overall stock capacity (value in *capacity.csv*).  

Parameters (indexed by \(i\in I\))  

* \(w_i\) – weight of product \(i\) (column *Weight*).  
* \(v_i\) – value (benefit) of product \(i\) (column *Value*).  

Decision variables  

* \(x_i \ge 0\) – number of units of product \(i\) ordered each day.  

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; c \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter / Variable | Source Table | Source Columns | Mapping |
|----------------------|--------------|----------------|---------|
| \(c\) | `file_0_view_0` | `Capacity` | `c = Capacity` |
| \(w_i\) | `file_1_view_0` | `Weight` | `w_{ProductName} = Weight` |
| \(v_i\) | `file_1_view_0` | `Value` | `v_{ProductName} = Value` |
| \(x_i\) | – | – | Decision variable, domain \(\mathbb{R}_+\) |

*Index set \(I\) is the set of all `ProductName` values returned from `file_1_view_0`.*