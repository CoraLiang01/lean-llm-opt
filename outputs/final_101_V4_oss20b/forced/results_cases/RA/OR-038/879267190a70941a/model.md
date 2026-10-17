**Mathematical Model**

Let  

* \(I\) be the set of vehicle types (the values of the column **VehicleType** in *capacity.csv*).  
* For each \(i\in I\) let  

  * \(c_i\) = daily inventory limit of type \(i\) (column **Capacity**).  
  * \(v_i\) = benefit coefficient of type \(i\) (column **Value**).  

Decision variable  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of vehicles of type \(i\) ordered per day.

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; c_i , &&\forall i\in I \\[2pt]
& \sum_{i\in I} x_i \;\le\; \sum_{i\in I} c_i .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(c_i\)   | `file_0_view_0` | `VehicleType` | `Capacity` |
| \(v_i\)   | `file_1_view_0` | `ProductName` | `Value` |

The index set \(I\) is obtained from the distinct values of the key columns in the two tables. No other parameters are required.