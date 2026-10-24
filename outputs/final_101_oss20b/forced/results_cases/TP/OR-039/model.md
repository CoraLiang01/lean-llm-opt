**Mathematical Model**

Let  

* \(P=\{\text{Sedans},\text{SUVs},\text{Electric Vehicles},\text{Hybrid Vehicles},\text{Trucks},\text{Sports Cars},\text{Compact Cars},\text{Luxury Sedans},\text{Vans},\text{Pickup Trucks}\}\)  
* \(W=\{\text{Warehouse 1},\dots ,\text{Warehouse 10}\}\)

Parameters  

* \(v_p\) – value of one unit of product \(p\) (from *products.csv*)  
* \(c_w\) – capacity of warehouse \(w\) (from *capacity.csv*)

Decision variables  

* \(x_{p,w}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(p\) stored in warehouse \(w\)

\[
\begin{aligned}
\max\;& \sum_{p\in P}\sum_{w\in W} v_p\,x_{p,w} \\[4pt]
\text{s.t.}\;& \sum_{p\in P} x_{p,w} \;\le\; c_w , &&\forall\,w\in W \\[4pt]
& x_{p,w}\;\in\;\mathbb{Z}_{\ge 0}, &&\forall\,p\in P,\;w\in W
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(c_w\) | `file_0_view_0` | `Warehouse ID`, `Capacity` | `c[Warehouse ID] = Capacity` |
| \(v_p\) | `file_1_view_0` | `ProductName`, `Value` | `v[ProductName] = Value` |

*All indices \(w\) and \(p\) are taken directly from the distinct values of the corresponding columns in the source tables.*