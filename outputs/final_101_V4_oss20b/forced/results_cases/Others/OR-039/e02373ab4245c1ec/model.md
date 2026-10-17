## Mathematical Model

Let  

* \(P=\{\text{Sedans},\text{SUVs},\text{Electric Vehicles},\text{Hybrid Vehicles},\text{Trucks},\text{Sports Cars},\text{Compact Cars},\text{Luxury Sedans},\text{Vans},\text{Pickup Trucks}\}\)  
* \(W=\{\text{Warehouse 1},\text{Warehouse 2},\dots ,\text{Warehouse 10}\}\)

Parameters  

* \(v_p\) – benefit (value) of one unit of product \(p\) (from **products.csv**)  
* \(w_p\) – weight of one unit of product \(p\) (from **products.csv**)  
* \(C_w\) – capacity of warehouse \(w\) (from **capacity.csv**)

Decision variables  

* \(x_{p,w}\in\mathbb{Z}_{\ge 0}\) – number of units of product \(p\) stored in warehouse \(w\)

**Objective**

\[
\max \;\sum_{p\in P}\sum_{w\in W} v_p\,x_{p,w}
\]

**Constraints**

\[
\sum_{p\in P} w_p\,x_{p,w}\;\le\; C_w \qquad \forall\,w\in W
\]

\[
x_{p,w}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall\,p\in P,\;w\in W
\]

---

## Data Mapping

| Parameter | Source Table | Source Columns | Mapping Rule |
|-----------|--------------|----------------|--------------|
| \(C_w\) | `file_0_view_0` | `Warehouse ID`, `Capacity` | For each record, set \(C_{\text{Warehouse ID}} = \text{Capacity}\) |
| \(v_p\) | `file_1_view_0` | `ProductName`, `Value` | For each record, set \(v_{\text{ProductName}} = \text{Value}\) |
| \(w_p\) | `file_1_view_0` | `ProductName`, `Weight` | For each record, set \(w_{\text{ProductName}} = \text{Weight}\) |

All indices \(p\) and \(w\) are taken from the full set of records returned in the observation; no additional filtering is applied. The model uses only the parameters and variables defined above.