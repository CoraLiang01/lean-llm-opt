**Mathematical Model**

```math
\begin{aligned}
\text{Sets} \quad & W \;=\; \{\text{Warehouse 1},\dots,\text{Warehouse 10}\},\\
                  & P \;=\; \{\text{Sedans},\text{SUVs},\text{Electric Vehicles},
                               \text{Hybrid Vehicles},\text{Trucks},
                               \text{Sports Cars},\text{Compact Cars},
                               \text{Luxury Sedans},\text{Vans},
                               \text{Pickup Trucks}\}.\\[4pt]

\text{Parameters} \quad & \text{cap}_w \;\; (w\in W) \quad\text{: capacity of warehouse }w,\\
                         & \text{val}_i \;\; (i\in P) \quad\text{: benefit coefficient of product }i.\\[4pt]

\text{Decision Variables} \quad & x_{i,w}\;\in\;\mathbb{Z}_{+}\quad (i\in P,\;w\in W)
   \;\;=\;\text{number of units of product }i\text{ stored in warehouse }w.\\[4pt]

\text{Objective} \quad & \max \;\sum_{i\in P}\sum_{w\in W}\text{val}_i\,x_{i,w}.\\[4pt]

\text{Constraints} \quad
& \sum_{i\in P} x_{i,w}\;\le\;\text{cap}_w, \qquad \forall\,w\in W,\\
& x_{i,w}\;\ge\;0,\;\;x_{i,w}\in\mathbb{Z}, \qquad \forall\,i\in P,\;w\in W.
\end{aligned}
```

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| `cap_w`   | `file_0_view_0` | `Warehouse ID` | `Capacity` |
| `val_i`   | `file_1_view_0` | `ProductName` | `Value` |

*The `Weight` column in `products.csv` is not used in this formulation.*