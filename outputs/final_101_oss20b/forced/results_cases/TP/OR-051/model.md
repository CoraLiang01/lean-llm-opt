**Mathematical Model**

Let  

- \(I=\{1,2,\dots ,10\}\) (identifiers of cabinets)  
- \(J=\{1,2,\dots ,18\}\) (identifiers of coffee products)

Parameters  

- \(\text{cap}_i\) = capacity of cabinet \(i\)  (\(i\in I\))  
- \(\text{val}_j\) = value of one unit of product \(j\)  (\(j\in J\))  
- \(\text{wt}_j\) = weight of one unit of product \(j\)  (\(j\in J\))

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) = number of units of product \(j\) placed in cabinet \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J}\text{val}_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J}\text{wt}_j\,x_{ij}\;\le\;\text{cap}_i,
   && \forall\, i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0},
   && \forall\, i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(\text{cap}_i\) | `file_0_view_0` | `CabinetID`, `Capacity` | `cap[CabinetID] = Capacity` |
| \(\text{val}_j\) | `file_1_view_0` | `ProductName`, `Value` | `val[ProductName] = Value` |
| \(\text{wt}_j\) | `file_1_view_0` | `ProductName`, `Weight` | `wt[ProductName] = Weight` |

*All identifiers (`CabinetID`, `ProductName`) are taken directly from the respective tables; no additional processing or aggregation is performed.*