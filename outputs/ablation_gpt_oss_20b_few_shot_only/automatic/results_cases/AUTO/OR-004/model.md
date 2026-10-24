**Mathematical Model**

Let  

* \(I=\{S1,S2,\dots ,S12\}\) (Distribution centers)  
* \(J=\{C1,C2,\dots ,C12\}\) (Customer groups)

Parameters  

* \(d_j\) – daily demand of customer \(j\) (from *customer_demand.csv*)  
* \(s_i\) – supply capacity of center \(i\) (from *supply_capacity.csv*)  
* \(c_{ij}\) – transportation cost per unit from center \(i\) to customer \(j\) (from *transportation_costs.csv*)

Decision variables  

\[
x_{ij}\;\ge 0 \qquad \forall i\in I,\; j\in J
\]

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
   \]

2. **Supply capacity**  
   \[
   \sum_{j\in J} x_{ij} \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J
   \]

---

**Data Mapping Rules**

| File Path | Column(s) | Parameter | Notes |
|-----------|-----------|-----------|-------|
| `Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv` | `customer`, `demand` | \(d_j\) | Each row gives demand for a customer \(j\). |
| `Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv` | `Unnamed: 0`, `supply_capacity` | \(s_i\) | Each row gives capacity for a distribution center \(i\). |
| `Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv` | `Unnamed: 0` (supplier), `C1`–`C12` | \(c_{ij}\) | Each row gives cost from supplier \(i\) to each customer \(j\). |

All identifiers (`S1`–`S12`, `C1`–`C12`) are taken verbatim from the source files; no aggregation or renaming is performed. The model uses the full set of entities present in the data.