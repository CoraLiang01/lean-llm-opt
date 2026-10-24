**Mathematical Model**

Let  

* \(I\) – set of vehicle types (indexed by `VehicleID` from *capacity.csv*).  
* \(x_i\) – number of vehicles of type \(i\) ordered per day.  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(Capacity_i\) | Daily inventory limit for vehicle type \(i\) | `Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv` – column **Capacity** |
| \(Value_i\) | Benefit coefficient (value) for vehicle type \(i\) | `Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv` – column **Value** (matched on `ProductName` = `VehicleType`) |
| \(TotalCapacity\) | Total daily inventory capacity (sum of all \(Capacity_i\)) | Derived: \(TotalCapacity = \sum_{i\in I} Capacity_i\) |

Variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

**Objective**

\[
\max \;\; \sum_{i\in I} Value_i \, x_i
\]

**Constraints**

1. **Per‑type inventory limits**

\[
x_i \;\le\; Capacity_i \qquad \forall i\in I
\]

2. **Total inventory capacity**

\[
\sum_{i\in I} x_i \;\le\; TotalCapacity
\]

---

**Data Mapping Rules**

| Parameter | File Path | Column Name | Index Key |
|-----------|-----------|-------------|-----------|
| \(Capacity_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv` | `Capacity` | `VehicleID` |
| \(Value_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv` | `Value` | `ProductName` (matches `VehicleType` in capacity.csv) |
| \(TotalCapacity\) | – | – | – (derived as sum of all `Capacity_i`) |

*All other columns (e.g., `Weight`, `VehicleType`) are not used in the model.*