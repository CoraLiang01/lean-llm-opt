**Mathematical Model**

Let  

* \(I=\{1,\dots ,14\}\) be the set of display areas (DisplayID).  
* \(J=\{\text{Speedboat},\text{Fishing Boat},\dots ,\text{Skiff}\}\) be the set of vessel types (ProductName).  

Parameters  

* \(C_i\) – capacity of display area \(i\).  
* \(V_j\) – value of vessel type \(j\).  
* \(W_j\) – weight (size) of vessel type \(j\).  

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} V_j\,x_{ij}
\]

---

**Constraints**

1. **Capacity of each display area**

\[
\sum_{j\in J} W_j\,x_{ij}\;\le\; C_i \qquad \forall i\in I
\]

2. **Non‑negativity and integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source File | Column |
|-----------|-------------|--------|
| \(C_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv` | `Capacity` |
| \(V_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv` | `Value` |
| \(W_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv` | `Weight` |

*Index sets \(I\) and \(J\) are derived from the unique values in the `DisplayID` and `ProductName` columns of the respective files.*