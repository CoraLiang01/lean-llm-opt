## Mathematical Model

**Sets**  
\(I=\{B1,B2,B3,B4,B5,B6,B7,B8\}\) (depot candidates)  
\(J=\{Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10\}\) (service zones)

**Parameters**  
\(c_i\) – opening cost of depot \(i\) (\(i\in I\))  
\(a_{ij}\) – 1 if zone \(j\) is covered by depot \(i\), 0 otherwise (\(i\in I,\;j\in J\))

**Decision Variables**  
\(y_i\in\{0,1\}\) (1 if depot \(i\) is opened)

**Objective**  
\[
\min \sum_{i\in I} c_i\,y_i
\]

**Constraints**  

Coverage for each zone  
\[
\sum_{i\in I} a_{ij}\,y_i \;\ge\; 1 \qquad \forall j\in J
\]

Binary restriction  
\[
y_i\in\{0,1\}\qquad \forall i\in I
\]

---

## Data Mapping

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_i\) | `facility_sites.csv` (`OpeningCost`) | \(c_i =\) value in column **OpeningCost** for row with **Center** = \(i\) |
| \(a_{ij}\) | `facility_sites.csv` (`CoveredDistricts`) | \(a_{ij}=1\) if **Zone** \(j\) appears in the semicolon‑separated list in **CoveredDistricts** for depot \(i\); otherwise \(a_{ij}=0\) |
| Sets \(I,J\) | `facility_sites.csv` (`Center`) and `service_zones.csv` (`Zone`) | \(I\) = distinct values of **Center**; \(J\) = distinct values of **Zone** |

All other data (e.g., the list of zones in `CoveredDistricts`) are used only to construct the binary matrix \(a_{ij}\) as described. No additional numerical values are introduced beyond those present in the source tables.