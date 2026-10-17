**Mathematical Model**

Let  

- \(I\) – set of display areas (indexed by the values in column **DisplayID** of *capacity.csv*).  
- \(J\) – set of boat types (indexed by the values in column **ProductName** of *products.csv*).  

Parameters  

- \(c_i\) – capacity of display area \(i\) (from column **Capacity**).  
- \(v_j\) – value of boat type \(j\) (from column **Value**).  
- \(w_j\) – size/weight of boat type \(j\) (from column **Weight**).  

Decision variables  

- \(x_{ij}\) – number of units of boat type \(j\) placed in display area \(i\).  
  \[
  x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
  \]

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

**Constraints**

1. **Capacity constraints (per display area)**  
   \[
   \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i \qquad \forall i\in I
   \]

2. **Non‑negativity and integrality**  
   \[
   x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `Value` |
| \(w_j\)   | `file_1_view_0` | `Weight` |

*Index sets*  
- \(I\) is the set of all distinct `DisplayID` values in `file_0_view_0`.  
- \(J\) is the set of all distinct `ProductName` values in `file_1_view_0`.  

No additional constraints or parameters are introduced beyond those specified in the query.