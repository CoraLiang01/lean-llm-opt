**Mathematical Model**

Let  

- \(I=\{S1,S2,\dots ,S10\}\) (warehouses)  
- \(J=\{C1,C2,\dots ,C10\}\) (retail stores)

Parameters  

- \(D_j\) : daily demand of store \(j\)  \(j\in J\)  
- \(S_i\) : daily supply capacity of warehouse \(i\) \(i\in I\)  
- \(C_{ij}\) : unit transportation cost from warehouse \(i\) to store \(j\) \(i\in I,\;j\in J\)

Decision variables  

- \(x_{ij}\ge 0\) : quantity shipped from warehouse \(i\) to store \(j\)

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} C_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=D_j ,\qquad \forall j\in J
   \]

2. **Supply capacity**  
   \[
   \sum_{j\in J} x_{ij}\le S_i ,\qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\ge 0 ,\qquad \forall i\in I,\;j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column(s) | Mapping Rule |
|-----------|--------------|------------------|--------------|
| \(D_j\) | `file_0_view_0` | `customer`, `demand` | For each record, set \(D_{\text{customer}} = \text{demand}\). |
| \(S_i\) | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | For each record, set \(S_{\text{Unnamed: 0}} = \text{supply_capacity}\). |
| \(C_{ij}\) | `file_2_view_0` | matrix of costs (rows `Unnamed: 0`, columns `C1`–`C10`) | For each row \(i\) and column \(j\), set \(C_{ij}\) to the cell value. |

All index sets \(I\) and \(J\) are taken from the full set of returned rows in the respective tables. No additional constraints or parameters are introduced.