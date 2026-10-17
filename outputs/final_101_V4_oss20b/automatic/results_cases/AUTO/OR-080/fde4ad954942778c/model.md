## Mathematical Model

Let  

* \(T=\{1,\dots ,10\}\) – set of trucks  
* \(P=\{1,2,3,4\}\) – set of periods  

Parameters (all taken from **file_0_view_0**)

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(Q_t\) | maximum capacity of truck \(t\) (kg) | `Q` |
| \(S_t\) | startup cost of truck \(t\) (yuan) | `S` |
| \(C_t\) | unit transportation cost of truck \(t\) (yuan/kg) | `C` |
| \(d_p\) | customer demand in period \(p\) (kg) | `d1`–`d4` (same value for all trucks) |

Decision variables  

| Symbol | Type | Meaning |
|--------|------|---------|
| \(y_{t,p}\) | binary | 1 if truck \(t\) is active in period \(p\) |
| \(s_{t,p}\) | binary | 1 if truck \(t\) is started in period \(p\) |
| \(z_{t,p}\) | binary | 1 if truck \(t\) is shut‑down in period \(p\) |
| \(w_{t,p}\) | \(\ge 0\) continuous | weight transported by truck \(t\) in period \(p\) (kg) |

### Objective
\[
\min \;\sum_{t\in T}\sum_{p\in P}\Bigl(S_t\,s_{t,p}+C_t\,w_{t,p}\Bigr)
\]

### Constraints

1. **Truck state dynamics**  
   \[
   y_{t,p}=y_{t,p-1}+s_{t,p}-z_{t,p}\qquad\forall t\in T,\;p\in P
   \]
   with \(y_{t,0}=0\) for all \(t\).

2. **Minimum‑up time (2 periods)**  
   \[
   s_{t,p}\le y_{t,p+1}\qquad\forall t\in T,\;p=1,2,3
   \]
   (implies \(y_{t,p+1}=1\) whenever \(s_{t,p}=1\)).  
   \[
   s_{t,4}=0\qquad\forall t\in T
   \]

3. **Minimum‑down time (2 periods)**  
   \[
   z_{t,p}\le 1-y_{t,p+1}\qquad\forall t\in T,\;p=1,2,3
   \]
   \[
   z_{t,p}\le 1-y_{t,p+2}\qquad\forall t\in T,\;p=1,2
   \]
   (ensures inactivity in the two periods following a shutdown).

4. **Weight limits**  
   \[
   0\le w_{t,p}\le Q_t\,y_{t,p}\qquad\forall t\in T,\;p\in P
   \]

5. **Weight‑change limits**  
   \[
   w_{t,p}-w_{t,p-1}\le 300\qquad\forall t\in T,\;p=1,2,3,4
   \]
   \[
   w_{t,p-1}-w_{t,p}\le 300\qquad\forall t\in T,\;p=1,2,3,4
   \]
   with \(w_{t,0}=0\) for all \(t\).

6. **Demand satisfaction**  
   \[
   \sum_{t\in T} w_{t,p}\;\ge\; d_p\qquad\forall p\in P
   \]

7. **Spare‑capacity buffer (10 %)**  
   \[
   \sum_{t\in T} w_{t,p}\;\le\;0.9\;\sum_{t\in T} Q_t\,y_{t,p}\qquad\forall p\in P
   \]

8. **Variable domains**  
   \[
   y_{t,p},\,s_{t,p},\,z_{t,p}\in\{0,1\}\qquad\forall t\in T,\;p\in P
   \]
   \[
   w_{t,p}\ge 0\qquad\forall t\in T,\;p\in P
   \]

---

## Data Mapping

| Parameter | Source table | Source column | Notes |
|-----------|--------------|---------------|-------|
| \(T\) | `file_0_view_0` | `truck_id` | All 10 rows |
| \(P\) | – | – | Period indices 1–4 |
| \(Q_t\) | `file_0_view_0` | `Q` | Capacity of truck \(t\) |
| \(S_t\) | `file_0_view_0` | `S` | Startup cost of truck \(t\) |
| \(C_t\) | `file_0_view_0` | `C` | Unit transport cost of truck \(t\) |
| \(d_p\) | `file_0_view_0` | `d1`–`d4` | Demand in period \(p\) (identical across rows) |

All other data (e.g., demand values) are derived directly from the corresponding columns in the source file. No additional rows or columns are introduced beyond those explicitly listed.