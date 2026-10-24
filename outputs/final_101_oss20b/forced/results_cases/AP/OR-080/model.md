**Mathematical Model**

Let  

* \(T=\{1,\dots ,10\}\) – set of trucks  
* \(P=\{1,2,3,4\}\) – set of time periods  

Parameters (all taken from *parameters.csv*)  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(Q_t\) | maximum capacity of truck \(t\) (kg) | column **Q** |
| \(S_t\) | startup cost of truck \(t\) | column **S** |
| \(C_t\) | unit transportation cost of truck \(t\) (per kg) | column **C** |
| \(D_p\) | customer demand in period \(p\) (kg) | columns **d1**–**d4** (identical for all trucks) |

Decision variables  

* \(a_{t,p}\in\{0,1\}\) = 1 if truck \(t\) is active in period \(p\)  
* \(y_{t,p}\in\{0,1\}\) = 1 if truck \(t\) is started in period \(p\)  
* \(z_{t,p}\in\{0,1\}\) = 1 if truck \(t\) is shut down in period \(p\)  
* \(w_{t,p}\ge 0\) = weight transported by truck \(t\) in period \(p\) (kg)

---

### Objective

\[
\min \;\; \sum_{t\in T}\sum_{p\in P}\Bigl(C_t\,w_{t,p}\;+\;S_t\,y_{t,p}\Bigr)
\]

---

### Constraints  

1. **Activation dynamics**  
   \[
   a_{t,1}=y_{t,1}\qquad\forall t\in T
   \]
   \[
   a_{t,p}=a_{t,p-1}+y_{t,p}-z_{t,p}\qquad\forall t\in T,\;p=2,\dots ,4
   \]

2. **Startup restrictions**  
   * No start in period 4  
     \[
     y_{t,4}=0\qquad\forall t\in T
     \]
   * Once started, active for at least two consecutive periods  
     \[
     y_{t,p}\le a_{t,p+1}\qquad\forall t\in T,\;p=1,2,3
     \]
   * Cannot start while already active  
     \[
     y_{t,p}\le 1-a_{t,p-1}\qquad\forall t\in T,\;p=2,\dots ,4
     \]

3. **Shutdown restrictions**  
   * If shut down, inactive in that and next period, and cannot restart before period \(p+2\)  
     \[
     z_{t,p}\le 1-a_{t,p}\qquad\forall t\in T,\;p=1,\dots ,4
     \]
     \[
     z_{t,p}\le 1-a_{t,p+1}\qquad\forall t\in T,\;p=1,\dots ,3
     \]
     \[
     y_{t,p+2}\le 1-z_{t,p}\qquad\forall t\in T,\;p=1,\dots ,2
     \]

4. **Weight limits**  
   \[
   0\le w_{t,p}\le Q_t\,a_{t,p}\qquad\forall t\in T,\;p\in P
   \]

5. **Weight change limits**  
   \[
   |w_{t,p}-w_{t,p-1}|\le 300\qquad\forall t\in T,\;p=2,\dots ,4
   \]
   (with \(w_{t,0}=0\))

6. **Demand satisfaction**  
   \[
   \sum_{t\in T} w_{t,p}\;\ge\; D_p\qquad\forall p\in P
   \]

7. **Spare‑capacity buffer**  
   \[
   \sum_{t\in T} w_{t,p}\;\le\;0.9\;\sum_{t\in T} Q_t\,a_{t,p}\qquad\forall p\in P
   \]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "index_sets": {
    "T": {"column": "truck_id", "type": "integer"},
    "P": {"values": [1,2,3,4], "type": "integer"}
  },
  "parameters": {
    "Q": {"column": "Q", "type": "float"},
    "S": {"column": "S", "type": "float"},
    "C": {"column": "C", "type": "float"},
    "D": {
      "p1": {"column": "d1", "type": "float"},
      "p2": {"column": "d2", "type": "float"},
      "p3": {"column": "d3", "type": "float"},
      "p4": {"column": "d4", "type": "float"}
    }
  }
}
```

The model above fully captures the logistics scheduling problem described in the query, using only the data provided in *parameters.csv* and the specified constraints.