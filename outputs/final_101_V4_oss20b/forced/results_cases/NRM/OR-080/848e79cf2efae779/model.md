**Index Sets**  
- \(K = \{1,2,\dots ,10\}\) (truck identifiers)  
- \(T = \{1,2,3,4\}\) (period indices)

**Parameters**  
- \(Q_k\) (maximum capacity of truck \(k\))  
- \(S_k\) (startup cost of truck \(k\))  
- \(C_k\) (unit transportation cost of truck \(k\))  
- \(d_t\) (customer demand in period \(t\))

**Variables**  
- \(x_{k,t}\) (weight transported by truck \(k\) in period \(t\)) \(\ge 0\)  
- \(y_{k,t}\) (active‑indicator of truck \(k\) in period \(t\)) \(\in\{0,1\}\)  
- \(z_{k,t}\) (startup‑indicator of truck \(k\) in period \(t\)) \(\in\{0,1\}\)

**Objective**  
\[
\min \;\sum_{k\in K}\sum_{t\in T}\bigl(S_k\,z_{k,t}+C_k\,x_{k,t}\bigr)
\]

**Constraints**

1. **Startup definition**  
   \[
   z_{k,1}=y_{k,1}\quad\forall k\in K
   \]
   \[
   z_{k,t}\ge y_{k,t}-y_{k,t-1}\quad\forall k\in K,\;t=2,3,4
   \]
   \[
   z_{k,t}\le y_{k,t}\quad\forall k\in K,\;t=2,3,4
   \]
   \[
   z_{k,t}\le 1-y_{k,t-1}\quad\forall k\in K,\;t=2,3,4
   \]
   \[
   z_{k,4}=0\quad\forall k\in K
   \]

2. **Minimum consecutive active periods**  
   \[
   y_{k,t}\le y_{k,t+1}\quad\forall k\in K,\;t=1,2,3
   \]
   \[
   y_{k,4}\le y_{k,3}\quad\forall k\in K
   \]

3. **Shutdown restriction**  
   \[
   y_{k,t-1}+y_{k,t+1}\le 1+y_{k,t}\quad\forall k\in K,\;t=2,3
   \]

4. **Capacity limits**  
   \[
   0\le x_{k,t}\le Q_k\,y_{k,t}\quad\forall k\in K,\;t\in T
   \]

5. **Weight change limits**  
   \[
   -300\le x_{k,1}\le 300\quad\forall k\in K
   \]
   \[
   -300\le x_{k,t}-x_{k,t-1}\le 300\quad\forall k\in K,\;t=2,3,4
   \]

6. **Demand satisfaction**  
   \[
   \sum_{k\in K}x_{k,t}\ge d_t\quad\forall t\in T
   \]

7. **Spare‑capacity buffer**  
   \[
   \sum_{k\in K}x_{k,t}\le 0.9\sum_{k\in K}Q_k\,y_{k,t}\quad\forall t\in T
   \]

8. **Variable domains**  
   \[
   x_{k,t}\ge 0,\;y_{k,t}\in\{0,1\},\;z_{k,t}\in\{0,1\}\quad\forall k\in K,\;t\in T
   \]

---

### Data Mapping

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(Q_k\) | `file_0_view_0` | `Q` |
| \(S_k\) | `file_0_view_0` | `S` |
| \(C_k\) | `file_0_view_0` | `C` |
| \(d_t\) | `file_0_view_0` | `d1`, `d2`, `d3`, `d4` |

The set \(K\) is derived from the distinct values in the `truck_id` column of `file_0_view_0`. The set \(T\) corresponds to the four period indices 1–4. No record values or counts are reproduced.