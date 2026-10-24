**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) – set of trucks  
* \(T=\{1,2,3,4\}\) – set of time periods  

Parameters (all taken from *parameters.csv*)  

| Symbol | Meaning | Data source |
|--------|---------|-------------|
| \(Q_i\) | Maximum capacity of truck \(i\) (kg) | `file_0_view_0`.`Q` |
| \(S_i\) | Startup cost of truck \(i\) | `file_0_view_0`.`S` |
| \(C_i\) | Unit transportation cost of truck \(i\) (per kg) | `file_0_view_0`.`C` |
| \(d_t\) | Customer demand in period \(t\) (kg) | `file_0_view_0`.`d1`–`d4` (identical for all trucks) |

Decision variables  

* \(y_{i,t}\in\{0,1\}\) – 1 if truck \(i\) is active in period \(t\), 0 otherwise  
* \(z_{i,t}\in\{0,1\}\) – 1 if truck \(i\) is started in period \(t\)  
* \(w_{i,t}\ge 0\) – weight transported by truck \(i\) in period \(t\) (kg)

Auxiliary constants  

* \(y_{i,0}=0,\; w_{i,0}=0\) for all \(i\)

---

### Objective  

\[
\min \;\sum_{i\in I}\sum_{t\in T}\Bigl(S_i\,z_{i,t}+C_i\,w_{i,t}\Bigr)
\]

---

### Constraints  

1. **Startup definition**  
   \[
   z_{i,t}\;\ge\; y_{i,t}-y_{i,t-1}\quad\forall i\in I,\;t\in T
   \]

2. **Minimum consecutive active periods** (cannot start in period 4)  
   \[
   y_{i,t+1}\;\ge\; z_{i,t}\quad\forall i\in I,\;t=1,2,3
   \]
   \[
   z_{i,4}=0\quad\forall i\in I
   \]

3. **Shutdown rule** – if a truck is shut down it must stay off for the next two periods  
   \[
   y_{i,t-1}-y_{i,t}+y_{i,t+1}\;\le\;1\quad\forall i\in I,\;t=2,3
   \]
   \[
   y_{i,t-1}-y_{i,t}+y_{i,t+2}\;\le\;1\quad\forall i\in I,\;t=2
   \]
   (For \(t=3\) the term \(y_{i,5}\) is omitted because period 5 does not exist.)

4. **Capacity limits**  
   \[
   w_{i,t}\;\le\; Q_i\,y_{i,t}\quad\forall i\in I,\;t\in T
   \]

5. **Weight‑change limits** (including transitions to/from zero load)  
   \[
   w_{i,t}-w_{i,t-1}\;\le\;300\quad\forall i\in I,\;t\in T
   \]
   \[
   w_{i,t-1}-w_{i,t}\;\le\;300\quad\forall i\in I,\;t\in T
   \]

6. **Demand satisfaction**  
   \[
   \sum_{i\in I} w_{i,t}\;\ge\; d_t\quad\forall t\in T
   \]

7. **Spare‑capacity buffer (10 % spare)**  
   \[
   \sum_{i\in I} w_{i,t}\;\le\;0.9\;\sum_{i\in I} Q_i\,y_{i,t}\quad\forall t\in T
   \]

8. **Variable domains**  
   \[
   y_{i,t}\in\{0,1\},\; z_{i,t}\in\{0,1\},\; w_{i,t}\ge 0\quad\forall i\in I,\;t\in T
   \]

---

### Data Mapping

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(Q_i\) | `file_0_view_0` | `Q` |
| \(S_i\) | `file_0_view_0` | `S` |
| \(C_i\) | `file_0_view_0` | `C` |
| \(d_t\) | `file_0_view_0` | `d1`, `d2`, `d3`, `d4` (identical across rows) |

All other constants (e.g., 300 kg change limit, 0.9 spare‑capacity factor) are fixed in the model.