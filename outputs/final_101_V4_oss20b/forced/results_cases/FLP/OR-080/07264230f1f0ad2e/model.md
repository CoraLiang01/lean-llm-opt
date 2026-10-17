**Decision Variables**

| Symbol | Meaning | Domain |
|--------|---------|--------|
| \(y_{i,t}\) | Truck \(i\) is active in period \(t\) | \(\{0,1\}\) |
| \(w_{i,t}\) | Weight (kg) transported by truck \(i\) in period \(t\) | \(\mathbb{R}_{+}\) |
| \(s_{i,t}\) | Truck \(i\) is started in period \(t\) | \(\{0,1\}\) |

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(I\) | Set of trucks \(\{1,\dots,10\}\) | implicit |
| \(T\) | Set of periods \(\{1,2,3,4\}\) | implicit |
| \(Q_i\) | Max capacity of truck \(i\) (kg) | `file_0_view_0`, column **Q** |
| \(S_i\) | Startup cost of truck \(i\) | `file_0_view_0`, column **S** |
| \(C_i\) | Unit transport cost of truck \(i\) (per kg) | `file_0_view_0`, column **C** |
| \(d_t\) | Customer demand in period \(t\) (kg) | `file_0_view_0`, column **d1**–**d4** (any row, e.g., row 0) |
| \(\Delta\) | Max change in weight between consecutive periods | constant = 300 |
| \(\beta\) | Spare‑capacity factor | constant = 0.9 |

**Auxiliary**

- \(y_{i,0}=0\) for all \(i\) (trucks are off before period 1).

---

### Objective

\[
\min \;\sum_{i\in I}\sum_{t\in T}\bigl(C_i\,w_{i,t}+S_i\,s_{i,t}\bigr)
\]

---

### Constraints

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} w_{i,t}\;\ge\; d_t \qquad \forall\,t\in T
   \]

2. **Spare‑capacity buffer**  
   \[
   \sum_{i\in I} w_{i,t}\;\le\; \beta \sum_{i\in I} Q_i\,y_{i,t}
   \qquad \forall\,t\in T
   \]

3. **Capacity limit**  
   \[
   w_{i,t}\;\le\; Q_i\,y_{i,t} \qquad \forall\,i\in I,\;t\in T
   \]

4. **Weight‑change limit**  
   \[
   w_{i,1}\;\le\;\Delta \qquad \forall\,i\in I
   \]
   \[
   w_{i,t}-w_{i,t-1}\;\le\;\Delta,\;
   w_{i,t-1}-w_{i,t}\;\le\;\Delta
   \qquad \forall\,i\in I,\;t=2,3,4
   \]

5. **Startup definition**  
   \[
   s_{i,t}\;\ge\; y_{i,t}-y_{i,t-1} \qquad \forall\,i\in I,\;t\in T
   \]
   \[
   s_{i,t}\;\le\; y_{i,t},\;
   s_{i,t}\;\le\; 1-y_{i,t-1} \qquad \forall\,i\in I,\;t\in T
   \]

6. **No start in period 4**  
   \[
   s_{i,4}=0 \qquad \forall\,i\in I
   \]

7. **Minimum consecutive active periods**  
   \[
   y_{i,t+1}\;\ge\; y_{i,t}-y_{i,t-1}
   \qquad \forall\,i\in I,\;t=1,2,3
   \]

8. **Shutdown rule**  
   \[
   y_{i,t+1}\;\le\; y_{i,t} + (1-y_{i,t-1})
   \qquad \forall\,i\in I,\;t=1,2,3
   \]

9. **Domains**  
   \[
   y_{i,t},\,s_{i,t}\in\{0,1\},\quad
   w_{i,t}\ge 0 \qquad \forall\,i\in I,\;t\in T
   \]

---

### Data Mapping Rules

| Parameter | Mapping Rule |
|-----------|--------------|
| \(Q_i\) | `file_0_view_0`, column **Q**, row where `truck_id = i` |
| \(S_i\) | `file_0_view_0`, column **S**, row where `truck_id = i` |
| \(C_i\) | `file_0_view_0`, column **C**, row where `truck_id = i` |
| \(d_t\) | `file_0_view_0`, column **d{t}** (e.g., **d1** for \(t=1\)), any row (e.g., row 0) |
| \(\Delta\) | constant = 300 |
| \(\beta\) | constant = 0.9 |

All other constants and index sets are defined implicitly in the model.