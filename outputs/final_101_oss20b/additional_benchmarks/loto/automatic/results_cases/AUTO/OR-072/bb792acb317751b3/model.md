##### Objective Function: 

$\quad \quad \min \sum_{t=1}^{24} d_t + \sum_{t=1}^{24} c_t$

##### Constraints

###### 1. Driver Coverage Constraints: 

$\quad \quad \sum_{k=0}^{3} d_{((t-k-1) \bmod 24)+1} \;\geq\; R_t , \quad \forall t=1,\dots ,24$

###### 2. Crew Coverage Constraints: 

$\quad \quad \sum_{k=0}^{3} c_{((t-k-1) \bmod 24)+1} \;\geq\; R_t , \quad \forall t=1,\dots ,24$

###### 3. Non‑negativity and Integrality Constraints: 

$\quad \quad d_t \in \mathbb{Z}_{\geq 0}, \; c_t \in \mathbb{Z}_{\geq 0}, \quad \forall t=1,\dots ,24$

Where $R_t$ denotes the “Number Required” for shift $t$ as listed in the table below.

###### 4. Data (Number Required for each shift)

| Shift | Time | Number Required |
|-------|------|-----------------|
| 1 | 0:00‑1:00 | 20 |
| 2 | 1:00‑2:00 | 18 |
| 3 | 2:00‑3:00 | 15 |
| 4 | 3:00‑4:00 | 15 |
| 5 | 4:00‑5:00 | 20 |
| 6 | 5:00‑6:00 | 30 |
| 7 | 6:00‑7:00 | 60 |
| 8 | 7:00‑8:00 | 70 |
| 9 | 8:00‑9:00 | 50 |
| 10 | 9:00‑10:00 | 55 |
| 11 | 10:00‑11:00 | 65 |
| 12 | 11:00‑12:00 | 75 |
| 13 | 12:00‑13:00 | 80 |
| 14 | 13:00‑14:00 | 70 |
| 15 | 14:00‑15:00 | 60 |
| 16 | 15:00‑16:00 | 55 |
| 17 | 16:00‑17:00 | 60 |
| 18 | 17:00‑18:00 | 75 |
| 19 | 18:00‑19:00 | 85 |
| 20 | 19:00‑20:00 | 70 |
| 21 | 20:00‑21:00 | 50 |
| 22 | 21:00‑22:00 | 40 |
| 23 | 22:00‑23:00 | 35 |
| 24 | 23:00‑0:00 | 25 |

The modulo operation in the coverage constraints ensures that a driver or crew member starting at the last few shifts continues to cover the first few shifts of the next day. This model yields the minimum total number of drivers and crew members that must be assigned to satisfy the hourly staffing requirements.