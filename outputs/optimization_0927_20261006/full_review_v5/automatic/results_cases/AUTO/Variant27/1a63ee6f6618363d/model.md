##### Objective Function:

$\quad \min \sum_{w \in \mathcal{W}} \sum_{p \in \mathcal{P}} c_{w,p} \cdot x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{w,p}$ is the assignment cost in USD cents (only for eligible assignments).

##### Constraints

###### 1. Project Assignment

$\sum_{w \in \mathcal{W}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

(Each project is assigned exactly one worker.)

###### 2. Worker Assignment

$\sum_{p \in \mathcal{P}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}$

(Each worker is assigned to at most one project.)

###### 3. Eligibility

$x_{w,p} = 0$ if any of the following holds:
- Worker $w$ has on\_leave = 1
- Worker $w$'s skill is less than project $p$'s required\_skill (with Junior < Intermediate < Senior < Expert)
- $c_{w,p}$ is blank (assignment forbidden)

###### 4. Variable Domain

$x_{w,p} \in \{0,1\} \quad \forall w \in \mathcal{W}, p \in \mathcal{P}$

---

##### Retrieved Information

###### Workers

| worker_id | skill        | on_leave |
|-----------|-------------|----------|
| W00       | Intermediate| 0        |
| W18       | Intermediate| 0        |
| W02       | Junior      | 0        |
| W17       | Junior      | 0        |
| W10       | Expert      | 0        |
| W05       | Senior      | 0        |
| W11       | Expert      | 0        |
| W14       | Expert      | 0        |
| W01       | Intermediate| 0        |
| W24       | Expert      | 0        |
| W16       | Senior      | 0        |
| W06       | Junior      | 1        |  ← Excluded (on_leave=1)

Eligible workers: W00, W01, W02, W05, W10, W11, W14, W16, W17, W18, W24

###### Projects

| project_id | required_skill |
|------------|---------------|
| P00        | Junior        |
| P01        | Junior        |
| P02        | Junior        |
| P03        | Intermediate  |
| P04        | Junior        |
| P05        | Junior        |
| P06        | Junior        |
| P07        | Expert        |
| P08        | Junior        |
| P09        | Intermediate  |

###### Skill Order

Junior < Intermediate < Senior < Expert

###### Cost Matrix (c_{w,p})

Only eligible assignments are listed (worker not on leave, skill $\geq$ required, cost cell not blank):

| worker_id | skill        | P00  | P01  | P02  | P03  | P04  | P05  | P06  | P07  | P08  | P09  |
|-----------|-------------|------|------|------|------|------|------|------|------|------|------|
| W00       | Intermediate|      |1116  |414   |      |554   |113   |      |11    |853   |1142  |
| W01       | Intermediate|      |170   |      |919   |660   |1314  |668   |4     |462   |614   |
| W02       | Junior      |330   |193   |      |      |431   |      |1130  |      |644   |      |
| W05       | Senior      |325   |772   |1042  |      |1394  |      |374   |2     |1140  |127   |
| W10       | Expert      |1079  |1128  |758   |      |      |108   |      |423   |744   |347   |
| W11       | Expert      |1143  |841   |920   |122   |634   |      |1250  |836   |180   |1105  |
| W14       | Expert      |      |687   |126   |547   |125   |1358  |1391  |814   |1362  |962   |
| W16       | Senior      |265   |1114  |      |1209  |231   |221   |      |17    |135   |      |
| W17       | Junior      |538   |      |      |      |1366  |702   |122   |      |358   |      |
| W18       | Intermediate|238   |      |1085  |529   |582   |889   |139   |18    |630   |538   |
| W24       | Expert      |784   |1178  |1171  |      |103   |137   |832   |1279  |893   |351   |

Eligibility notes:
- W02, W17: cannot be assigned to projects with required_skill > Junior (i.e., P03, P09, P07 forbidden)
- W00, W01, W18: cannot be assigned to projects with required_skill > Intermediate (i.e., P07 forbidden)
- W05, W16: cannot be assigned to projects with required_skill > Senior (i.e., P07 forbidden)
- Only W10, W11, W14, W24 (Experts) can be assigned to P07 (required_skill=Expert)
- Blank cells are forbidden assignments

###### Sets

$\mathcal{W} = \{$W00, W01, W02, W05, W10, W11, W14, W16, W17, W18, W24$\}$

$\mathcal{P} = \{$P00, P01, P02, P03, P04, P05, P06, P07, P08, P09$\}$

###### Cost Table (Eligible Assignments Only)

| worker_id | P00  | P01  | P02  | P03  | P04  | P05  | P06  | P07  | P08  | P09  |
|-----------|------|------|------|------|------|------|------|------|------|------|
| W00       |      |1116  |414   |      |554   |113   |      |      |853   |1142  |
| W01       |      |170   |      |919   |660   |1314  |668   |      |462   |614   |
| W02       |330   |193   |      |      |431   |      |1130  |      |644   |      |
| W05       |325   |772   |1042  |      |1394  |      |374   |      |1140  |127   |
| W10       |1079  |1128  |758   |      |      |108   |      |423   |744   |347   |
| W11       |1143  |841   |920   |122   |634   |      |1250  |836   |180   |1105  |
| W14       |      |687   |126   |547   |125   |1358  |1391  |814   |1362  |962   |
| W16       |265   |1114  |      |1209  |231   |221   |      |      |135   |      |
| W17       |538   |      |      |      |1366  |702   |122   |      |358   |      |
| W18       |238   |      |1085  |529   |582   |889   |139   |      |630   |538   |
| W24       |784   |1178  |1171  |      |103   |137   |832   |1279  |893   |351   |

(Cells left blank above are forbidden by either cost matrix, skill, or leave status.)

---

##### Decision Variables

$x_{w,p} = \begin{cases}
1 & \text{if worker $w$ is assigned to project $p$ (eligible)} \\
0 & \text{otherwise}
\end{cases}$

---

##### Model Summary

- Minimize total assignment cost (USD cents)
- Each project assigned exactly one eligible worker
- Each worker assigned to at most one project
- Only eligible assignments allowed (not on leave, skill $\geq$ required, cost cell not blank)

---

##### All Required Parameters

- Workers: W00, W01, W02, W05, W10, W11, W14, W16, W17, W18, W24 (with skill and on_leave as above)
- Projects: P00, P01, P02, P03, P04, P05, P06, P07, P08, P09 (with required_skill as above)
- Cost matrix: as above, only for eligible assignments

---

**The minimum total cost in USD cents is the optimal value of the objective function for this model.**