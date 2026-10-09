##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{w,p}$ is the assignment cost in USD cents (forbidden assignments are omitted).

##### Constraints

###### 1. Each project is assigned exactly one worker:
$\quad \sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Each worker is assigned to at most one project:
$\quad \sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Assignment only allowed if:
- Worker is not on leave ($on\_leave = 0$)
- Worker skill $\geq$ required skill for project
- Cost $c_{w,p}$ is defined (cell not blank)

That is, $x_{w,p}$ is only defined for eligible $(w,p)$ pairs.

###### 4. Variable constraints:
$\quad x_{w,p} \in \{0,1\}$ for all eligible $(w,p)$

---

##### Retrieved Information

###### Workers (with skill and on_leave status):

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
| W06       | Junior      | 1        |  ← Excluded (on leave)

Eligible workers:  
W00, W18, W02, W17, W10, W05, W11, W14, W01, W24, W16

###### Projects (with required_skill):

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

###### Skill hierarchy:
Junior < Intermediate < Senior < Expert

###### Cost Matrix (c_{w,p}) for eligible assignments (blank cells forbidden):

- Only include $c_{w,p}$ if:
    - Worker $w$ is not on leave
    - Worker $w$'s skill $\geq$ required_skill for project $p$
    - Cost cell is not blank

Below is the full cost matrix for all eligible assignments (in USD cents):

| worker_id | skill        | P00  | P01  | P02  | P03  | P04  | P05  | P06  | P07  | P08  | P09  |
|-----------|-------------|------|------|------|------|------|------|------|------|------|------|
| W00       | Intermediate|      |1116  |414   |      |554   |113   |      |      |853   |1142  |
| W18       | Intermediate|238   |      |1085  |529   |582   |889   |139   |      |630   |538   |
| W02       | Junior      |330   |193   |      |      |431   |      |1130  |      |644   |      |
| W17       | Junior      |538   |      |1359  |      |1366  |702   |122   |      |358   |      |
| W10       | Expert      |1079  |1128  |758   |      |      |108   |      |423   |744   |347   |
| W05       | Senior      |325   |772   |1042  |      |1394  |      |374   |      |1140  |127   |
| W11       | Expert      |1143  |841   |920   |122   |634   |      |1250  |836   |180   |1105  |
| W14       | Expert      |      |687   |126   |547   |125   |1358  |1391  |814   |1362  |962   |
| W01       | Intermediate|      |170   |      |919   |660   |1314  |668   |      |462   |614   |
| W24       | Expert      |784   |1178  |1171  |      |103   |137   |832   |1279  |893   |351   |
| W16       | Senior      |265   |1114  |      |1209  |231   |221   |      |      |135   |      |

- For each cell, assignment is only allowed if:
    - Worker skill $\geq$ required_skill for that project (see above)
    - Cell is not blank

###### Skill eligibility per project:

- For projects with required_skill = Junior: all eligible workers (except those on leave)
- For projects with required_skill = Intermediate: only Intermediate, Senior, Expert
- For projects with required_skill = Senior: only Senior, Expert (none in this instance)
- For projects with required_skill = Expert: only Expert

Thus, for P03 and P09 (Intermediate), only workers with skill Intermediate, Senior, or Expert are eligible.  
For P07 (Expert), only workers with skill Expert are eligible.

###### Final eligible assignments (non-blank, skill-eligible):

| worker_id | skill        | P00  | P01  | P02  | P03  | P04  | P05  | P06  | P07  | P08  | P09  |
|-----------|-------------|------|------|------|------|------|------|------|------|------|------|
| W00       | Intermediate|      |1116  |414   |      |554   |113   |      |      |853   |1142  |
| W18       | Intermediate|238   |      |1085  |529   |582   |889   |139   |      |630   |538   |
| W02       | Junior      |330   |193   |      |      |431   |      |1130  |      |644   |      |
| W17       | Junior      |538   |      |1359  |      |1366  |702   |122   |      |358   |      |
| W10       | Expert      |1079  |1128  |758   |      |      |108   |      |423   |744   |347   |
| W05       | Senior      |325   |772   |1042  |      |1394  |      |374   |      |1140  |127   |
| W11       | Expert      |1143  |841   |920   |122   |634   |      |1250  |836   |180   |1105  |
| W14       | Expert      |      |687   |126   |547   |125   |1358  |1391  |814   |1362  |962   |
| W01       | Intermediate|      |170   |      |919   |660   |1314  |668   |      |462   |614   |
| W24       | Expert      |784   |1178  |1171  |      |103   |137   |832   |1279  |893   |351   |
| W16       | Senior      |265   |1114  |      |1209  |231   |221   |      |      |135   |      |

- For P03 and P09, only Intermediate, Senior, Expert rows are eligible (W02, W17 not eligible).
- For P07, only Expert rows are eligible (W10, W11, W14, W24).

##### Variable domains:

- $x_{w,p} \in \{0,1\}$ for all eligible $(w,p)$ pairs as above.

---

##### Sets

- $W$ = {W00, W18, W02, W17, W10, W05, W11, W14, W01, W24, W16}
- $P$ = {P00, P01, P02, P03, P04, P05, P06, P07, P08, P09}

##### Parameters

- $c_{w,p}$: as in the table above, for eligible $(w,p)$ pairs only.

---

##### Full Mathematical Model

$\boxed{
\begin{align*}
\min \quad & \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p} \\
\text{s.t.} \quad & \sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P \\
& \sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W \\
& x_{w,p} = 0 \quad \text{if $w$ is on leave, or $w$'s skill $<$ required\_skill for $p$, or $c_{w,p}$ is blank} \\
& x_{w,p} \in \{0,1\} \quad \forall (w,p) \text{ eligible}
\end{align*}
}$

All costs $c_{w,p}$ are in USD cents.

---

##### Data

{
  "workers": {
    "W00": {"skill": "Intermediate", "on_leave": 0},
    "W18": {"skill": "Intermediate", "on_leave": 0},
    "W02": {"skill": "Junior", "on_leave": 0},
    "W17": {"skill": "Junior", "on_leave": 0},
    "W10": {"skill": "Expert", "on_leave": 0},
    "W05": {"skill": "Senior", "on_leave": 0},
    "W11": {"skill": "Expert", "on_leave": 0},
    "W14": {"skill": "Expert", "on_leave": 0},
    "W01": {"skill": "Intermediate", "on_leave": 0},
    "W24": {"skill": "Expert", "on_leave": 0},
    "W16": {"skill": "Senior", "on_leave": 0}
  },
  "projects": {
    "P00": {"required_skill": "Junior"},
    "P01": {"required_skill": "Junior"},
    "P02": {"required_skill": "Junior"},
    "P03": {"required_skill": "Intermediate"},
    "P04": {"required_skill": "Junior"},
    "P05": {"required_skill": "Junior"},
    "P06": {"required_skill": "Junior"},
    "P07": {"required_skill": "Expert"},
    "P08": {"required_skill": "Junior"},
    "P09": {"required_skill": "Intermediate"}
  },
  "cost": {
    "W00": {"P01": 1116, "P02": 414, "P04": 554, "P05": 113, "P08": 853, "P09": 1142},
    "W18": {"P00": 238, "P02": 1085, "P03": 529, "P04": 582, "P05": 889, "P06": 139, "P08": 630, "P09": 538},
    "W02": {"P00": 330, "P01": 193, "P04": 431, "P06": 1130, "P08": 644},
    "W17": {"P00": 538, "P02": 1359, "P04": 1366, "P05": 702, "P06": 122, "P08": 358},
    "W10": {"P00": 1079, "P01": 1128, "P02": 758, "P05": 108, "P07": 423, "P08": 744, "P09": 347},
    "W05": {"P00": 325, "P01": 772, "P02": 1042, "P04": 1394, "P06": 374, "P08": 1140, "P09": 127},
    "W11": {"P00": 1143, "P01": 841, "P02": 920, "P03": 122, "P04": 634, "P06": 1250, "P07": 836, "P08": 180, "P09": 1105},
    "W14": {"P01": 687, "P02": 126, "P03": 547, "P04": 125, "P05": 1358, "P06": 1391, "P07": 814, "P08": 1362, "P09": 962},
    "W01": {"P01": 170, "P03": 919, "P04": 660, "P05": 1314, "P06": 668, "P08": 462, "P09": 614},
    "W24": {"P00": 784, "P01": 1178, "P02": 1171, "P04": 103, "P05": 137, "P06": 832, "P07": 1279, "P08": 893, "P09": 351},
    "W16": {"P00": 265, "P01": 1114, "P03": 1209, "P04": 231, "P05": 221, "P08": 135}
  }
}

All costs are in USD cents. Only the above $(w,p)$ pairs are eligible for assignment.