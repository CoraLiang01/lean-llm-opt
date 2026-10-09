##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{w,p}$ is the assignment cost in USD cents (only defined for allowed assignments).

##### Constraints

###### 1. Each project is assigned exactly one worker:

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Each worker is assigned to at most one project:

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Assignment only allowed if:
- Worker is not on leave,
- Worker skill $\geq$ project required_skill,
- Cost $c_{w,p}$ is defined (cell is not blank).

So,

$x_{w,p} = 0$ if any of the following holds:
- $w$ is on leave,
- $skill(w) < required\_skill(p)$,
- $c_{w,p}$ is blank.

###### 4. Variable domain:

$x_{w,p} \in \{0,1\} \quad \forall w \in W,\, p \in P$

---

##### Retrieved Information

###### Workers (excluding on_leave=1):

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

(Excluded: W06, on_leave=1)

###### Projects:

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

###### Cost Matrix (c_{w,p}), only for allowed assignments (blank cells forbidden):

- W00 (Intermediate):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|     |1116 | 414 |     | 554 | 113 |     | 11  | 853 |1142 |

- W18 (Intermediate):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|238  |     |1085 | 529 | 582 | 889 | 139 | 18  | 630 | 538 |

- W02 (Junior):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|330  |193  |     | 4   |431  |     |1130 | 15  | 644 | 9  |

- W17 (Junior):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|538  |     |1359 | 20  |1366 | 702 |122  | 13  | 358 |     |

- W10 (Expert):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|1079 |1128 | 758 |     |     |108  |     |423  | 744 |347  |

- W05 (Senior):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|325  |772  |1042 |     |1394 |     |374  | 2   |1140 |127  |

- W11 (Expert):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|1143 |841  | 920 |122  |634  |     |1250 |836  |180  |1105 |

- W14 (Expert):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|     |687  |126  |547  |125  |1358 |1391 |814  |1362 |962  |

- W01 (Intermediate):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|     |170  |     |919  |660  |1314 |668  | 4   |462  |614  |

- W24 (Expert):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|784  |1178 |1171 |     |103  |137  |832  |1279 |893  |351  |

- W16 (Senior):

| P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|265  |1114 |     |1209 |231  |221  |     |17   |135  |     |

###### Assignment allowed only if:
- Worker skill $\geq$ project required_skill (using: Junior=1, Intermediate=2, Senior=3, Expert=4)
- Cost cell is not blank

###### Sets:

- $W$ (eligible workers): {W00, W18, W02, W17, W10, W05, W11, W14, W01, W24, W16}
- $P$ (projects): {P00, P01, P02, P03, P04, P05, P06, P07, P08, P09}

###### Cost matrix $c_{w,p}$ (only for allowed assignments):

{
  "W00": {"P01": 1116, "P02": 414, "P04": 554, "P05": 113, "P07": 11, "P08": 853, "P09": 1142},
  "W18": {"P00": 238, "P02": 1085, "P03": 529, "P04": 582, "P05": 889, "P06": 139, "P07": 18, "P08": 630, "P09": 538},
  "W02": {"P00": 330, "P01": 193, "P03": 4, "P04": 431, "P06": 1130, "P07": 15, "P08": 644, "P09": 9},
  "W17": {"P00": 538, "P02": 1359, "P03": 20, "P04": 1366, "P05": 702, "P06": 122, "P07": 13, "P08": 358},
  "W10": {"P00": 1079, "P01": 1128, "P02": 758, "P05": 108, "P07": 423, "P08": 744, "P09": 347},
  "W05": {"P00": 325, "P01": 772, "P02": 1042, "P04": 1394, "P06": 374, "P07": 2, "P08": 1140, "P09": 127},
  "W11": {"P00": 1143, "P01": 841, "P02": 920, "P03": 122, "P04": 634, "P06": 1250, "P07": 836, "P08": 180, "P09": 1105},
  "W14": {"P01": 687, "P02": 126, "P03": 547, "P04": 125, "P05": 1358, "P06": 1391, "P07": 814, "P08": 1362, "P09": 962},
  "W01": {"P01": 170, "P03": 919, "P04": 660, "P05": 1314, "P06": 668, "P07": 4, "P08": 462, "P09": 614},
  "W24": {"P00": 784, "P01": 1178, "P02": 1171, "P04": 103, "P05": 137, "P06": 832, "P07": 1279, "P08": 893, "P09": 351},
  "W16": {"P00": 265, "P01": 1114, "P03": 1209, "P04": 231, "P05": 221, "P07": 17, "P08": 135}
}

###### Skill mapping:

{
  "Junior": 1,
  "Intermediate": 2,
  "Senior": 3,
  "Expert": 4
}

###### Project required_skill:

{
  "P00": "Junior",
  "P01": "Junior",
  "P02": "Junior",
  "P03": "Intermediate",
  "P04": "Junior",
  "P05": "Junior",
  "P06": "Junior",
  "P07": "Expert",
  "P08": "Junior",
  "P09": "Intermediate"
}

###### Worker skills:

{
  "W00": "Intermediate",
  "W18": "Intermediate",
  "W02": "Junior",
  "W17": "Junior",
  "W10": "Expert",
  "W05": "Senior",
  "W11": "Expert",
  "W14": "Expert",
  "W01": "Intermediate",
  "W24": "Expert",
  "W16": "Senior"
}

---

##### Summary

- Sets: $W$ (eligible workers), $P$ (projects)
- Parameters: $c_{w,p}$ (cost in cents, only for allowed assignments)
- Variables: $x_{w,p} \in \{0,1\}$
- Objective: Minimize $\sum_{w \in W} \sum_{p \in P} c_{w,p} x_{w,p}$
- Each project assigned to exactly one worker
- Each worker assigned to at most one project
- Assignment only allowed if worker not on leave, skill $\geq$ required, and cost cell is not blank

---

##### All required data and constraints are included above.