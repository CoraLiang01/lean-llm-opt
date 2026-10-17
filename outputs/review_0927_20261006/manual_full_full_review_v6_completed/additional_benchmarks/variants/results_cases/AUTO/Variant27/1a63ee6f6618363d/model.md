##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \cdot x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{w,p}$ is the assignment cost in USD cents.

##### Constraints

###### 1. Assignment Constraints:

Each project is assigned to exactly one worker:
$$
\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P
$$

Each worker is assigned to at most one project:
$$
\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W
$$

###### 2. Eligibility Constraints:

For each $(w,p)$ pair, $x_{w,p}$ is only allowed if:
- Worker $w$ is not on leave ($on\_leave = 0$)
- Worker $w$'s skill is at least the required skill for project $p$ (Junior < Intermediate < Senior < Expert)
- The cost $c_{w,p}$ is not blank

For all ineligible $(w,p)$ pairs, enforce:
$$
x_{w,p} = 0
$$

###### 3. Variable Constraints:

$$
x_{w,p} \in \{0,1\} \quad \forall w \in W,\, p \in P
$$

---

##### Retrieved Information

###### Workers (with skill and on_leave):

| worker_id | skill        | on_leave |
|-----------|-------------|----------|
| W00       | Intermediate| 0        |
| W01       | Intermediate| 0        |
| W02       | Junior      | 0        |
| W05       | Senior      | 0        |
| W06       | Junior      | 1        |  ← Excluded
| W10       | Expert      | 0        |
| W11       | Expert      | 0        |
| W14       | Expert      | 0        |
| W16       | Senior      | 0        |
| W17       | Junior      | 0        |
| W18       | Intermediate| 0        |
| W24       | Expert      | 0        |

Eligible workers: W00, W01, W02, W05, W10, W11, W14, W16, W17, W18, W24

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

###### Skill Order (for eligibility):

Junior = 1 < Intermediate = 2 < Senior = 3 < Expert = 4

###### Cost Matrix (c_{w,p}), with eligibility mask:

For each worker $w$ and project $p$, $x_{w,p}$ is eligible if:
- $on\_leave = 0$
- $skill(w) \geq required\_skill(p)$
- $c_{w,p}$ is not blank

Below is the cost matrix for eligible pairs (blank cells or ineligible pairs are omitted and must have $x_{w,p}=0$):

| worker_id | skill        | P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----------|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| W00       | Intermediate|     |1116 | 414 |     | 554 | 113 |     |     | 853 |1142 |
| W01       | Intermediate|     | 170 |     | 919 | 660 |1314 | 668 |     | 462 | 614 |
| W02       | Junior      | 330 | 193 |     |     | 431 |     |1130 |     | 644 |     |
| W05       | Senior      | 325 | 772 |1042 |     |1394 |     | 374 |     |1140 | 127 |
| W10       | Expert      |1079 |1128 | 758 |     |     | 108 |     | 423 | 744 | 347 |
| W11       | Expert      |1143 | 841 | 920 | 122 | 634 |     |1250 | 836 | 180 |1105 |
| W14       | Expert      |     | 687 | 126 | 547 | 125 |1358 |1391 | 814 |1362 | 962 |
| W16       | Senior      | 265 |1114 |     |1209 | 231 | 221 |     |     | 135 |     |
| W17       | Junior      | 538 |     |1359 |     |1366 | 702 | 122 |     | 358 |     |
| W18       | Intermediate| 238 |     |1085 | 529 | 582 | 889 | 139 |     | 630 | 538 |
| W24       | Expert      | 784 |1178 |1171 |     | 103 | 137 | 832 |1279 | 893 | 351 |

Eligibility mask (forbidden pairs, $x_{w,p}=0$):

- Any cell that is blank in the above table
- Any cell where worker's skill is less than required_skill (e.g., W02 and W17 cannot take P03 or P09 (Intermediate), or P07 (Expert))
- W06 is excluded entirely (on_leave=1)

###### Sets:

- $W = \{$W00, W01, W02, W05, W10, W11, W14, W16, W17, W18, W24$\}$
- $P = \{$P00, P01, P02, P03, P04, P05, P06, P07, P08, P09$\}$

###### Parameters:

- $c_{w,p}$: as above, in USD cents, for eligible pairs; undefined or forbidden otherwise.

###### Decision Variables:

- $x_{w,p} \in \{0,1\}$ for all eligible $(w,p)$ pairs; $x_{w,p}=0$ for all ineligible pairs.

---

##### Full Mathematical Model

$\boxed{
\begin{align*}
\min \quad & \sum_{w \in W} \sum_{p \in P} c_{w,p} \cdot x_{w,p} \\
\text{s.t.} \quad & \sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P \\
& \sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W \\
& x_{w,p} = 0 \quad \text{for all ineligible } (w,p) \\
& x_{w,p} \in \{0,1\} \quad \forall w \in W,\, p \in P
\end{align*}
}$

where eligibility is defined as above.

---

##### Data Used

{
  "workers": {
    "W00": {"skill": "Intermediate", "on_leave": 0},
    "W01": {"skill": "Intermediate", "on_leave": 0},
    "W02": {"skill": "Junior", "on_leave": 0},
    "W05": {"skill": "Senior", "on_leave": 0},
    "W10": {"skill": "Expert", "on_leave": 0},
    "W11": {"skill": "Expert", "on_leave": 0},
    "W14": {"skill": "Expert", "on_leave": 0},
    "W16": {"skill": "Senior", "on_leave": 0},
    "W17": {"skill": "Junior", "on_leave": 0},
    "W18": {"skill": "Intermediate", "on_leave": 0},
    "W24": {"skill": "Expert", "on_leave": 0}
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
    "W01": {"P01": 170, "P03": 919, "P04": 660, "P05": 1314, "P06": 668, "P08": 462, "P09": 614},
    "W02": {"P00": 330, "P01": 193, "P04": 431, "P06": 1130, "P08": 644},
    "W05": {"P00": 325, "P01": 772, "P02": 1042, "P04": 1394, "P06": 374, "P08": 1140, "P09": 127},
    "W10": {"P00": 1079, "P01": 1128, "P02": 758, "P05": 108, "P07": 423, "P08": 744, "P09": 347},
    "W11": {"P00": 1143, "P01": 841, "P02": 920, "P03": 122, "P04": 634, "P06": 1250, "P07": 836, "P08": 180, "P09": 1105},
    "W14": {"P01": 687, "P02": 126, "P03": 547, "P04": 125, "P05": 1358, "P06": 1391, "P07": 814, "P08": 1362, "P09": 962},
    "W16": {"P00": 265, "P01": 1114, "P03": 1209, "P04": 231, "P05": 221, "P08": 135},
    "W17": {"P00": 538, "P02": 1359, "P04": 1366, "P05": 702, "P06": 122, "P08": 358},
    "W18": {"P00": 238, "P02": 1085, "P03": 529, "P04": 582, "P05": 889, "P06": 139, "P08": 630, "P09": 538},
    "W24": {"P00": 784, "P01": 1178, "P02": 1171, "P04": 103, "P05": 137, "P06": 832, "P07": 1279, "P08": 893, "P09": 351}
  }
}

##### Notes

- All costs are in USD cents.
- Only eligible $(w,p)$ pairs (as above) may have $x_{w,p}=1$; all others must have $x_{w,p}=0$.
- Each project must be assigned to exactly one eligible worker; each worker may be assigned at most one project.