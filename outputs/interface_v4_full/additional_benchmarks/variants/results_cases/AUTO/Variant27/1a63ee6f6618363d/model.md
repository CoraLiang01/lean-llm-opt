##### Objective Function:

$\quad \min \sum_{i \in \mathcal{W}} \sum_{j \in \mathcal{P}} c_{ij} x_{ij}$

where:
- $\mathcal{W}$ is the set of eligible workers (on_leave=0)
- $\mathcal{P}$ is the set of projects
- $c_{ij}$ is the assignment cost in USD cents for worker $i$ to project $j$ (forbidden assignments have no $x_{ij}$ variable)
- $x_{ij} \in \{0,1\}$ indicates if worker $i$ is assigned to project $j$

##### Constraints

###### 1. Each project is assigned to exactly one worker:

$\sum_{i \in \mathcal{W}_j} x_{ij} = 1 \quad \forall j \in \mathcal{P}$

where $\mathcal{W}_j$ is the set of eligible workers for project $j$ (i.e., not on leave, skill $\geq$ required_skill, and $c_{ij}$ is defined).

###### 2. Each worker is assigned to at most one project:

$\sum_{j \in \mathcal{P}_i} x_{ij} \leq 1 \quad \forall i \in \mathcal{W}$

where $\mathcal{P}_i$ is the set of projects for which worker $i$ is eligible (skill $\geq$ required_skill, and $c_{ij}$ is defined).

###### 3. Skill and forbidden assignment constraints:

- $x_{ij}$ is only defined if:
    - Worker $i$ is not on leave,
    - Worker $i$'s skill $\geq$ project $j$'s required_skill,
    - $c_{ij}$ is specified (cell not blank).
- For all other $(i,j)$, $x_{ij}$ is not included in the model.

###### 4. Variable domain:

$x_{ij} \in \{0,1\}$ for all allowed assignments.

---

##### Retrieved Information

**Eligible Workers (on_leave=0):**

| worker_id | skill        |
|-----------|-------------|
| W00       | Intermediate|
| W01       | Intermediate|
| W02       | Junior      |
| W05       | Senior      |
| W10       | Expert      |
| W11       | Expert      |
| W14       | Expert      |
| W16       | Senior      |
| W17       | Junior      |
| W18       | Intermediate|
| W24       | Expert      |

**Projects and Required Skills:**

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

**Skill Hierarchy:** Junior < Intermediate < Senior < Expert

**Cost Matrix (USD cents):**

| worker_id |   P00   |   P01   |   P02   |   P03   |   P04   |   P05   |   P06   |   P07   |   P08   |   P09   | skill        |
|-----------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|--------------|
| W00       |         | 1116    | 414     |         | 554     | 113     |         | 11      | 853     | 1142    | Intermediate |
| W01       |         | 170     |         | 919     | 660     | 1314    | 668     | 4       | 462     | 614     | Intermediate |
| W02       | 330     | 193     |         | 4       | 431     |         | 1130    | 15      | 644     | 9       | Junior       |
| W05       | 325     | 772     | 1042    |         | 1394    |         | 374     | 2       | 1140    | 127     | Senior       |
| W10       | 1079    | 1128    | 758     |         |         | 108     |         | 423     | 744     | 347     | Expert       |
| W11       | 1143    | 841     | 920     | 122     | 634     |         | 1250    | 836     | 180     | 1105    | Expert       |
| W14       |         | 687     | 126     | 547     | 125     | 1358    | 1391    | 814     | 1362    | 962     | Expert       |
| W16       | 265     | 1114    |         | 1209    | 231     | 221     |         | 17      | 135     |         | Senior       |
| W17       | 538     |         | 1359    | 20      | 1366    | 702     | 122     | 13      | 358     |         | Junior       |
| W18       | 238     |         | 1085    | 529     | 582     | 889     | 139     | 18      | 630     | 538     | Intermediate |
| W24       | 784     | 1178    | 1171    |         | 103     | 137     | 832     | 1279    | 893     | 351     | Expert       |

**Skill eligibility for each assignment:**
- Worker $i$ can be assigned to project $j$ only if skill($i$) $\geq$ required_skill($j$).
- For example, only Experts can be assigned to P07 (required_skill=Expert).

**Forbidden assignments:**
- Any blank cell in the cost matrix is forbidden.

---

**Sets:**
- $\mathcal{W}$ = {W00, W01, W02, W05, W10, W11, W14, W16, W17, W18, W24}
- $\mathcal{P}$ = {P00, P01, P02, P03, P04, P05, P06, P07, P08, P09}

**Parameters:**
- $c_{ij}$: as given in the cost matrix above, in USD cents, for allowed assignments only.

**Variables:**
- $x_{ij} \in \{0,1\}$ for each allowed assignment (see above).

---

**Summary:**
- Assign each project to exactly one eligible worker.
- Each worker is assigned to at most one project.
- Only allow assignments where the worker is not on leave, skill is sufficient, and the cost cell is not blank.
- Minimize the total cost in USD cents.

**All required vectors and matrices are included above.**