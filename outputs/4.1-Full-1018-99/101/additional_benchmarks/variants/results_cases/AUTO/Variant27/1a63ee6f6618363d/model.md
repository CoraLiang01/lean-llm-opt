##### Objective Function:

$\quad \min \sum_{i \in \mathcal{W}} \sum_{j \in \mathcal{P}} c_{ij} x_{ij}$

where $x_{ij} = 1$ if worker $i$ is assigned to project $j$, $0$ otherwise, and $c_{ij}$ is the cost in USD cents for worker $i$ to do project $j$.

##### Constraints

###### 1. Assignment Constraints:

Each project is assigned exactly one worker:
$$
\sum_{i \in \mathcal{W}} x_{ij} = 1 \quad \forall j \in \mathcal{P}
$$

Each worker is assigned to at most one project:
$$
\sum_{j \in \mathcal{P}} x_{ij} \leq 1 \quad \forall i \in \mathcal{W}
$$

###### 2. Eligibility Mask:

Let $E_{ij} = 1$ if worker $i$ is eligible for project $j$ (not on leave, skill at least required, and cost cell is nonblank), $0$ otherwise. For all ineligible pairs:
$$
x_{ij} = 0 \quad \text{if } E_{ij} = 0
$$

###### 3. Variable Constraints:

$$
x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{W},\ j \in \mathcal{P}
$$

---

##### Retrieved Information

###### Workers (with skill and on_leave):

| worker_id | skill        | on_leave |
|-----------|--------------|----------|
| W00       | Intermediate | 0        |
| W18       | Intermediate | 0        |
| W02       | Junior       | 0        |
| W17       | Junior       | 0        |
| W10       | Expert       | 0        |
| W05       | Senior       | 0        |
| W11       | Expert       | 0        |
| W14       | Expert       | 0        |
| W01       | Intermediate | 0        |
| W24       | Expert       | 0        |
| W16       | Senior       | 0        |
| W06       | Junior       | 1        |  ← Excluded (on_leave=1)

Eligible workers:  
$\mathcal{W} = \{$W00, W18, W02, W17, W10, W05, W11, W14, W01, W24, W16$\}$

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

$\mathcal{P} = \{$P00, P01, P02, P03, P04, P05, P06, P07, P08, P09$\}$

###### Skill Order (for eligibility):

Junior < Intermediate < Senior < Expert

Skill mapping:  
Junior = 1, Intermediate = 2, Senior = 3, Expert = 4

Eligibility rule:  
Worker $i$ is eligible for project $j$ if  
- on_leave = 0  
- worker_skill $\geq$ required_skill  
- cost cell $c_{ij}$ is nonblank

###### Cost Matrix $c_{ij}$ (USD cents) and Eligibility Mask $E_{ij}$

Below, blank cells are ineligible ($E_{ij}=0$).  
Only workers not on leave are included.

| worker_id | skill        | P00  | P01  | P02  | P03  | P04  | P05  | P06  | P07  | P08  | P09  |
|-----------|--------------|------|------|------|------|------|------|------|------|------|------|
| W00       | Intermediate |      |1116  |414   |      |554   |113   |      |11    |853   |1142  |
| W18       | Intermediate |238   |      |1085  |529   |582   |889   |139   |18    |630   |538   |
| W02       | Junior       |330   |193   |      |4     |431   |      |1130  |15    |644   |9     |
| W17       | Junior       |538   |      |1359  |20    |1366  |702   |122   |13    |358   |      |
| W10       | Expert       |1079  |1128  |758   |      |      |108   |      |423   |744   |347   |
| W05       | Senior       |325   |772   |1042  |      |1394  |      |374   |2     |1140  |127   |
| W11       | Expert       |1143  |841   |920   |122   |634   |      |1250  |836   |180   |1105  |
| W14       | Expert       |      |687   |126   |547   |125   |1358  |1391  |814   |1362  |962   |
| W01       | Intermediate |      |170   |      |919   |660   |1314  |668   |4     |462   |614   |
| W24       | Expert       |784   |1178  |1171  |      |103   |137   |832   |1279  |893   |351   |
| W16       | Senior       |265   |1114  |      |1209  |231   |221   |      |17    |135   |      |

Eligibility mask $E_{ij}$ is defined as follows for each $(i,j)$:
- $E_{ij}=1$ if:
    - worker $i$ is not on leave,
    - worker_skill $\geq$ required_skill for project $j$,
    - $c_{ij}$ is nonblank.
- $E_{ij}=0$ otherwise.

For example:
- W02 (Junior, skill=1) can only be assigned to projects with required_skill=Junior (skill=1), and only where cost is nonblank.
- W00 (Intermediate, skill=2) can be assigned to projects with required_skill=Junior or Intermediate (skill=1 or 2), and only where cost is nonblank.
- W10 (Expert, skill=4) can be assigned to any project where cost is nonblank.

###### Full eligibility mask $E_{ij}$ (1=eligible, 0=ineligible):

| worker_id | P00 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| W00       |  0  |  1  |  1  |  0  |  1  |  1  |  0  |  0  |  1  |  1  |
| W18       |  1  |  0  |  1  |  1  |  1  |  1  |  1  |  0  |  1  |  1  |
| W02       |  1  |  1  |  0  |  0  |  1  |  0  |  1  |  0  |  1  |  0  |
| W17       |  1  |  0  |  1  |  0  |  1  |  1  |  1  |  0  |  1  |  0  |
| W10       |  1  |  1  |  1  |  0  |  0  |  1  |  0  |  1  |  1  |  1  |
| W05       |  1  |  1  |  1  |  0  |  1  |  0  |  1  |  1  |  1  |  1  |
| W11       |  1  |  1  |  1  |  1  |  1  |  0  |  1  |  1  |  1  |  1  |
| W14       |  0  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| W01       |  0  |  1  |  0  |  1  |  1  |  1  |  1  |  0  |  1  |  1  |
| W24       |  1  |  1  |  1  |  0  |  1  |  1  |  1  |  1  |  1  |  1  |
| W16       |  1  |  1  |  0  |  1  |  1  |  1  |  0  |  1  |  1  |  0  |

(Eligibility mask constructed by applying all rules above.)

---

##### Sets

- $\mathcal{W}$: $\{$W00, W18, W02, W17, W10, W05, W11, W14, W01, W24, W16$\}$
- $\mathcal{P}$: $\{$P00, P01, P02, P03, P04, P05, P06, P07, P08, P09$\}$

##### Parameters

- $c_{ij}$: Cost in USD cents for worker $i$ to do project $j$ (see table above).
- $E_{ij}$: Eligibility mask (see table above).

---

##### Complete Mathematical Model

$\boxed{
\begin{align*}
\min\ & \sum_{i \in \mathcal{W}} \sum_{j \in \mathcal{P}} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in \mathcal{W}} x_{ij} = 1 \quad \forall j \in \mathcal{P} \\
& \sum_{j \in \mathcal{P}} x_{ij} \leq 1 \quad \forall i \in \mathcal{W} \\
& x_{ij} = 0 \quad \text{if } E_{ij} = 0 \\
& x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{W},\ j \in \mathcal{P}
\end{align*}
}$

---

##### Retrieved Data

{
  "workers": [
    {"worker_id": "W00", "skill": "Intermediate", "on_leave": "0"},
    {"worker_id": "W18", "skill": "Intermediate", "on_leave": "0"},
    {"worker_id": "W02", "skill": "Junior", "on_leave": "0"},
    {"worker_id": "W17", "skill": "Junior", "on_leave": "0"},
    {"worker_id": "W10", "skill": "Expert", "on_leave": "0"},
    {"worker_id": "W05", "skill": "Senior", "on_leave": "0"},
    {"worker_id": "W11", "skill": "Expert", "on_leave": "0"},
    {"worker_id": "W14", "skill": "Expert", "on_leave": "0"},
    {"worker_id": "W01", "skill": "Intermediate", "on_leave": "0"},
    {"worker_id": "W24", "skill": "Expert", "on_leave": "0"},
    {"worker_id": "W16", "skill": "Senior", "on_leave": "0"}
  ],
  "projects": [
    {"project_id": "P00", "required_skill": "Junior"},
    {"project_id": "P01", "required_skill": "Junior"},
    {"project_id": "P02", "required_skill": "Junior"},
    {"project_id": "P03", "required_skill": "Intermediate"},
    {"project_id": "P04", "required_skill": "Junior"},
    {"project_id": "P05", "required_skill": "Junior"},
    {"project_id": "P06", "required_skill": "Junior"},
    {"project_id": "P07", "required_skill": "Expert"},
    {"project_id": "P08", "required_skill": "Junior"},
    {"project_id": "P09", "required_skill": "Intermediate"}
  ],
  "cost": {
    "W00": {"P00": "", "P01": "1116", "P02": "414", "P03": "", "P04": "554", "P05": "113", "P06": "", "P07": "11", "P08": "853", "P09": "1142"},
    "W18": {"P00": "238", "P01": "", "P02": "1085", "P03": "529", "P04": "582", "P05": "889", "P06": "139", "P07": "18", "P08": "630", "P09": "538"},
    "W02": {"P00": "330", "P01": "193", "P02": "", "P03": "4", "P04": "431", "P05": "", "P06": "1130", "P07": "15", "P08": "644", "P09": "9"},
    "W17": {"P00": "538", "P01": "", "P02": "1359", "P03": "20", "P04": "1366", "P05": "702", "P06": "122", "P07": "13", "P08": "358", "P09": ""},
    "W10": {"P00": "1079", "P01": "1128", "P02": "758", "P03": "", "P04": "", "P05": "108", "P06": "", "P07": "423", "P08": "744", "P09": "347"},
    "W05": {"P00": "325", "P01": "772", "P02": "1042", "P03": "", "P04": "1394", "P05": "", "P06": "374", "P07": "2", "P08": "1140", "P09": "127"},
    "W11": {"P00": "1143", "P01": "841", "P02": "920", "P03": "122", "P04": "634", "P05": "", "P06": "1250", "P07": "836", "P08": "180", "P09": "1105"},
    "W14": {"P00": "", "P01": "687", "P02": "126", "P03": "547", "P04": "125", "P05": "1358", "P06": "1391", "P07": "814", "P08": "1362", "P09": "962"},
    "W01": {"P00": "", "P01": "170", "P02": "", "P03": "919", "P04": "660", "P05": "1314", "P06": "668", "P07": "4", "P08": "462", "P09": "614"},
    "W24": {"P00": "784", "P01": "1178", "P02": "1171", "P03": "", "P04": "103", "P05": "137", "P06": "832", "P07": "1279", "P08": "893", "P09": "351"},
    "W16": {"P00": "265", "P01": "1114", "P02": "", "P03": "1209", "P04": "231", "P05": "221", "P06": "", "P07": "17", "P08": "135", "P09": ""}
  },
  "eligibility_mask": {
    "W00": {"P00": 0, "P01": 1, "P02": 1, "P03": 0, "P04": 1, "P05": 1, "P06": 0, "P07": 0, "P08": 1, "P09": 1},
    "W18": {"P00": 1, "P01": 0, "P02": 1, "P03": 1, "P04": 1, "P05": 1, "P06": 1, "P07": 0, "P08": 1, "P09": 1},
    "W02": {"P00": 1, "P01": 1, "P02": 0, "P03": 0, "P04": 1, "P05": 0, "P06": 1, "P07": 0, "P08": 1, "P09": 0},
    "W17": {"P00": 1, "P01": 0, "P02": 1, "P03": 0, "P04": 1, "P05": 1, "P06": 1, "P07": 0, "P08": 1, "P09": 0},
    "W10": {"P00": 1, "P01": 1, "P02": 1, "P03": 0, "P04": 0, "P05": 1, "P06": 0, "P07": 1, "P08": 1, "P09": 1},
    "W05": {"P00": 1, "P01": 1, "P02": 1, "P03": 0, "P04": 1, "P05": 0, "P06": 1, "P07": 1, "P08": 1, "P09": 1},
    "W11": {"P00": 1, "P01": 1, "P02": 1, "P03": 1, "P04": 1, "P05": 0, "P06": 1, "P07": 1, "P08": 1, "P09": 1},
    "W14": {"P00": 0, "P01": 1, "P02": 1, "P03": 1, "P04": 1, "P05": 1, "P06": 1, "P07": 1, "P08": 1, "P09": 1},
    "W01": {"P00": 0, "P01": 1, "P02": 0, "P03": 1, "P04": 1, "P05": 1, "P06": 1, "P07": 0, "P08": 1, "P09": 1},
    "W24": {"P00": 1, "P01": 1, "P02": 1, "P03": 0, "P04": 1, "P05": 1, "P06": 1, "P07": 1, "P08": 1, "P09": 1},
    "W16": {"P00": 1, "P01": 1, "P02": 0, "P03": 1, "P04": 1, "P05": 1, "P06": 0, "P07": 1, "P08": 1, "P09": 0}
  }
}