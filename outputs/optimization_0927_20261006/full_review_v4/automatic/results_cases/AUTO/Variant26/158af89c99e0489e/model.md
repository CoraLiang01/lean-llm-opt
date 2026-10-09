##### Sets and Parameters

Let $W$ be the set of workers and $P$ the set of projects. Let $O$ be the set of listed offers, each with attributes:
- $w$: worker\_id
- $p$: project\_id
- $c_{wp}$: cost\_cents
- $l_{wp}$: on\_leave (0 or 1)
- $s_w$: worker\_skill (Junior < Intermediate < Senior < Expert)
- $r_p$: required\_skill (Junior < Intermediate < Senior < Expert)

Define $S(\text{Junior})=1$, $S(\text{Intermediate})=2$, $S(\text{Senior})=3$, $S(\text{Expert})=4$.

Let $E$ be the set of eligible assignment pairs $(w,p)$ such that:
- $(w,p)$ is a listed offer,
- $l_{wp}=0$,
- $S(s_w) \geq S(r_p)$.

##### Decision Variables

For each eligible $(w,p) \in E$:
- $x_{wp} \in \{0,1\}$, where $x_{wp}=1$ if worker $w$ is assigned to project $p$.

##### Objective Function

$\min \sum_{(w,p) \in E} c_{wp} \cdot x_{wp}$

##### Constraints

1. **Each project assigned to exactly one worker:**

$\sum_{\substack{w: (w,p) \in E}} x_{wp} = 1 \quad \forall p \in P$

2. **Each worker assigned to at most one project:**

$\sum_{\substack{p: (w,p) \in E}} x_{wp} \leq 1 \quad \forall w \in W$

3. **Eligibility:**

$x_{wp} = 0$ for all $(w,p) \notin E$ (enforced by only defining variables for eligible pairs).

4. **Variable domain:**

$x_{wp} \in \{0,1\} \quad \forall (w,p) \in E$

##### Retrieved Information

```json
[
  {"worker_id": "W10", "project_id": "P00", "cost_cents": 147, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
  {"worker_id": "W10", "project_id": "P01", "cost_cents": 114, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
  {"worker_id": "W10", "project_id": "P03", "cost_cents": 998, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W10", "project_id": "P04", "cost_cents": 1270, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W10", "project_id": "P07", "cost_cents": 953, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W19", "project_id": "P03", "cost_cents": 109, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W19", "project_id": "P04", "cost_cents": 1306, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W19", "project_id": "P05", "cost_cents": 626, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W19", "project_id": "P06", "cost_cents": 466, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W19", "project_id": "P07", "cost_cents": 1365, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W11", "project_id": "P00", "cost_cents": 235, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Intermediate"},
  {"worker_id": "W11", "project_id": "P02", "cost_cents": 1034, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Senior"},
  {"worker_id": "W11", "project_id": "P03", "cost_cents": 782, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Junior"},
  {"worker_id": "W11", "project_id": "P04", "cost_cents": 556, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Junior"},
  {"worker_id": "W11", "project_id": "P05", "cost_cents": 1018, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Junior"},
  {"worker_id": "W11", "project_id": "P07", "cost_cents": 1136, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Junior"},
  {"worker_id": "W00", "project_id": "P04", "cost_cents": 430, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W00", "project_id": "P05", "cost_cents": 1385, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W00", "project_id": "P06", "cost_cents": 260, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W00", "project_id": "P07", "cost_cents": 1107, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W20", "project_id": "P00", "cost_cents": 675, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
  {"worker_id": "W20", "project_id": "P01", "cost_cents": 1055, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
  {"worker_id": "W20", "project_id": "P03", "cost_cents": 703, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
  {"worker_id": "W20", "project_id": "P06", "cost_cents": 893, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
  {"worker_id": "W20", "project_id": "P07", "cost_cents": 129, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
  {"worker_id": "W04", "project_id": "P03", "cost_cents": 543, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W04", "project_id": "P04", "cost_cents": 205, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W04", "project_id": "P05", "cost_cents": 1149, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W04", "project_id": "P06", "cost_cents": 533, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W04", "project_id": "P07", "cost_cents": 732, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
  {"worker_id": "W06", "project_id": "P00", "cost_cents": 1217, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
  {"worker_id": "W06", "project_id": "P01", "cost_cents": 425, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
  {"worker_id": "W06", "project_id": "P03", "cost_cents": 1097, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W06", "project_id": "P04", "cost_cents": 822, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W06", "project_id": "P05", "cost_cents": 221, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W06", "project_id": "P06", "cost_cents": 496, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W06", "project_id": "P07", "cost_cents": 908, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W14", "project_id": "P03", "cost_cents": 379, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
  {"worker_id": "W14", "project_id": "P04", "cost_cents": 239, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
  {"worker_id": "W14", "project_id": "P05", "cost_cents": 460, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
  {"worker_id": "W14", "project_id": "P06", "cost_cents": 130, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
  {"worker_id": "W15", "project_id": "P00", "cost_cents": 722, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
  {"worker_id": "W15", "project_id": "P03", "cost_cents": 1062, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W15", "project_id": "P04", "cost_cents": 1314, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W15", "project_id": "P05", "cost_cents": 528, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W15", "project_id": "P06", "cost_cents": 835, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
  {"worker_id": "W15", "project_id": "P07", "cost_cents": 918, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"}
]
```

##### Notes

- Only the above listed $(w,p)$ pairs are eligible for assignment.
- All skill and leave constraints are enforced by the eligibility set $E$.
- The objective is the sum of selected assignment costs in USD cents.