##### Sets and Indices

- Let $W$ be the set of eligible managers (not on leave):  
  $W = \{\text{W00}, \text{W02}, \text{W04}, \text{W06}, \text{W10}, \text{W11}, \text{W12}\}$

- Let $P$ be the set of projects:  
  $P = \{\text{P00}, \text{P01}, \text{P02}, \text{P03}, \text{P04}, \text{P05}\}$

##### Parameters

- $\text{skill}_w$: skill level of manager $w$  
  $\text{skill}_w \in \{\text{Junior}=1, \text{Intermediate}=2, \text{Senior}=3, \text{Expert}=4\}$

  {
    "W00": "Expert",
    "W02": "Intermediate",
    "W04": "Intermediate",
    "W06": "Expert",
    "W10": "Intermediate",
    "W11": "Junior",
    "W12": "Junior"
  }

- $\text{required\_skill}_p$: required skill for project $p$  
  $\text{required\_skill}_p \in \{\text{Junior}=1, \text{Intermediate}=2, \text{Senior}=3, \text{Expert}=4\}$

  {
    "P00": "Junior",
    "P01": "Junior",
    "P02": "Junior",
    "P03": "Junior",
    "P04": "Junior",
    "P05": "Intermediate"
  }

- $c_{w,p}$: assignment cost in USD cents for manager $w$ to project $p$ (blank cell = forbidden assignment)

  {
    "W00": {"P00": 1063, "P01": 219, "P02": null, "P03": 1329, "P04": null, "P05": 436},
    "W02": {"P00": null, "P01": 1397, "P02": 953, "P03": 714, "P04": 205, "P05": null},
    "W04": {"P00": 642, "P01": null, "P02": 1130, "P03": 133, "P04": 199, "P05": 311},
    "W06": {"P00": null, "P01": 822, "P02": null, "P03": 223, "P04": 1155, "P05": 1055},
    "W10": {"P00": 1176, "P01": 1111, "P02": 1380, "P03": 542, "P04": 158, "P05": 922},
    "W11": {"P00": 1091, "P01": 324, "P02": 379, "P03": 272, "P04": null, "P05": null},
    "W12": {"P00": 102, "P01": 353, "P02": 651, "P03": 102, "P04": null, "P05": 7}
  }

##### Decision Variables

- $x_{w,p} \in \{0,1\}$: 1 if manager $w$ is assigned to project $p$, 0 otherwise.

##### Objective Function

$\min \sum_{w \in W} \sum_{p \in P} c_{w,p} \cdot x_{w,p}$

where the sum is only over eligible $(w,p)$ pairs (see below).

##### Constraints

1. **Each project is assigned to exactly one manager:**

   $\sum_{w \in W: (w,p) \text{ eligible}} x_{w,p} = 1 \quad \forall p \in P$

2. **Each manager is assigned to at most one project:**

   $\sum_{p \in P: (w,p) \text{ eligible}} x_{w,p} \leq 1 \quad \forall w \in W$

3. **Eligibility constraints:**

   $x_{w,p} = 0$ unless all of the following hold:
   - $w$ is not on leave (already filtered)
   - $c_{w,p}$ is not blank (assignment allowed)
   - $\text{skill}_w \geq \text{required\_skill}_p$ (using Junior=1, Intermediate=2, Senior=3, Expert=4)

   For all $(w,p)$ pairs where any of these fail, $x_{w,p} = 0$.

4. **Variable domain:**

   $x_{w,p} \in \{0,1\}$ for all eligible $(w,p)$

##### Retrieved Information

{
  "managers": [
    {"worker_id": "W00", "skill": "Expert", "on_leave": 0},
    {"worker_id": "W02", "skill": "Intermediate", "on_leave": 0},
    {"worker_id": "W04", "skill": "Intermediate", "on_leave": 0},
    {"worker_id": "W06", "skill": "Expert", "on_leave": 0},
    {"worker_id": "W10", "skill": "Intermediate", "on_leave": 0},
    {"worker_id": "W11", "skill": "Junior", "on_leave": 0},
    {"worker_id": "W12", "skill": "Junior", "on_leave": 0}
  ],
  "projects": [
    {"project_id": "P00", "required_skill": "Junior"},
    {"project_id": "P01", "required_skill": "Junior"},
    {"project_id": "P02", "required_skill": "Junior"},
    {"project_id": "P03", "required_skill": "Junior"},
    {"project_id": "P04", "required_skill": "Junior"},
    {"project_id": "P05", "required_skill": "Intermediate"}
  ],
  "cost": {
    "W00": {"P00": 1063, "P01": 219, "P02": null, "P03": 1329, "P04": null, "P05": 436},
    "W02": {"P00": null, "P01": 1397, "P02": 953, "P03": 714, "P04": 205, "P05": null},
    "W04": {"P00": 642, "P01": null, "P02": 1130, "P03": 133, "P04": 199, "P05": 311},
    "W06": {"P00": null, "P01": 822, "P02": null, "P03": 223, "P04": 1155, "P05": 1055},
    "W10": {"P00": 1176, "P01": 1111, "P02": 1380, "P03": 542, "P04": 158, "P05": 922},
    "W11": {"P00": 1091, "P01": 324, "P02": 379, "P03": 272, "P04": null, "P05": null},
    "W12": {"P00": 102, "P01": 353, "P02": 651, "P03": 102, "P04": null, "P05": 7}
  },
  "skill_order": {
    "Junior": 1,
    "Intermediate": 2,
    "Senior": 3,
    "Expert": 4
  }
}

##### Eligible Assignment Pairs

Eligible $(w,p)$ pairs are those where:
- $w$ is not on leave,
- $c_{w,p}$ is not blank,
- $\text{skill}_w \geq \text{required\_skill}_p$.

For reference, the eligible pairs and their costs are:

| Worker | Skill        | Project | Required Skill | Cost (USD cents) |
|--------|-------------|---------|---------------|------------------|
| W00    | Expert      | P00     | Junior        | 1063             |
| W00    | Expert      | P01     | Junior        | 219              |
| W00    | Expert      | P03     | Junior        | 1329             |
| W00    | Expert      | P05     | Intermediate  | 436              |
| W02    | Intermediate| P01     | Junior        | 1397             |
| W02    | Intermediate| P02     | Junior        | 953              |
| W02    | Intermediate| P03     | Junior        | 714              |
| W02    | Intermediate| P04     | Junior        | 205              |
| W04    | Intermediate| P00     | Junior        | 642              |
| W04    | Intermediate| P02     | Junior        | 1130             |
| W04    | Intermediate| P03     | Junior        | 133              |
| W04    | Intermediate| P04     | Junior        | 199              |
| W04    | Intermediate| P05     | Intermediate  | 311              |
| W06    | Expert      | P01     | Junior        | 822              |
| W06    | Expert      | P03     | Junior        | 223              |
| W06    | Expert      | P04     | Junior        | 1155             |
| W06    | Expert      | P05     | Intermediate  | 1055             |
| W10    | Intermediate| P00     | Junior        | 1176             |
| W10    | Intermediate| P01     | Junior        | 1111             |
| W10    | Intermediate| P02     | Junior        | 1380             |
| W10    | Intermediate| P03     | Junior        | 542              |
| W10    | Intermediate| P04     | Junior        | 158              |
| W10    | Intermediate| P05     | Intermediate  | 922              |
| W11    | Junior      | P00     | Junior        | 1091             |
| W11    | Junior      | P01     | Junior        | 324              |
| W11    | Junior      | P02     | Junior        | 379              |
| W11    | Junior      | P03     | Junior        | 272              |
| W12    | Junior      | P00     | Junior        | 102              |
| W12    | Junior      | P01     | Junior        | 353              |
| W12    | Junior      | P02     | Junior        | 651              |
| W12    | Junior      | P03     | Junior        | 102              |
| W12    | Junior      | P05     | Intermediate  | 7                |

##### Model Summary

$\boxed{
\begin{align*}
\min\ & \sum_{(w,p)\ \text{eligible}} c_{w,p} x_{w,p} \\
\text{s.t.}\quad
& \sum_{w:\ (w,p)\ \text{eligible}} x_{w,p} = 1 \quad \forall p \in P \\
& \sum_{p:\ (w,p)\ \text{eligible}} x_{w,p} \leq 1 \quad \forall w \in W \\
& x_{w,p} \in \{0,1\} \quad \forall (w,p)\ \text{eligible} \\
& x_{w,p} = 0 \quad \forall (w,p)\ \text{not eligible}
\end{align*}
}$

All costs are in USD cents. The minimum value of the objective function is the minimum total assignment cost in cents.