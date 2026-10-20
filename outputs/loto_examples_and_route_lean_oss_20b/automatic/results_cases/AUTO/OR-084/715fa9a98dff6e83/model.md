##### Objective Function: 

$\quad \quad \min T$

##### Constraints

###### 1. Load Constraints for each CPU: 

$\quad \quad \sum_{i=1}^{40} \frac{p_i}{f_j} \, y_{ij} \leq T, \quad \forall j \in \{1,2,3\}$

###### 2. Assignment Constraints: 

$\quad \quad \sum_{j=1}^{3} y_{ij} = 1, \quad \forall i \in \{1,\dots,40\}$

###### 3. Binary Decision Variables: 

$\quad \quad y_{ij} \in \{0,1\}, \quad \forall i \in \{1,\dots,40\}, \; \forall j \in \{1,2,3\}$

###### 4. Non‑negativity of Makespan: 

$\quad \quad T \geq 0$

I indexes the 40 tasks; J indexes the 3 CPUs. $p_i$ is the basic instruction count (in billions) for task $i$, $f_j$ is the frequency (in GHz) of CPU $j$, and $y_{ij}$ indicates whether task $i$ is assigned to CPU $j$. The objective is to minimize the completion time of the last task (makespan).

###### Retrieved Information

{
  "tasks": [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40"
  ],
  "processing_time": {
    "1": 1.1,
    "2": 2.1,
    "3": 3.0,
    "4": 1.0,
    "5": 0.7,
    "6": 5.0,
    "7": 3.0,
    "8": 3.5,
    "9": 4.4,
    "10": 3.8,
    "11": 3.5,
    "12": 2.8,
    "13": 4.1,
    "14": 2.9,
    "15": 5.4,
    "16": 5.8,
    "17": 2.6,
    "18": 4.9,
    "19": 3.4,
    "20": 3.6,
    "21": 5.6,
    "22": 0.9,
    "23": 1.0,
    "24": 0.6,
    "25": 5.1,
    "26": 4.8,
    "27": 5.3,
    "28": 5.9,
    "29": 4.9,
    "30": 3.0,
    "31": 4.8,
    "32": 1.2,
    "33": 4.0,
    "34": 1.3,
    "35": 5.7,
    "36": 3.4,
    "37": 2.8,
    "38": 2.0,
    "39": 4.8,
    "40": 3.0
  },
  "cpus": [
    "1",
    "2",
    "3"
  ],
  "frequencies": {
    "1": 1.33,
    "2": 2.0,
    "3": 2.66
  }
}