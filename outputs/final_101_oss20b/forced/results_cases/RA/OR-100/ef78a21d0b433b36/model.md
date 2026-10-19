##### Objective Function: 

$\quad \quad \max \sum_{i} p_{i} \cdot x_{i}$

##### Constraints

###### 1. Workshop Capacity Constraints: 

$\quad \quad \sum_{i} a_{i,\text{Casting}} \, x_{i} \leq 7650,\quad \sum_{i} a_{i,\text{Milling}} \, x_{i} \leq 6320,\quad \sum_{i} a_{i,\text{Finishing}} \, x_{i} \leq 5538,\quad \sum_{i} a_{i,\text{Assembly}} \, x_{i} \leq 5957,\quad \sum_{i} a_{i,\text{QA \& Packaging}} \, x_{i} \leq 6988$

###### 2. Non-negativity and Integrality Constraints: 

$\quad \quad  x_{i} \in \mathbb{Z}_{\geq 0}, \quad \forall i$

##### Retrieved Information

{
  "components": [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "C22",
    "C23",
    "C24",
    "C25",
    "C26",
    "C27",
    "C28",
    "C29",
    "C30",
    "C31",
    "C32",
    "C33",
    "C34",
    "C35",
    "C36",
    "C37",
    "C38",
    "C39",
    "C40",
    "C41",
    "C42",
    "C43",
    "C44",
    "C45",
    "C46",
    "C47",
    "C48",
    "C49",
    "C50",
    "C51",
    "C52",
    "C53",
    "C54",
    "C55",
    "C56",
    "C57",
    "C58",
    "C59",
    "C60",
    "C61",
    "C62",
    "C63",
    "C64",
    "C65",
    "C66",
    "C67",
    "C68",
    "C69",
    "C70",
    "C71",
    "C72",
    "C73",
    "C74",
    "C75",
    "C76",
    "C77",
    "C78",
    "C79",
    "C80",
    "C81",
    "C82",
    "C83",
    "C84",
    "C85",
    "C86",
    "C87",
    "C88",
    "C89",
    "C90",
    "C91",
    "C92",
    "C93",
    "C94",
    "C95",
    "C96",
    "C97",
    "C98",
    "C99",
    "C100",
    "C101",
    "C102",
    "C103",
    "C104",
    "C105",
    "C106",
    "C107",
    "C108",
    "C109",
    "C110",
    "C111"
  ],
  "workshops": [
    "Casting",
    "Milling",
    "Finishing",
    "Assembly",
    "QA & Packaging"
  ],
  "processing_time": {
    "C1": {
      "Casting": 0.74,
      "Milling": 0.77,
      "Finishing": 1.41,
      "Assembly": 4.84,
      "QA & Packaging": 0.92
    },
    "C2": {
      "Casting": 3.81,
      "Milling": 4.56,
      "Finishing": 3.47,
      "Assembly": 1.19,
      "QA & Packaging": 3.44
    },
    "C3": {
      "Casting": 3.47,
      "Milling": 4.81,
      "Finishing": 3.17,
      "Assembly": 0.0,
      "QA & Packaging": 3.84
    },
    "C4": {
      "Casting": 3.81,
      "Milling": 3.07,
      "Finishing": 2.23,
      "Assembly": 4.07,
      "QA & Packaging": 2.01
    },
    "C5": {
      "Casting": 4.93,
      "Milling": 2.25,
      "Finishing": 1.22,
      "Assembly": 1.52,
      "QA & Packaging": 4.49
    },
    "C6": {
      "Casting": 4.02,
      "Milling": 4.29,
      "Finishing": 3.44,
      "Assembly": 4.52,
      "QA & Packaging": 2.76
    },
    "C7": {
      "Casting": 3.76,
      "Milling": 2.22,
      "Finishing": 4.54,
      "Assembly": 1.19,
      "QA & Packaging": 4.58
    },
    "C8": {
      "Casting": 4.31,
      "Milling": 1.69,
      "Finishing": 2.68,
      "Assembly": 4.92,
      "QA & Packaging": 4.58
    },
    "C9": {
      "Casting": 0.0,
      "Milling": 3.83,
      "Finishing": 1.12,
      "Assembly": 2.11,
      "QA & Packaging": 0.57
    },
    "C10": {
      "Casting": 3.27,
      "Milling": 4.72,
      "Finishing": 1.32,
      "Assembly": 4.18,
      "QA & Packaging": 4.18
    },
    "C11": {
      "Casting": 0.0,
      "Milling": 0.0,
      "Finishing": 2.34,
      "Assembly": 2.6,
      "QA & Packaging": 0.6
    },
    "C12": {
      "Casting": 1.0,
      "Milling": 3.63,
      "Finishing": 1.57,
      "Assembly": 4.72,
      "QA & Packaging": 4.72
    },
    "C13": {
      "Casting": 2.19,
      "Milling": 1.25,
      "Finishing": 2.49,
      "Assembly": 4.15,
      "QA & Packaging": 0.0
    },
    "C14": {
      "Casting": 4.86,
      "Milling": 2.74,
      "Finishing": 4.83,
      "Assembly": 4.33,
      "QA & Packaging": 4.33
    },
    "C15": {
      "Casting": 4.51,
      "Milling": 4.82,
      "Finishing": 0.58,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C16": {
      "Casting": 2.35,
      "Milling": 3.63,
      "Finishing": 4.64,
      "Assembly": 2.34,
      "QA & Packaging": 2.34
    },
    "C17": {
      "Casting": 4.7,
      "Milling": 0.0,
      "Finishing": 1.1,
      "Assembly": 3.95,
      "QA & Packaging": 3.95
    },
    "C18": {
      "Casting": 3.68,
      "Milling": 4.19,
      "Finishing": 4.91,
      "Assembly": 3.06,
      "QA & Packaging": 3.06
    },
    "C19": {
      "Casting": 4.23,
      "Milling": 4.26,
      "Finishing": 2.28,
      "Assembly": 4.54,
      "QA & Packaging": 4.54
    },
    "C20": {
      "Casting": 0.0,
      "Milling": 0.0,
      "Finishing": 4.31,
      "Assembly": 4.18,
      "QA & Packaging": 4.18
    },
    "C21": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C22": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C23": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C24": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C25": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C26": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C27": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C28": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C29": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C30": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C31": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C32": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C33": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C34": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C35": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C36": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C37": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C38": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C39": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C40": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C41": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C42": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C43": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C44": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C45": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C46": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C47": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C48": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C49": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C50": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C51": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C52": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C53": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C54": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C55": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C56": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C57": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C58": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C59": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C60": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C61": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C62": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C63": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C64": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C65": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C66": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C67": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C68": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C69": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C70": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C71": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C72": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C73": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C74": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C75": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C76": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C77": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C78": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C79": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C80": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C81": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C82": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C83": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C84": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C85": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C86": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C87": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C88": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C89": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C90": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C91": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C92": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C93": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C94": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C95": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C96": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C97": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C98": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C99": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C100": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C101": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C102": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C103": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C104": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C105": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C106": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C107": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C108": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C109": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C110": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    },
    "C111": {
      "Casting": 4.23,
      "Milling": 4.82,
      "Finishing": 4.51,
      "Assembly": 4.51,
      "QA & Packaging": 4.51
    }
  },
  "unit_price": {
    "C1": 193,
    "C2": 64,
    "C3": 103,
    "C4": 210,
    "C5": 85,
    "C6": 126,
    "C7": 226,
    "C8": 94,
    "C9": 73,
    "C10": 120,
    "C11": 81,
    "C12": 94,
    "C13": 133,
    "C14": 197,
    "C15": 63,
    "C16": 159,
    "C17": 160,
    "C18": 97,
    "C19": 182,
    "C20": 128,
    "C21": 181,
    "C22": 171,
    "C23": 91,
    "C24": 228,
    "C25": 152,
    "C26": 85,
    "C27": 203,
    "C28": 134,
    "C29": 232,
    "C30": 125,
    "C31": 181,
    "C32": 246,
    "C33": 226,
    "C34": 88,
    "C35": 187,
    "C36": 152,
    "C37": 130,
    "C38": 86,
    "C39": 50,
    "C40": 229,
    "C41": 93,
    "C42": 169,
    "C43": 72,
    "C44": 67,
    "C45": 136,
    "C46": 118,
    "C47": 101,
    "C48": 94,
    "C49": 78,
    "C50": 76,
    "C51": 155,
    "C52": 114,
    "C53": 225,
    "C54": 238,
    "C55": 59,
    "C56": 135,
    "C57": 245,
    "C58": 231,
    "C59": 219,
    "C60": 167,
    "C61": 164,
    "C62": 139,
    "C63": 220,
    "C64": 167,
    "C65": 240,
    "C66": 170,
    "C67": 91,
    "C68": 106,
    "C69": 135,
    "C70": 91,
    "C71": 51,
    "C72": 73,
    "C73": 211,
    "C74": 189,
    "C75": 169,
    "C76": 153,
    "C77": 151,
    "C78": 167,
    "C79": 173,
    "C80": 152,
    "C81": 101,
    "C82": 216,
    "C83": 196,
    "C84": 92,
    "C85": 92,
    "C86": 97,
    "C87": 224,
    "C88": 128,
    "C89": 139,
    "C90": 109,
    "C91": 206,
    "C92": 161,
    "C93": 227,
    "C94": 187,
    "C95": 106,
    "C96": 248,
    "C97": 82,
    "C98": 222,
    "C99": 209,
    "C100": 223,
    "C101": 204,
    "C102": 114,
    "C103": 146,
    "C104": 231,
    "C105": 93,
    "C106": 224,
    "C107": 220,
    "C108": 100,
    "C109": 187,
    "C110": 213,
    "C111": 142
  },
  "total_hours": {
    "Casting": 7650,
    "Milling": 6320,
    "Finishing": 5538,
    "Assembly": 5957,
    "QA & Packaging": 6988
  }
}