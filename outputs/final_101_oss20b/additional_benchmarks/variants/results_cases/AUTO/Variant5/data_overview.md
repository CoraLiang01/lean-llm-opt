{
  "query": "A maintenance company must assign a set of jobs to technician teams. The processing capacity of each team is listed in machine_capacity.csv. The cost of assigning each job to each team is provided in assignment_costs.csv, and the amount of team capacity consumed by each possible assignment is provided in assignment_resources.csv. Jobs are labeled J1 through J8 in the assignment data.\n\nFormulate a generalized assignment model. For each team i and job j, define x_ij as a binary variable equal to 1 if job j is assigned to team i and 0 otherwise. The objective is to minimize total assignment cost. The model should assign every job to exactly one team, ensure that the total capacity consumed on each team does not exceed its available capacity, and impose binary restrictions on all assignment variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Machine",
        "Capacity"
      ],
      "file_name": "machine_capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Capacity": "13",
            "Machine": "M1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Capacity": "12",
            "Machine": "M2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Capacity": "12",
            "Machine": "M3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Capacity": "12",
            "Machine": "M4"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Machine",
        "J1",
        "J2",
        "J3",
        "J4",
        "J5",
        "J6",
        "J7",
        "J8"
      ],
      "file_name": "assignment_costs.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "J1": "8",
            "J2": "7",
            "J3": "25",
            "J4": "24",
            "J5": "27",
            "J6": "26",
            "J7": "28",
            "J8": "29",
            "Machine": "M1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "J1": "23",
            "J2": "24",
            "J3": "6",
            "J4": "9",
            "J5": "25",
            "J6": "27",
            "J7": "26",
            "J8": "28",
            "Machine": "M2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "J1": "27",
            "J2": "26",
            "J3": "24",
            "J4": "25",
            "J5": "5",
            "J6": "8",
            "J7": "23",
            "J8": "24",
            "Machine": "M3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "J1": "25",
            "J2": "27",
            "J3": "26",
            "J4": "24",
            "J5": "23",
            "J6": "25",
            "J7": "6",
            "J8": "7",
            "Machine": "M4"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "Machine",
        "J1",
        "J2",
        "J3",
        "J4",
        "J5",
        "J6",
        "J7",
        "J8"
      ],
      "file_name": "assignment_resources.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "J1": "5",
            "J2": "6",
            "J3": "8",
            "J4": "7",
            "J5": "9",
            "J6": "8",
            "J7": "7",
            "J8": "7",
            "Machine": "M1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "J1": "8",
            "J2": "7",
            "J3": "4",
            "J4": "7",
            "J5": "8",
            "J6": "9",
            "J7": "8",
            "J8": "7",
            "Machine": "M2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "J1": "9",
            "J2": "8",
            "J3": "7",
            "J4": "8",
            "J5": "6",
            "J6": "5",
            "J7": "8",
            "J8": "7",
            "Machine": "M3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "J1": "8",
            "J2": "8",
            "J3": "7",
            "J4": "8",
            "J5": "8",
            "J6": "7",
            "J7": "5",
            "J8": "6",
            "Machine": "M4"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
