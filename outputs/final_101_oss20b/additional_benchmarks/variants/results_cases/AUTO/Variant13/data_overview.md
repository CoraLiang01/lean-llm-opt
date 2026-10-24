{
  "query": "An installation project consists of activities with precedence relationships. Normal duration, crash duration, and crash cost per day are listed in project_activities.csv; the required project deadline is listed in project_parameters.csv. Activity durations can be shortened only by integer numbers of days within the listed crash limits.\n\nFormulate a minimum-cost project crashing model. For each activity i, define s_i as the nonnegative start time and z_i as the integer number of crash days used; define T as the project completion time. The objective is to minimize total crashing cost. The model should include precedence constraints using crashed durations, crash-day lower and upper bounds, project-completion constraints for all activities, the project-deadline constraint, nonnegativity constraints for start and completion-time variables, and integer restrictions for crash variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Activity",
        "Predecessors",
        "NormalDuration",
        "CrashDuration",
        "CrashCostPerDay"
      ],
      "file_name": "project_activities.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Activity": "A",
            "CrashCostPerDay": "180",
            "CrashDuration": "3",
            "NormalDuration": "5",
            "Predecessors": ""
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Activity": "B",
            "CrashCostPerDay": "150",
            "CrashDuration": "4",
            "NormalDuration": "6",
            "Predecessors": ""
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Activity": "C",
            "CrashCostPerDay": "130",
            "CrashDuration": "4",
            "NormalDuration": "7",
            "Predecessors": "A"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Activity": "D",
            "CrashCostPerDay": "210",
            "CrashDuration": "3",
            "NormalDuration": "4",
            "Predecessors": "A"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Activity": "E",
            "CrashCostPerDay": "160",
            "CrashDuration": "3",
            "NormalDuration": "5",
            "Predecessors": "B"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Activity": "F",
            "CrashCostPerDay": "190",
            "CrashDuration": "4",
            "NormalDuration": "6",
            "Predecessors": "C;D"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Activity": "G",
            "CrashCostPerDay": "140",
            "CrashDuration": "5",
            "NormalDuration": "7",
            "Predecessors": "D;E"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Activity": "H",
            "CrashCostPerDay": "220",
            "CrashDuration": "2",
            "NormalDuration": "4",
            "Predecessors": "F"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Activity": "I",
            "CrashCostPerDay": "170",
            "CrashDuration": "3",
            "NormalDuration": "5",
            "Predecessors": "G"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Activity": "J",
            "CrashCostPerDay": "260",
            "CrashDuration": "2",
            "NormalDuration": "3",
            "Predecessors": "H;I"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Parameter",
        "Value"
      ],
      "file_name": "project_parameters.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Parameter": "ProjectDeadline",
            "Value": "23"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
