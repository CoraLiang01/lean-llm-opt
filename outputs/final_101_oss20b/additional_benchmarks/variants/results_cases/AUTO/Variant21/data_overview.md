{
  "query": "A renovation project is made up of precedence-linked activities. Normal duration, crash duration, and crash cost per day are listed in project_activities.csv; the required completion deadline is listed in project_parameters.csv. Activity durations may be shortened by integer numbers of days within the listed crash limits.\n\nFormulate a minimum-cost project crashing model. For each activity i, define s_i as the nonnegative start time and z_i as the integer number of crash days used; define T as the project completion time. The objective is to minimize total crashing cost. The model should include precedence constraints using crashed durations, crash-day lower and upper bounds, project-completion constraints for all activities, the project-deadline constraint, nonnegativity constraints for start and completion-time variables, and integer restrictions for crash variables.",
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
            "CrashCostPerDay": "120",
            "CrashDuration": "3",
            "NormalDuration": "4",
            "Predecessors": ""
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Activity": "B",
            "CrashCostPerDay": "150",
            "CrashDuration": "5",
            "NormalDuration": "7",
            "Predecessors": ""
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Activity": "C",
            "CrashCostPerDay": "180",
            "CrashDuration": "4",
            "NormalDuration": "6",
            "Predecessors": "A"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Activity": "D",
            "CrashCostPerDay": "160",
            "CrashDuration": "3",
            "NormalDuration": "5",
            "Predecessors": "A"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Activity": "E",
            "CrashCostPerDay": "210",
            "CrashDuration": "2",
            "NormalDuration": "4",
            "Predecessors": "B"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Activity": "F",
            "CrashCostPerDay": "140",
            "CrashDuration": "5",
            "NormalDuration": "7",
            "Predecessors": "C;D"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Activity": "G",
            "CrashCostPerDay": "170",
            "CrashDuration": "4",
            "NormalDuration": "6",
            "Predecessors": "D;E"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Activity": "H",
            "CrashCostPerDay": "200",
            "CrashDuration": "3",
            "NormalDuration": "5",
            "Predecessors": "F;G"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Activity": "I",
            "CrashCostPerDay": "260",
            "CrashDuration": "2",
            "NormalDuration": "3",
            "Predecessors": "H"
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
            "Value": "22"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
