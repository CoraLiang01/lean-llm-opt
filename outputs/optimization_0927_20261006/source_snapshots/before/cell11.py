def load_csv_documents(dataset_address):
    """Build one legacy document per source row in deterministic source order."""
    documents, source_order = [], 0
    for table in _load_tables(dataset_address):
        for source_row, row in table["frame"].iterrows():
            values = {str(column): str(row[column]) for column in row.index}
            documents.append(Document(
                page_content=json.dumps({"values": values}, ensure_ascii=False),
                metadata={"source": str(table["path"]), "source_order": source_order},
            ))
            source_order += 1
    return documents


def build_csvqa_components(
    dataset_address, system_prompt, tool_description, *, route, user_query="",
    result_sink=None,
):
    """Build the shared CSVQA tool in query-planned or legacy full-data mode."""
    route = normalize_route(route)
    planned = CSVQA_MODE_BY_ROUTE[route] == "planned"
    llm = make_llm()

    state = result_sink if result_sink is not None else {}

    def rag_answer(tool_query):
        documents = load_csv_documents(dataset_address)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            (
                "human",
                "Original user query (the only authority for subsets):\n{user_query}\n\n"
                "Agent retrieval request (must not narrow beyond the original query):\n{input}",
            ),
        ])
        result = create_stuff_documents_chain(llm, prompt).invoke({
            "input": tool_query, "user_query": user_query, "context": documents,
        })
        if not str(result or "").strip():
            raise RuntimeError("Legacy CSVQA model returned an empty result")
        return str(result)

    def qa_wrapper(tool_query):
        plan, planner_outputs, errors, status, fallback_reason = None, [], [], "LEGACY_FULL_DATA", None
        attempt_count = 0
        if planned:
            tables = _load_tables(dataset_address)
            attempt_count = 1
            raw_output = ""
            try:
                plan, raw_output = _ask_extraction_plan(
                    llm, route, user_query, tables, system_prompt
                )
                payload = _execute_plan(plan, tables, user_query, route)
                planner_outputs.append(raw_output)
                observation = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                status = "PLANNED"
            except (KeyError, TypeError, ValueError) as exc:
                planner_outputs.append(getattr(exc, "planner_output", raw_output))
                errors.append(str(exc))
                if route != "NRM":
                    raise ValueError(
                        f"CSVQA extraction plan failed for {route}: {errors[-1]}"
                    ) from exc
                payload_tables = []
                for table in tables:
                    frame, columns = table["frame"], list(table["frame"].columns)
                    payload_tables.append({
                        "table_id": f"file_{table['file_index']}_view_0",
                        "file_index": table["file_index"], "file_name": table["path"].name,
                        "role": f"file_{table['file_index']}", "columns": columns,
                        "original_rows": len(frame), "returned_rows": len(frame),
                        "filters": {"logic": "and", "conditions": []},
                        "records": [{"source_row": int(index), "values": {
                            column: str(row[column]) for column in columns
                        }} for index, row in frame.iterrows()],
                    })
                fallback_reason = errors[-1] if errors else "planner did not produce a valid plan"
                observation = json.dumps({
                    "route": route, "query": str(user_query or ""), "tables": payload_tables,
                    "relationships": [], "ignored_file_indices": [],
                    "validation": {
                        "status": "FALLBACK_FULL_DATA", "planner_errors": errors,
                        "fallback_reason": fallback_reason,
                    },
                }, ensure_ascii=False, sort_keys=True)
                status = "FALLBACK_FULL_DATA"
        else:
            observation = rag_answer(tool_query)
        trace = {
            "version": "GPT4.1-LO", "mode": "planned" if planned else "legacy",
            "status": status, "plan": plan, "planner_attempt_count": attempt_count,
            "planner_outputs": planner_outputs, "planner_errors": errors,
            "validation_errors": errors, "fallback_reason": fallback_reason,
            "payload_hash": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
        }
        state.clear()
        state.update({"observation": observation, "trace": trace})
        return observation

    return llm, Tool(name="CSVQA", func=qa_wrapper, description=tool_description), state
