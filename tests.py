import pytest


@pytest.fixture(scope="session")
def processed_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path in [config.CLEAN_SUBMISSIONS_TRAIN_FILE, config.CLEAN_SUBMISSIONS_EVAL_FILE,
                          config.CLEAN_AUDIENCES_TRAIN_FILE, config.CLEAN_AUDIENCES_EVAL_FILE,
                          config.CLEAN_CLEARANCE_NOTES_FILE]:
            loaded[file_path] = pd.read_csv(file_path, keep_default_na=False, na_values=[""],
                                            low_memory=False)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The cleaned extracts could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on cleaning, features and models reads the files the data "
            "pipeline writes, so the pipeline has to have run and written all five."
        )
    return loaded


@pytest.fixture(scope="session")
def raw_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path, columns in [
                (config.RAW_SUBMISSIONS_TRAIN_FILE, ["submission_id",
                                                     config.SUBMISSION_TARGET_COLUMN]),
                (config.RAW_SUBMISSIONS_EVAL_FILE, None),
                (config.RAW_AUDIENCES_TRAIN_FILE, ["airing_id", config.AUDIENCE_TARGET_COLUMN]),
                (config.RAW_AUDIENCES_EVAL_FILE, None),
                (config.RAW_CLEARANCE_NOTES_FILE, None)]:
            loaded[file_path] = pd.read_csv(file_path, dtype=str, keep_default_na=False,
                                            usecols=columns)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The raw extracts supplied with the project could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The data folder has to stay as it was supplied."
        )
    return loaded


@pytest.fixture(scope="session")
def saved_models():
    from config import config

    loaded = {"problem": None}
    try:
        from modeling.model_registry import load_feature_schema, load_model

        loaded.update({
            "amendment": load_model(config.AMENDMENT_MODEL_FILE),
            "exposure": load_model(config.EXPOSURE_MODEL_FILE),
            "issue": load_model(config.ISSUE_MODEL_FILE),
            "amendment_schema": list(load_feature_schema(config.AMENDMENT_SCHEMA_FILE)),
            "exposure_schema": list(load_feature_schema(config.EXPOSURE_SCHEMA_FILE)),
        })
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The saved models and their feature schemas could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Model training saves them, and every check on the models scores what "
            "training saved."
        )
    return loaded


@pytest.fixture(scope="session")
def open_queue():
    from config import config

    loaded = {"problem": None}
    try:
        from data_pipeline.dataset_loader import load_raw_dataset

        loaded.update({
            "campaigns": load_raw_dataset(config.OPEN_CAMPAIGNS_FILE),
            "spots": load_raw_dataset(config.BOOKED_SPOTS_FILE),
            "schedule": load_raw_dataset(config.PROGRAMME_SCHEDULE_FILE),
            "notes": load_raw_dataset(config.OPEN_CAMPAIGN_NOTES_FILE),
        })
        for name in ["campaigns", "spots", "notes"]:
            assert config.CAMPAIGN_REFERENCE_COLUMN in getattr(loaded[name], "columns", []), (
                "load_raw_dataset returned "
                + type(loaded[name]).__name__
                + " for the live "
                + name
                + ", where it returns a table with a "
                + config.CAMPAIGN_REFERENCE_COLUMN
                + " column."
            )
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The live queue could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The tools, the queue and the workflow all start from the open campaigns, "
            "their booked spots, the programme schedule and the clearance notes, read as text."
        )
    return loaded


@pytest.fixture(scope="session")
def knowledge_collection():
    from config import config

    opened = {"problem": None}
    try:
        from knowledge.vector_index import open_vector_index

        opened["collection"] = open_vector_index(config.VECTOR_STORE_DIRECTORY)
    except (Exception, SystemExit) as error:
        opened["problem"] = (
            "The vector store could not be opened ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Knowledge indexing builds it, and every check on retrieval, on the codes "
            "and on the spot check reads it."
        )
    return opened


@pytest.fixture(scope="session")
def document_passages():
    from config import config

    cut = {"problem": None}
    try:
        from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections
        from knowledge.pdf_extraction import extract_text_from_directory

        cut["documents"] = extract_text_from_directory(config.DOCS_DIRECTORY)
        cut["sections"] = split_documents_into_sections(cut["documents"])
        cut["chunks"] = build_chunk_records(cut["sections"])
    except (Exception, SystemExit) as error:
        cut["problem"] = (
            "The documents could not be read, split into sections and cut into passages ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on the knowledge base starts from the passages the sixteen "
            "documents are cut into."
        )
    return cut


@pytest.fixture(scope="session")
def planted_codes(tmp_path_factory):
    from config import config

    documents = config.CATEGORY_CODE_DOCUMENTS
    hours = ("Spots for this category may not be transmitted between {start} and {end}.")
    any_time = "Spots for this category may be transmitted at any time of the broadcast day."
    exception = ("The restricted hours do not apply to a centre break within live sport coverage "
                 "that begins more than {after} minutes after the coverage starts and more than "
                 "{before} minutes before it ends.")
    in_or_adjacent = "Spots may not be transmitted in or adjacent to a programme made for children."
    in_only = "Spots may not be transmitted in a programme made for children."
    threshold = ("Spots may not be transmitted in or adjacent to a programme whose under-18 "
                 "audience index is {index} or more.")
    cap = ("No more than {cap} spots of one campaign may be transmitted on one channel in one "
           "broadcast day.")
    separation = ("Two spots of one campaign on one channel must be at least {minutes} minutes "
                  "apart.")
    opening = ("Purpose and scope. This code sets out when and where spots for the category may be "
               "transmitted on every Harbourlight channel. ")
    closing = ("Questions about this code. Agencies ask through the Agency Liaison assigned to "
               "them, and the desk checks every new break against this code. ")
    passages = {
        "Betting and gaming": [
            opening + separation.format(minutes=25),
            "Restricted hours. " + exception.format(after=15, before=20) + " "
            + hours.format(start="04:45", end="22:15")
            + " An end break is never within live sport coverage.",
            "The under-18 audience. " + threshold.format(index=105)
            + " A spot in an end break is tested against both programmes.",
            "Programmes made for children. " + in_or_adjacent + " " + cap.format(cap=4),
            closing,
        ],
        "Less healthy food": [
            opening + hours.format(start="04:45", end="22:15"),
            "The under-18 audience. " + threshold.format(index=135),
            "How many spots, and how far apart. " + separation.format(minutes=12) + " "
            + cap.format(cap=9),
            closing + in_or_adjacent,
        ],
        "Alcohol": [
            opening + in_only + " " + hours.format(start="23:30", end="02:00"),
            "How many spots. " + cap.format(cap=3) + " Every booked spot counts towards it.",
            "How far apart. " + separation.format(minutes=45),
            closing + threshold.format(index=115),
        ],
        "Consumer credit": [
            opening + threshold.format(index=140),
            "Restricted hours. " + any_time + " The rules on the audience still apply.",
            "Programmes made for children. " + in_or_adjacent + " " + separation.format(minutes=18),
            closing + cap.format(cap=6),
        ],
        "Medicines and health": [
            opening + cap.format(cap=2),
            "Restricted hours. " + hours.format(start="07:00", end="09:30"),
            "The audience and children. " + threshold.format(index=160) + " " + in_or_adjacent,
            "How many spots, and how far apart. " + cap.format(cap=2) + " "
            + separation.format(minutes=60),
            closing,
        ],
    }
    decoy = ("Superseded rules. This notice keeps rules that no longer apply to any category. "
             + hours.format(start="01:00", end="02:30") + " "
             + exception.format(after=1, before=1) + " " + threshold.format(index=99) + " "
             + in_only + " " + cap.format(cap=1) + " " + separation.format(minutes=90))
    contradictory = {
        "Betting and gaming": [
            opening + hours.format(start="03:15", end="22:45") + " " + threshold.format(index=135),
            "The under-18 audience. " + threshold.format(index=145) + " " + in_only,
            cap.format(cap=9) + " " + separation.format(minutes=35),
        ],
        "Alcohol": [
            opening + threshold.format(index=145) + " " + in_only,
            cap.format(cap=11) + " " + separation.format(minutes=35),
        ],
        "Less healthy food": [
            opening + hours.format(start="03:15", end="20:45") + " " + in_or_adjacent,
            "Programmes made for children. " + in_only + " " + threshold.format(index=130),
            cap.format(cap=12) + " " + separation.format(minutes=40),
        ],
        "Consumer credit": [
            opening + any_time + " " + exception.format(after=12, before=14),
            threshold.format(index=170) + " " + in_only + " " + cap.format(cap=9) + " "
            + separation.format(minutes=50),
        ],
        "Medicines and health": [
            opening + hours.format(start="07:15", end="10:45") + " " + threshold.format(index=175)
            + " " + in_or_adjacent,
            cap.format(cap=13) + " " + separation.format(minutes=8),
            "How far apart. " + separation.format(minutes=55),
        ],
    }
    built = {
        "problem": None,
        "decoy_figures": {"01:00", "02:30", 1, 99, 90},
        "expected": {
            "Betting and gaming": {
                "restricted_hours": {"start": "04:45", "end": "22:15"},
                "live_sport_exception": {"minutes_after_start": 15, "minutes_before_end": 20},
                "children_rule": config.CHILDREN_IN_OR_ADJACENT, "index_threshold": 105,
                "daily_cap": 4, "separation_minutes": 25},
            "Less healthy food": {
                "restricted_hours": {"start": "04:45", "end": "22:15"},
                "live_sport_exception": None, "children_rule": config.CHILDREN_IN_OR_ADJACENT,
                "index_threshold": 135, "daily_cap": 9, "separation_minutes": 12},
            "Alcohol": {
                "restricted_hours": {"start": "23:30", "end": "02:00"},
                "live_sport_exception": None, "children_rule": config.CHILDREN_IN_ONLY,
                "index_threshold": 115, "daily_cap": 3, "separation_minutes": 45},
            "Consumer credit": {
                "restricted_hours": None, "live_sport_exception": None,
                "children_rule": config.CHILDREN_IN_OR_ADJACENT, "index_threshold": 140,
                "daily_cap": 6, "separation_minutes": 18},
            "Medicines and health": {
                "restricted_hours": {"start": "07:00", "end": "09:30"},
                "live_sport_exception": None, "children_rule": config.CHILDREN_IN_OR_ADJACENT,
                "index_threshold": 160, "daily_cap": 2, "separation_minutes": 60},
        },
    }
    try:
        from knowledge.vector_index import build_vector_index, open_vector_index

        for store_name, written in [("valid", passages), ("contradictory", contradictory)]:
            chunk_records = [{"chunk_id": documents[category] + "-" + str(position),
                              "chunk_text": text, "document_name": documents[category]}
                             for category, texts in written.items()
                             for position, text in enumerate(texts)]
            chunk_records.append({"chunk_id": "Superseded-Rules-Notice.pdf-0",
                                  "chunk_text": decoy,
                                  "document_name": "Superseded-Rules-Notice.pdf"})
            store_path = str(tmp_path_factory.mktemp("planted_" + store_name) / "store")
            built[store_name + "_count"] = build_vector_index(chunk_records, store_path)
            built[store_name] = open_vector_index(store_path)
            built[store_name + "_records"] = len(chunk_records)
    except (Exception, SystemExit) as error:
        built["problem"] = (
            "A store of planted category codes could not be built with build_vector_index and "
            "opened with open_vector_index ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The checks on reading the codes read figures the suite invented, so a "
            "solution that remembered the real ones is not rewarded for it."
        )
    return built


@pytest.fixture
def planted_schedule():
    import pandas as pd

    columns = ["schedule_entry_id", "channel", "scheduling_system", "transmission_day",
               "start_time", "end_time", "programme_title", "genre", "live_flag",
               "live_sport_flag", "made_for_children_flag", "premiere_flag", "episode_number",
               "school_holiday_flag", "day_type", "lead_in_genre", "forecast_under18_index",
               "region_feed_code", "playout_code"]
    programmes = [
        ("Drama", "Friday", "06:00", "09:00", "Morning Repeats", "Drama", "No", "No", "40"),
        ("Drama", "Friday", "09:00", "12:00", "Daytime Drama", "Drama", "No", "No", "45"),
        ("Drama", "Friday", "12:00", "13:00", "Lunchtime Mystery", "Crime drama", "No", "No",
         "55"),
        ("Drama", "Friday", "13:30", "17:00", "Afternoon Classics", "Film", "No", "No", "50"),
        ("Drama", "Friday", "17:00", "19:00", "Teatime Cartoons", "Animation", "No", "Yes", "95"),
        ("Drama", "Friday", "19:00", "21:00", "Evening Drama", "Drama", "No", "No", "100"),
        ("Drama", "Friday", "21:00", "23:00", "Crime Hour", "Crime drama", "No", "No", "70"),
        ("Drama", "Friday", "23:00", "25:30", "Late Film", "Film", "No", "No", "30"),
        ("Drama", "Friday", "25:30", "30:00", "Night Shop", "Lifestyle", "No", "No", "10"),
        ("Drama", "Saturday", "06:00", "12:00", "Weekend Breakfast", "Family entertainment",
         "No", "No", "60"),
        ("Drama", "Saturday", "12:00", "30:00", "Saturday Marathon", "Drama", "No", "No", "50"),
        ("Life", "Friday", "06:00", "18:00", "Life Daytime", "Lifestyle", "No", "No", "40"),
        ("Life", "Friday", "18:00", "30:00", "Life Evening", "Lifestyle", "No", "No", "45"),
        ("Sport", "Friday", "06:00", "17:00", "Sport Daytime", "Sport", "No", "No", "50"),
        ("Sport", "Friday", "17:00", "19:00", "Cup Football Live", "Football", "Yes", "No", "90"),
        ("Sport", "Friday", "19:00", "20:45", "Second Half Live", "Football", "Yes", "No", "92"),
        ("Sport", "Friday", "20:45", "21:30", "Studio Analysis", "Sport", "No", "No", "80"),
        ("Sport", "Friday", "21:30", "30:00", "Late Sport", "Sport", "No", "No", "40"),
    ]
    rows = [{"schedule_entry_id": "SCH9" + str(position).zfill(4),
             "channel": "Harbourlight " + channel, "scheduling_system": "TXLine",
             "transmission_day": day, "start_time": start, "end_time": end,
             "programme_title": title, "genre": genre, "live_flag": live_sport,
             "live_sport_flag": live_sport, "made_for_children_flag": children,
             "premiere_flag": "No", "episode_number": "", "school_holiday_flag": "No",
             "day_type": "Weekend" if day == "Saturday" else "Weekday", "lead_in_genre": genre,
             "forecast_under18_index": index, "region_feed_code": "RF-01",
             "playout_code": "PLY-9" + str(position).zfill(3)}
            for position, (channel, day, start, end, title, genre, live_sport, children, index)
            in enumerate(programmes)]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def planted_code():
    from config import config

    return {
        "document_name": config.CATEGORY_CODE_DOCUMENTS["Betting and gaming"],
        "restricted_hours": None,
        "live_sport_exception": None,
        "children_rule": config.CHILDREN_IN_OR_ADJACENT,
        "index_threshold": 999,
        "daily_cap": 99,
        "separation_minutes": 0,
        "rule_sentences": {breach_code: "The planted rule for " + breach_code + "."
                           for breach_code in config.BREACH_CODES},
    }


@pytest.fixture(scope="session")
def service_client():
    service = {"problem": None}
    try:
        from fastapi.testclient import TestClient

        from api.routes import service_application

        service["client"] = TestClient(service_application, raise_server_exceptions=False)
    except (Exception, SystemExit) as error:
        service["problem"] = (
            "The REST service could not be started ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on sign-in, access, the queue, the workflow and the chat goes "
            "through it."
        )
    return service


@pytest.fixture(scope="session")
def staff(service_client):
    from config import config

    signed_in = {"problem": service_client["problem"]}
    if signed_in["problem"] is not None:
        return signed_in
    try:
        controllers = [account for account in config.SEED_USER_ACCOUNTS
                       if account["role"] == config.ROLE_CONTROLLER]
        executives = [account for account in config.SEED_USER_ACCOUNTS
                      if account["role"] == config.ROLE_EXECUTIVE]
        for staff_key, account in [("controller", controllers[0]),
                                   ("second_controller", controllers[-1]),
                                   ("executive", executives[0])]:
            reply = service_client["client"].post(
                "/auth/signin",
                json={"username": account["username"], "password": account["password"]})
            assert reply.status_code == 200, (
                "Signing in as "
                + account["username"]
                + " with the password the configuration seeds returned "
                + str(reply.status_code)
                + ", so no check that needs a signed-in "
                + account["role"]
                + " can run."
            )
            signed_in[staff_key] = dict(reply.json(), headers={
                config.SESSION_TOKEN_HEADER: reply.json()["session_token"]})
        signed_in["liaisons"] = [
            {key: account[key] for key in ["username", "display_name", "role",
                                           "assigned_agencies"]}
            for account in config.SEED_USER_ACCOUNTS if account["role"] == config.ROLE_LIAISON]
    except (Exception, SystemExit) as error:
        signed_in["problem"] = (
            "The seeded staff could not be signed in ("
            + type(error).__name__
            + ": "
            + str(error)[:400]
            + "). The accounts are the ones the configuration seeds through the database "
            "setup, and a sign-in answers with the user's display name, role, agencies and a "
            "session token."
        )
    return signed_in


@pytest.fixture(scope="session")
def saved_run(service_client, staff, open_queue):
    from config import config

    run = {"problem": service_client["problem"] or staff["problem"] or open_queue["problem"],
           "decision_problem": None}
    if run["problem"] is not None:
        return run
    client = service_client["client"]
    executive = staff["executive"]
    opener = staff["controller"]
    decider = staff["second_controller"]
    run["executive_request"] = ("Check every booked spot against the category's code and tell "
                                "me whether the copy's offer can be cleared.")
    run["decision"] = config.DECISION_CLEAR_WITH_RESTRICTIONS
    run["decision_reason"] = ("The compliant spots may air; the breaching spots are withdrawn "
                              "and the agency is told how to re-book them.")
    run["opener"] = opener
    run["decider"] = decider
    for name in ["detail", "calls_at_gate", "pending_at_gate", "final_calls"]:
        run[name] = {"status": None, "body": "never requested"}
    run["own_decision"] = {"status": None, "body": "never requested", "still_pending": None,
                           "calls_before": None, "calls_after": None}
    run["closed_decision"] = {"status": None, "body": "never requested", "calls_before": None,
                              "calls_after": None}
    run["run_record"] = {"problem": "never read"}
    run["heard_at_gate"] = {"problem": "never asked"}
    run["heard_after"] = {"problem": "never asked"}
    try:
        queue = client.get("/queue", params={"product_category": config.QUEUE_CATEGORY_ALL},
                           headers=executive["headers"])
        assert queue.status_code == 200 and isinstance(queue.json(), list), (
            "The clearance queue returned "
            + str(queue.status_code)
            + " with "
            + queue.text[:300]
            + ", so no campaign could be chosen to assess."
        )
        assessed = client.get("/clearances",
                              params={"status_value": config.CAMPAIGN_STATUS_OPTIONS},
                              headers=executive["headers"])
        assert assessed.status_code == 200, (
            "Listing the assessed campaigns returned "
            + str(assessed.status_code)
            + " with "
            + assessed.text[:300]
            + ", so a campaign nobody has assessed yet could not be chosen."
        )
        already_assessed = {record["campaign_reference"] for record in assessed.json()}
        candidates = sorted((row["spots_booked"], row["campaign_reference"])
                            for row in queue.json()
                            if 0 < row["spots_breaching"] < row["spots_booked"]
                            and row["campaign_reference"] not in already_assessed)
        assert len(candidates) > 0, (
            "No campaign in the queue with both breaching and compliant spots is left "
            "unassessed, so the one assessment run cannot be opened. Start again from an empty "
            "database."
        )
        run["reference"] = candidates[0][1]
        campaigns = open_queue["campaigns"]
        agency = campaigns.loc[campaigns[config.CAMPAIGN_REFERENCE_COLUMN] == run["reference"],
                               config.AGENCY_CODE_COLUMN].iloc[0]
        run["liaison"] = [liaison for liaison in staff["liaisons"]
                          if agency in liaison["assigned_agencies"]][0]
        run["rival"] = [liaison for liaison in staff["liaisons"]
                        if agency not in liaison["assigned_agencies"]][0]
        opened = client.post("/assessments",
                             json={"campaign_reference": run["reference"],
                                   "executive_request": run["executive_request"]},
                             headers=opener["headers"])
        assert opened.status_code == 201, (
            "Opening an assessment on "
            + run["reference"]
            + " returned "
            + str(opened.status_code)
            + " with "
            + opened.text[:400]
            + ", so the workflow never reached the Clearance Controller."
        )
        run["at_gate"] = opened.json()
        assessment_id = run["at_gate"]["assessment_id"]
    except (Exception, SystemExit) as error:
        run["problem"] = (
            "The one assessment run that the agent, workflow and record checks read could not "
            "be opened ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )
        return run

    calls_path = "/clearances/" + assessment_id + "/calls"
    try:
        reply = client.get("/queue/" + run["reference"], headers=executive["headers"])
        run["detail"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get(calls_path, headers=executive["headers"])
        run["calls_at_gate"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/clearances",
                           params={"status_value": [config.CAMPAIGN_STATUS_PENDING]},
                           headers=executive["headers"])
        run["pending_at_gate"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["calls_at_gate"]["body"] = type(error).__name__ + ": " + str(error)[:300]
    try:
        from chat.campaign_status_answer import answer_campaign_status
        from persistence.database import session_factory
        from persistence.models import AgentRun

        with session_factory() as session:
            run["run_record"] = {"problem": None,
                                 "status_at_gate": session.get(AgentRun,
                                                               run["at_gate"]["run_id"]).status}
            run["heard_at_gate"] = {
                "problem": None,
                "assigned": answer_campaign_status(session, open_queue["campaigns"],
                                                   run["liaison"], run["reference"]),
                "rival": answer_campaign_status(session, open_queue["campaigns"], run["rival"],
                                                run["reference"]),
                "desk": answer_campaign_status(session, open_queue["campaigns"], executive,
                                               run["reference"]),
            }
    except (Exception, SystemExit) as error:
        run["run_record"] = {"problem": ("The paused run could not be read from the database, "
                                         "or its status could not be answered ("
                                         + type(error).__name__ + ": " + str(error)[:300]
                                         + ").")}
        run["heard_at_gate"] = {"problem": run["run_record"]["problem"]}

    try:
        decision_path = "/clearances/" + assessment_id + config.DECISION_PATH_ENDING
        decision_payload = {"decision": run["decision"], "decision_reason": run["decision_reason"]}
        calls_before = client.get(calls_path, headers=executive["headers"])
        attempted = client.post(decision_path, json=decision_payload, headers=opener["headers"])
        calls_after = client.get(calls_path, headers=executive["headers"])
        pending = client.get("/clearances",
                             params={"status_value": [config.CAMPAIGN_STATUS_PENDING]},
                             headers=executive["headers"])
        run["own_decision"] = {
            "status": attempted.status_code,
            "body": attempted.text[:300],
            "still_pending": pending.status_code == 200 and assessment_id in [
                str(record.get("assessment_id")) for record in pending.json()],
            "calls_before": (calls_before.json().get("call_count")
                             if calls_before.status_code == 200 else None),
            "calls_after": (calls_after.json().get("call_count")
                            if calls_after.status_code == 200 else None),
        }
        closed = client.post(decision_path,
                             json={"decision": config.DECISION_CLEAR,
                                   "decision_reason": "Every booked spot may air as booked."},
                             headers=decider["headers"])
        calls_after_closed = client.get(calls_path, headers=executive["headers"])
        run["closed_decision"] = {
            "status": closed.status_code,
            "body": closed.json() if closed.headers.get("content-type", "").startswith(
                "application/json") else closed.text[:300],
            "calls_before": run["own_decision"]["calls_after"],
            "calls_after": (calls_after_closed.json().get("call_count")
                            if calls_after_closed.status_code == 200 else None),
        }
        decided = client.post(decision_path, json=decision_payload, headers=decider["headers"])
        assert decided.status_code == 200, (
            "Recording the decision on "
            + assessment_id
            + " returned "
            + str(decided.status_code)
            + " with "
            + decided.text[:400]
            + ", so the paused run was never resumed."
        )
        run["final"] = decided.json()
    except (Exception, SystemExit) as error:
        run["decision_problem"] = (
            "The Clearance Controller's decision on the one assessment run could not be recorded ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )
        return run

    try:
        reply = client.get(calls_path, headers=executive["headers"])
        run["final_calls"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["final_calls"]["body"] = type(error).__name__ + ": " + str(error)[:300]
    try:
        from chat.campaign_status_answer import answer_campaign_status
        from persistence.database import session_factory
        from persistence.models import AgentRun

        with session_factory() as session:
            finished_run = session.get(AgentRun, run["at_gate"]["run_id"])
            run["run_record"].update({"status_final": finished_run.status,
                                      "thread_id": finished_run.thread_id})
            run["heard_after"] = {
                "problem": None,
                "assigned": answer_campaign_status(session, open_queue["campaigns"],
                                                   run["liaison"], run["reference"]),
                "rival": answer_campaign_status(session, open_queue["campaigns"], run["rival"],
                                                run["reference"]),
                "desk": answer_campaign_status(session, open_queue["campaigns"], executive,
                                               run["reference"]),
            }
    except (Exception, SystemExit) as error:
        run["heard_after"] = {
            "problem": ("The finished run could not be read from the database, or its status "
                        "could not be answered ("
                        + type(error).__name__ + ": " + str(error)[:300] + ")."),
        }
    return run


@pytest.fixture(scope="session")
def chat_replies(service_client, staff):
    replies = {"problem": service_client["problem"] or staff["problem"],
               "rules_question": ("How many spots of one alcohol campaign may go out on one "
                                  "channel in one broadcast day?"),
               "rules": {"status": None, "body": "never asked"},
               "turns": {"status": None, "body": "never read"},
               "latest_turn_before": 0}
    if replies["problem"] is not None:
        return replies
    client = service_client["client"]
    executive = staff["executive"]
    replies["asker"] = executive["username"]
    replies["someone_else"] = staff["liaisons"][0]["username"]
    try:
        earlier = client.get("/chat/turns", params={"turn_limit": 1},
                             headers=executive["headers"])
        if earlier.status_code == 200:
            replies["latest_turn_before"] = max([turn["turn_id"] for turn in earlier.json()]
                                                + [0])
        reply = client.post("/chat/ask", json={"question": replies["rules_question"]},
                            headers=executive["headers"])
        replies["rules"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/chat/turns", params={"turn_limit": 5}, headers=executive["headers"])
        replies["turns"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        replies["rules"]["body"] = type(error).__name__ + ": " + str(error)[:300]
    return replies


def test_cleaning_keeps_one_row_for_every_record_with_an_answer(raw_frames, processed_frames,
                                                                tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.audience_cleaning import clean_audience_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    rebuilt_path = str(tmp_path / "audiences_eval_cleaned_again.csv")
    clean_audience_dataset(config.RAW_AUDIENCES_EVAL_FILE, rebuilt_path)
    rebuilt = pd.read_csv(rebuilt_path, keep_default_na=False, na_values=[""], low_memory=False)
    saved = processed_frames[config.CLEAN_AUDIENCES_EVAL_FILE]
    assert rebuilt.shape == saved.shape and rebuilt.equals(saved), (
        "Cleaning "
        + config.RAW_AUDIENCES_EVAL_FILE
        + " again gives "
        + str(rebuilt.shape)
        + " rows and columns where "
        + config.CLEAN_AUDIENCES_EVAL_FILE
        + " holds "
        + str(saved.shape)
        + ", or the same shape with different values. Every check reads the cleaned files, "
        "so they have to be what the cleaning code produces now: run the data pipeline "
        "again after changing it."
    )
    datasets = [
        (config.RAW_SUBMISSIONS_TRAIN_FILE, config.CLEAN_SUBMISSIONS_TRAIN_FILE, "submission_id",
         config.SUBMISSION_TARGET_COLUMN),
        (config.RAW_SUBMISSIONS_EVAL_FILE, config.CLEAN_SUBMISSIONS_EVAL_FILE, "submission_id",
         config.SUBMISSION_TARGET_COLUMN),
        (config.RAW_AUDIENCES_TRAIN_FILE, config.CLEAN_AUDIENCES_TRAIN_FILE, "airing_id",
         config.AUDIENCE_TARGET_COLUMN),
        (config.RAW_AUDIENCES_EVAL_FILE, config.CLEAN_AUDIENCES_EVAL_FILE, "airing_id",
         config.AUDIENCE_TARGET_COLUMN),
        (config.RAW_CLEARANCE_NOTES_FILE, config.CLEAN_CLEARANCE_NOTES_FILE, "note_id",
         config.NOTE_TARGET_COLUMN),
    ]
    for raw_path, clean_path, identifier, target in datasets:
        raw_frame = raw_frames[raw_path]
        clean_frame = processed_frames[clean_path]
        raw_identifiers = raw_frame[identifier].str.strip()
        kept_identifiers = set(clean_frame[identifier].astype(str))
        given = raw_frame[target].str.strip()
        unanswered = given.str.lower().isin(config.NULL_TOKENS)
        never_answered = set(raw_identifiers[unanswered]) - set(raw_identifiers[~unanswered])
        kept_unanswered = sorted(never_answered & kept_identifiers)
        assert len(never_answered) > 0 and len(kept_unanswered) == 0, (
            clean_path
            + " keeps "
            + str(len(kept_unanswered))
            + " records whose answer "
            + raw_path
            + " never gives, for example "
            + str(kept_unanswered[:3])
            + ". A record with no answer is removed, never given an invented one: a model "
            "fitted on invented answers learns the invention."
        )
        assert len(clean_frame) < len(raw_frame), (
            clean_path
            + " keeps "
            + str(len(clean_frame))
            + " rows of the "
            + str(len(raw_frame))
            + " in "
            + raw_path
            + ". The extract carries repeated rows and rows with no answer, so nothing "
            "has been removed that should have been."
        )
        repeated = int(clean_frame[identifier].duplicated().sum())
        assert repeated == 0, (
            clean_path
            + " still holds "
            + str(repeated)
            + " rows whose "
            + identifier
            + " appears on another row. A record counted twice is weighted twice by every "
            "model fitted on it."
        )
        empty_answers = int(clean_frame[target].isna().sum())
        assert empty_answers == 0 and clean_frame[target].nunique() > 1, (
            "The answer column "
            + target
            + " in "
            + clean_path
            + " holds "
            + str(empty_answers)
            + " empty cells and "
            + str(clean_frame[target].nunique())
            + " distinct values. Every row kept carries an answer, and a single answer leaves "
            "a model nothing to learn."
        )
        if target == config.SUBMISSION_TARGET_COLUMN:
            lost = sorted(set(raw_identifiers[~unanswered]) - kept_identifiers)
            assert len(lost) == 0, (
                clean_path
                + " has lost "
                + str(len(lost))
                + " submissions whose answer "
                + raw_path
                + " gives, for example "
                + str(lost[:3])
                + ". Only repeated rows and rows with no answer are removed: a gap in another "
                "column is filled when the model is fitted, not by throwing the submission away."
            )
            answers = sorted(clean_frame[target].astype(str).unique())
            assert answers == sorted([config.BOOLEAN_NO, config.BOOLEAN_YES]), (
                target
                + " in "
                + clean_path
                + " holds "
                + str(answers[:6])
                + " where every spelling of yes and no is folded onto "
                + config.BOOLEAN_YES
                + " and "
                + config.BOOLEAN_NO
                + "."
            )
        if target == config.AUDIENCE_TARGET_COLUMN:
            plain = given.str.fullmatch(config.PLAIN_NUMBER_PATTERN)
            possible = plain & (pd.to_numeric(given.where(plain)) >= 0)
            lost = sorted(set(raw_identifiers[possible]) - kept_identifiers)
            assert len(lost) == 0, (
                clean_path
                + " has lost "
                + str(len(lost))
                + " airings whose under-18 audience "
                + raw_path
                + " gives as a plain number of thousands that is not negative, for example "
                + str(lost[:3])
                + ". A gap in another column is not a reason to throw an airing away."
            )
            viewers = pd.to_numeric(clean_frame[target])
            assert bool((viewers >= 0).all()), (
                target
                + " in "
                + clean_path
                + " falls as low as "
                + str(viewers.min())
                + ". An audience below nought is no answer, and the airing goes."
            )
        if target == config.NOTE_TARGET_COLUMN:
            answers = sorted(clean_frame[target].astype(str).unique())
            assert answers == sorted(config.GUIDE_DOCUMENT_BY_ISSUE), (
                target
                + " in "
                + clean_path
                + " holds "
                + str(answers)
                + " where the desk codes every note to one of "
                + str(sorted(config.GUIDE_DOCUMENT_BY_ISSUE))
                + "."
            )


def test_times_are_read_on_each_schedulers_own_clock_and_the_small_hours_join_the_day_before(
        raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.audience_cleaning import clean_audience_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_audiences = raw_frames[config.RAW_AUDIENCES_EVAL_FILE]
    template = raw_audiences[raw_audiences[config.AUDIENCE_TARGET_COLUMN].str.strip()
                             .str.fullmatch(r"\d+\.\d")].iloc[0].to_dict()
    cases = [
        ("Airtime Planner", "Monday", "01:10", "02:00", "Sunday", "25:10", "26:00"),
        ("TXLine", "Tuesday", "25:30", "26:15", "Tuesday", "25:30", "26:15"),
        ("SlotMaster", "Sa.", "02:15", "03:45", "Friday", "26:15", "27:45"),
        ("Planit", "Sunday", "05:00", "06:00", "Saturday", "29:00", "30:00"),
        ("txline", "fri", "06:00", "07:30", "Friday", "06:00", "07:30"),
        ("PLANIT", "Wed", "21:00", "22:00", "Wednesday", "21:00", "22:00"),
        ("Airtime Planner", "THU", "05:59", "06:00", "Wednesday", "29:59", "30:00"),
    ]
    crafted = [dict(template, **{
        "airing_id": "AIR99000" + str(position),
        config.SCHEDULING_SYSTEM_COLUMN: case[0],
        config.TRANSMISSION_DAY_COLUMN: case[1],
        config.START_TIME_COLUMN: case[2],
        config.END_TIME_COLUMN: case[3],
    }) for position, case in enumerate(cases)]
    crafted_path = str(tmp_path / "times_written_every_way.csv")
    cleaned_path = str(tmp_path / "times_written_every_way_clean.csv")
    pd.DataFrame(crafted).to_csv(crafted_path, index=False)
    clean_audience_dataset(crafted_path, cleaned_path)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index("airing_id")
    for position, (system, day, start, end, broadcast_day, broadcast_start,
                   broadcast_end) in enumerate(cases):
        airing = "AIR99000" + str(position)
        read_as = ((cleaned.loc[airing, config.TRANSMISSION_DAY_COLUMN],
                    cleaned.loc[airing, config.START_TIME_COLUMN],
                    cleaned.loc[airing, config.END_TIME_COLUMN])
                   if airing in cleaned.index else "a removed row")
        assert read_as == (broadcast_day, broadcast_start, broadcast_end), (
            "A programme that "
            + system
            + " wrote as "
            + day
            + " "
            + start
            + " to "
            + end
            + " cleaned to "
            + str(read_as)
            + " where it runs on "
            + broadcast_day
            + " from "
            + broadcast_start
            + " to "
            + broadcast_end
            + " on the broadcast clock. The configuration says which systems write the "
            "calendar clock and which the broadcast clock; a calendar time before 06:00 "
            "belongs to the previous weekday's broadcast day, 24 hours later on its clock, "
            "and the booking week's Monday morning belongs to its Sunday."
        )

    refusals = [
        ("TXLine", "Monday", "03:00", "04:00",
         "a broadcast-clock system writing a time before 06:00, which its clock never shows"),
        ("Airtime Planner", "Monday", "24:30", "25:00",
         "a calendar-clock system writing 24:30, which its clock never shows"),
        ("", "Monday", "01:10", "02:00",
         "no scheduling system, so 01:10 could be Sunday's broadcast day or no time at all"),
        ("Harbour Playout", "Monday", "10:00", "11:00",
         "a scheduling system the configuration does not know"),
        ("Planit", "Monday", "7.30pm", "21:00", "a time written in no known notation"),
    ]
    for position, (system, day, start, end, reason) in enumerate(refusals):
        refused_path = str(tmp_path / ("refused_time_" + str(position) + ".csv"))
        pd.DataFrame([dict(template, **{
            "airing_id": "AIR99100" + str(position),
            config.SCHEDULING_SYSTEM_COLUMN: system,
            config.TRANSMISSION_DAY_COLUMN: day,
            config.START_TIME_COLUMN: start,
            config.END_TIME_COLUMN: end,
        })]).to_csv(refused_path, index=False)
        refused = False
        try:
            clean_audience_dataset(refused_path, refused_path.replace(".csv", "_clean.csv"))
        except ValueError:
            refused = True
        assert refused, (
            "A programme written "
            + repr(day + " " + start + " to " + end)
            + " by "
            + repr(system or "no scheduling system")
            + " was cleaned without a ValueError. That is "
            + reason
            + ", and a time that cannot be placed for certain is refused, never guessed."
        )


def test_spot_lengths_written_every_way_become_seconds_and_the_clock_number_wins(
        raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.submission_cleaning import clean_submission_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_submissions = raw_frames[config.RAW_SUBMISSIONS_EVAL_FILE]
    template = raw_submissions[raw_submissions[config.SUBMISSION_TARGET_COLUMN].str.strip()
                               .str.lower().isin(["yes", "no"])].iloc[0].to_dict()
    cases = [
        ("TST/PLNT001/030", "30", 30, False), ("TST/PLNT001/030", '30"', 30, False),
        ("TST/PLNT001/030", "30s", 30, False), ("TST/PLNT001/030", "30 sec", 30, False),
        ("TST/PLNT001/030", "0:30", 30, False), ("TST/PLNT001/030", "00:00:30", 30, False),
        ("TST/PLNT002/090", "1:30", 90, False), ("TST/PLNT003/060", "00:01:00", 60, False),
        ("TST/PLNT004/020", "20s", 20, False), ("TST/PLNT007/010", '10"', 10, False),
        ("TST/PLNT005/030", "20", 30, True), ("TST/PLNT006/040", "1:00", 40, True),
    ]
    crafted = [dict(template, **{
        "submission_id": "SUB99000" + str(position).zfill(2),
        config.CLOCK_NUMBER_COLUMN: case[0],
        config.COPY_LENGTH_COLUMN: case[1],
    }) for position, case in enumerate(cases)]
    crafted_path = str(tmp_path / "lengths_written_every_way.csv")
    cleaned_path = str(tmp_path / "lengths_written_every_way_clean.csv")
    pd.DataFrame(crafted).to_csv(crafted_path, index=False)
    clean_submission_dataset(crafted_path, cleaned_path)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index(
        "submission_id")
    for position, (clock_number, written, seconds, disagreeing) in enumerate(cases):
        submission = "SUB99000" + str(position).zfill(2)
        found = (cleaned.loc[submission, config.COPY_LENGTH_COLUMN]
                 if submission in cleaned.index else "a removed row")
        assert found.replace(".", "", 1).isdigit() and float(found) == seconds, (
            "A copy whose clock number is "
            + clock_number
            + " and whose length was written "
            + repr(written)
            + " cleaned to "
            + repr(found)
            + " where it is "
            + str(seconds)
            + " seconds. "
            + ("The clock number ends in the copy's length in seconds, and it wins a "
               "disagreement: the clock number is what the playout system plays."
               if disagreeing else
               "A length is read from seconds, from minutes and seconds, or from hours, "
               "minutes and seconds, with or without a unit, and kept as whole seconds.")
        )

    refusals = [
        ("TST/PLNT010/030", "half a minute", "a length written in words, which no length "
                                             "notation reads"),
        ("TST/PLNT011/030", "30 seconds", "a unit the length notations do not use"),
        ("TST/PLNT012/045", "45", "a clock number ending in 45 seconds, where only the standard "
                                  "lengths " + str(config.STANDARD_SPOT_LENGTHS) + " are booked"),
        ("TST/PLNT013", "30", "a clock number that carries no length at all"),
    ]
    for position, (clock_number, written, reason) in enumerate(refusals):
        refused_path = str(tmp_path / ("refused_length_" + str(position) + ".csv"))
        pd.DataFrame([dict(template, **{
            "submission_id": "SUB99100" + str(position),
            config.CLOCK_NUMBER_COLUMN: clock_number,
            config.COPY_LENGTH_COLUMN: written,
        })]).to_csv(refused_path, index=False)
        refused = False
        try:
            clean_submission_dataset(refused_path, refused_path.replace(".csv", "_clean.csv"))
        except ValueError:
            refused = True
        assert refused, (
            "A copy with clock number "
            + repr(clock_number)
            + " and length "
            + repr(written)
            + " was cleaned without a ValueError. That is "
            + reason
            + ", so the length cannot be read for certain."
        )


def test_audiences_counts_and_rates_written_with_units_become_numbers(raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.audience_cleaning import clean_audience_dataset
    from data_pipeline.submission_cleaning import clean_submission_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_audiences = raw_frames[config.RAW_AUDIENCES_EVAL_FILE]
    audience_template = raw_audiences[raw_audiences[config.SCHEDULING_SYSTEM_COLUMN].str.strip()
                                      == "TXLine"].iloc[0].to_dict()
    audience_cases = [("1.2m", "48.4", 1200.0, 48.4), ("850k", "0.05m", 850.0, 50.0),
                      ("850 thousand", "212,500", 850.0, 212.5),
                      ("1,250,000", "33,333", 1250.0, 33.3), ("2.25m", "7.5k", 2250.0, 7.5)]
    crafted_path = str(tmp_path / "audiences_written_every_way.csv")
    cleaned_path = str(tmp_path / "audiences_written_every_way_clean.csv")
    pd.DataFrame([dict(audience_template, **{
        "airing_id": "AIR99200" + str(position), "audience_all": case[0],
        config.AUDIENCE_TARGET_COLUMN: case[1],
        config.TRANSMISSION_DAY_COLUMN: "Friday", config.START_TIME_COLUMN: "19:00",
        config.END_TIME_COLUMN: "20:00",
    }) for position, case in enumerate(audience_cases)]).to_csv(crafted_path, index=False)
    clean_audience_dataset(crafted_path, cleaned_path)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index("airing_id")
    for position, (everyone, under_18, everyone_thousands, under_18_thousands) in enumerate(
            audience_cases):
        airing = "AIR99200" + str(position)
        read_as = ((float(cleaned.loc[airing, "audience_all"] or "nan"),
                    float(cleaned.loc[airing, config.AUDIENCE_TARGET_COLUMN] or "nan"))
                   if airing in cleaned.index else "a removed row")
        assert read_as == (everyone_thousands, under_18_thousands), (
            "Audiences written "
            + repr(everyone)
            + " and "
            + repr(under_18)
            + " cleaned to "
            + str(read_as)
            + " where they are "
            + str(everyone_thousands)
            + " and "
            + str(under_18_thousands)
            + " thousand. Millions, thousands and viewers counted in full all become "
            "thousands, to "
            + str(config.AUDIENCE_DECIMALS)
            + " decimal place, so every audience is on one scale."
        )

    raw_submissions = raw_frames[config.RAW_SUBMISSIONS_EVAL_FILE]
    submission_template = raw_submissions[raw_submissions[config.SUBMISSION_TARGET_COLUMN]
                                          .str.strip().str.lower().isin(["yes", "no"])
                                          ].iloc[0].to_dict()
    fields = ["spots_booked", "channels_booked", "share_before_21_pct",
              "share_in_live_sport_pct", "agency_amendment_rate_pct", "supers_count"]
    submission_cases = [
        (["12 spots", "3 channels", "45%", "0%", "41.1%", "2"], [12, 3, 45, 0, 41.1, 2]),
        (["1 spot", "1 channel", "45 %", "20 %", "30 %", "0"], [1, 1, 45, 20, 30.0, 0]),
        (["8", "2", "60", "15", "44.8", "3"], [8, 2, 60, 15, 44.8, 3]),
    ]
    crafted_path = str(tmp_path / "counts_written_every_way.csv")
    cleaned_path = str(tmp_path / "counts_written_every_way_clean.csv")
    pd.DataFrame([dict(submission_template, **{"submission_id": "SUB99200" + str(position)},
                       **dict(zip(fields, case[0])))
                  for position, case in enumerate(submission_cases)]).to_csv(
        crafted_path, index=False)
    clean_submission_dataset(crafted_path, cleaned_path)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index(
        "submission_id")
    for position, (written, numbers) in enumerate(submission_cases):
        submission = "SUB99200" + str(position)
        read_as = ([float(cleaned.loc[submission, field] or "nan") for field in fields]
                   if submission in cleaned.index else "a removed row")
        assert read_as == [float(number) for number in numbers], (
            "The counts and shares written "
            + str(dict(zip(fields, written)))
            + " cleaned to "
            + str(read_as if isinstance(read_as, str) else dict(zip(fields, read_as)))
            + " where they are "
            + str(dict(zip(fields, numbers)))
            + ". A count written with its unit, or a share written with a per cent sign, is "
            "the number it states."
        )

    refusals = [
        ("submission", "spots_booked", "twelve"), ("submission", "supers_count", "2.5"),
        ("submission", "agency_amendment_rate_pct", "about 40%"),
        ("audience", config.AUDIENCE_TARGET_COLUMN, "1.2 million"),
        ("audience", "audience_all", "12%"),
    ]
    for position, (extract, field, written) in enumerate(refusals):
        refused_path = str(tmp_path / ("refused_number_" + str(position) + ".csv"))
        if extract == "submission":
            pd.DataFrame([dict(submission_template, **{"submission_id": "SUB99300" + str(position),
                                                       field: written})]).to_csv(
                refused_path, index=False)
            cleaner = clean_submission_dataset
        else:
            pd.DataFrame([dict(audience_template, **{"airing_id": "AIR99300" + str(position),
                                                     field: written})]).to_csv(
                refused_path, index=False)
            cleaner = clean_audience_dataset
        refused = False
        try:
            cleaner(refused_path, refused_path.replace(".csv", "_clean.csv"))
        except ValueError:
            refused = True
        assert refused, (
            field
            + " written "
            + repr(written)
            + " was cleaned without a ValueError. A value that is not a number in one of the "
            "notations the extracts use is refused, never guessed at or quietly dropped."
        )


def test_cleaning_repairs_facts_that_contradict_each_other_and_impossible_readings(
        raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.audience_cleaning import clean_audience_dataset
    from data_pipeline.submission_cleaning import clean_submission_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_submissions = raw_frames[config.RAW_SUBMISSIONS_EVAL_FILE]
    submission_template = dict(
        raw_submissions[raw_submissions[config.SUBMISSION_TARGET_COLUMN].str.strip().str.lower()
                        .isin(["yes", "no"])].iloc[0].to_dict(),
        **{config.CLOCK_NUMBER_COLUMN: "TST/PLNT020/030", config.COPY_LENGTH_COLUMN: "30",
           config.LEGAL_TEXT_SECONDS_COLUMN: "5", config.ADVERTISER_CAMPAIGNS_COLUMN: "10",
           config.ADVERTISER_AMENDMENTS_COLUMN: "2", "share_before_21_pct": "50",
           "channels_booked": "2", "spots_booked": "10", "agency_amendment_rate_pct": "30"})
    submission_cases = [
        ({config.LEGAL_TEXT_SECONDS_COLUMN: "40"}, config.LEGAL_TEXT_SECONDS_COLUMN, None,
         "legal text on screen for 40 seconds of a 30-second copy"),
        ({config.LEGAL_TEXT_SECONDS_COLUMN: "30"}, config.LEGAL_TEXT_SECONDS_COLUMN, 30.0,
         "legal text on screen for the whole of a 30-second copy, which is possible"),
        ({config.ADVERTISER_CAMPAIGNS_COLUMN: "4", config.ADVERTISER_AMENDMENTS_COLUMN: "9"},
         config.ADVERTISER_AMENDMENTS_COLUMN, None,
         "nine campaigns needing changes out of four campaigns"),
        ({config.ADVERTISER_CAMPAIGNS_COLUMN: "4", config.ADVERTISER_AMENDMENTS_COLUMN: "4"},
         config.ADVERTISER_AMENDMENTS_COLUMN, 4.0,
         "every one of four campaigns needing changes, which is possible"),
        ({"share_before_21_pct": "130%"}, "share_before_21_pct", None,
         "a share of 130 per cent"),
        ({"channels_booked": "6 channels"}, "channels_booked", None,
         "six channels booked where Harbourlight has four"),
        ({"spots_booked": "0 spots"}, "spots_booked", None, "a booking of no spots"),
        ({"agency_amendment_rate_pct": "104.5%"}, "agency_amendment_rate_pct", None,
         "an agency rate of 104.5 per cent"),
    ]
    crafted_path = str(tmp_path / "contradicting_submissions.csv")
    cleaned_path = str(tmp_path / "contradicting_submissions_clean.csv")
    pd.DataFrame([dict(submission_template, **{"submission_id": "SUB99400" + str(position)},
                       **case[0])
                  for position, case in enumerate(submission_cases)]).to_csv(
        crafted_path, index=False)
    clean_submission_dataset(crafted_path, cleaned_path)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index(
        "submission_id")
    for position, case in enumerate(submission_cases):
        field, wanted, described = case[1], case[2], case[3]
        submission = "SUB99400" + str(position)
        assert submission in cleaned.index, (
            "A submission with "
            + described
            + " was removed. A reading that cannot be true is made missing, and the "
            "submission with its answer is kept."
        )
        found = cleaned.loc[submission, field]
        read_as = float(found) if found != "" else None
        assert read_as == wanted, (
            "A submission with "
            + described
            + " cleaned "
            + field
            + " to "
            + repr(found)
            + " where it is "
            + ("left empty" if wanted is None else str(wanted))
            + ". A reading that cannot be true, or that contradicts another fact of the same "
            "submission, is made missing for the model to fill; one that can be true is kept "
            "as it is."
        )

    raw_audiences = raw_frames[config.RAW_AUDIENCES_EVAL_FILE]
    audience_template = dict(
        raw_audiences[raw_audiences[config.SCHEDULING_SYSTEM_COLUMN].str.strip() == "TXLine"]
        .iloc[0].to_dict(),
        **{"audience_all": "500.0", config.AUDIENCE_TARGET_COLUMN: "50.0",
           "forecast_under18_index": "80", "episode_number": "3"})
    audience_cases = [
        ({config.SCHEDULING_SYSTEM_COLUMN: "SlotMaster", config.TRANSMISSION_DAY_COLUMN: "Saturday",
          config.START_TIME_COLUMN: "23:30", config.END_TIME_COLUMN: "00:30"},
         {config.TRANSMISSION_DAY_COLUMN: "Saturday", config.START_TIME_COLUMN: "23:30",
          config.END_TIME_COLUMN: "24:30"},
         "a SlotMaster programme from 23:30 to 00:30, its end written on the next calendar day"),
        ({config.SCHEDULING_SYSTEM_COLUMN: "Airtime Planner",
          config.TRANSMISSION_DAY_COLUMN: "Friday", config.START_TIME_COLUMN: "22:45",
          config.END_TIME_COLUMN: "01:15"},
         {config.TRANSMISSION_DAY_COLUMN: "Friday", config.START_TIME_COLUMN: "22:45",
          config.END_TIME_COLUMN: "25:15"},
         "an Airtime Planner programme from 22:45 to 01:15"),
        ({config.SCHEDULING_SYSTEM_COLUMN: "TXLine", config.TRANSMISSION_DAY_COLUMN: "Friday",
          config.START_TIME_COLUMN: "23:00", config.END_TIME_COLUMN: "25:15"},
         {config.TRANSMISSION_DAY_COLUMN: "Friday", config.START_TIME_COLUMN: "23:00",
          config.END_TIME_COLUMN: "25:15"},
         "a TXLine programme from 23:00 to 25:15, already on the broadcast clock"),
        ({"forecast_under18_index": "0"}, {"forecast_under18_index": ""},
         "a forecast under-18 index of nought"),
        ({"episode_number": "0"}, {"episode_number": ""}, "an episode numbered nought"),
        ({"audience_all": "-50"}, {"audience_all": "", config.AUDIENCE_TARGET_COLUMN: "50.0"},
         "a whole audience of minus fifty thousand"),
    ]
    crafted = [dict(audience_template, **{"airing_id": "AIR99400" + str(position)}, **case[0])
               for position, case in enumerate(audience_cases)]
    crafted.append(dict(audience_template, **{"airing_id": "AIR99499",
                                              config.AUDIENCE_TARGET_COLUMN: "-3.2"}))
    crafted_path = str(tmp_path / "contradicting_airings.csv")
    cleaned_path = str(tmp_path / "contradicting_airings_clean.csv")
    pd.DataFrame(crafted).to_csv(crafted_path, index=False)
    clean_audience_dataset(crafted_path, cleaned_path)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False).set_index("airing_id")
    for position, case in enumerate(audience_cases):
        wanted, described = case[1], case[2]
        airing = "AIR99400" + str(position)
        found = ({field: cleaned.loc[airing, field] for field in wanted}
                 if airing in cleaned.index else "a removed row")
        agrees = found != "a removed row" and all(
            found[field] == value
            or (value.replace(".", "", 1).isdigit() and found[field].replace(".", "", 1).isdigit()
                and float(found[field]) == float(value))
            for field, value in wanted.items())
        assert agrees, (
            "An airing with "
            + described
            + " cleaned to "
            + str(found)
            + " where it is "
            + str(wanted)
            + " (an empty value meaning missing). A programme that runs past midnight on the "
            "calendar clock ends after 24:00 on the broadcast clock, and an impossible reading "
            "is made missing while the airing and its answer are kept."
        )
    assert "AIR99499" not in cleaned.index, (
        "An airing whose under-18 audience is minus 3.2 thousand was kept. An impossible "
        "answer is no answer, and a model fitted on it learns something that cannot happen."
    )
    refused_path = str(tmp_path / "ends_before_it_starts.csv")
    pd.DataFrame([dict(audience_template, **{
        "airing_id": "AIR99500", config.SCHEDULING_SYSTEM_COLUMN: "TXLine",
        config.TRANSMISSION_DAY_COLUMN: "Friday", config.START_TIME_COLUMN: "23:00",
        config.END_TIME_COLUMN: "22:00"})]).to_csv(refused_path, index=False)
    refused = False
    try:
        clean_audience_dataset(refused_path, refused_path.replace(".csv", "_clean.csv"))
    except ValueError:
        refused = True
    assert refused, (
        "A TXLine programme written from 23:00 to 22:00 was cleaned without a ValueError. "
        "TXLine writes the broadcast clock, on which every programme ends after it starts, so "
        "an end before the start cannot be repaired and is refused."
    )


def test_saved_feature_schemas_leave_out_the_answer_every_identifier_and_the_audience_seen_after(
        processed_frames, saved_models):
    from config import config

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    for schema_name, schema, target, identifiers, declared in [
            ("amendment", saved_models["amendment_schema"], config.SUBMISSION_TARGET_COLUMN,
             ["submission_id", config.CAMPAIGN_REFERENCE_COLUMN, config.CLOCK_NUMBER_COLUMN,
              "advertiser_code", "batch_code"],
             config.AMENDMENT_NUMERIC_FEATURES + config.AMENDMENT_CATEGORICAL_FEATURES),
            ("exposure", saved_models["exposure_schema"], config.AUDIENCE_TARGET_COLUMN,
             ["airing_id", "playout_code", "region_feed_code"],
             config.EXPOSURE_NUMERIC_FEATURES + config.EXPOSURE_CATEGORICAL_FEATURES)]:
        assert len(schema) > 0, (
            "The saved " + schema_name + " feature schema is empty, so nothing records which "
            "columns the model was fitted on."
        )
        assert target not in schema, (
            "The saved "
            + schema_name
            + " schema lists "
            + target
            + ", the answer the model predicts. A model shown its own answer scores almost "
            "perfectly and has learned nothing."
        )
        leaked = [column_name for column_name in identifiers if column_name in schema]
        assert len(leaked) == 0, (
            "The saved "
            + schema_name
            + " schema lists "
            + str(leaked)
            + ". An identifier says which record a row is, not what it means, and a model "
            "fitted on it memorises rows instead of learning from them."
        )
        assert len(schema) == len(set(schema)), (
            "The saved "
            + schema_name
            + " schema names a column twice: "
            + str(schema)
            + ". A row built to that schema no longer lines up with what the model was fitted on."
        )
        assert sorted(schema) == sorted(declared), (
            "The saved "
            + schema_name
            + " schema lists "
            + str(sorted(set(schema) - set(declared)))
            + " beyond the columns the configuration declares and leaves out "
            + str(sorted(set(declared) - set(schema)))
            + ". The inputs are declared, not guessed: every declared number and category "
            "column, and no other."
        )
    assert "audience_all" not in saved_models["exposure_schema"], (
        "The saved exposure schema lists audience_all, the whole audience of a programme. It "
        "is counted only after the programme has aired, so it is never known for the booking "
        "week the forecast is made for, and a model that leans on it forecasts nothing."
    )
    unknown = [column_name for column_name in saved_models["amendment_schema"]
               if column_name not in processed_frames[config.CLEAN_SUBMISSIONS_TRAIN_FILE].columns]
    assert len(unknown) == 0, (
        "The saved amendment schema lists "
        + str(unknown)
        + ", which "
        + config.CLEAN_SUBMISSIONS_TRAIN_FILE
        + " does not have, so no cleaned submission can be scored."
    )


def test_model_inputs_are_built_in_the_saved_schema_order(processed_frames, saved_models):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import add_slot_minutes, build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    slots = pd.DataFrame({config.START_TIME_COLUMN: ["06:00", "23:00", "25:30", "29:59"],
                          config.END_TIME_COLUMN: ["09:00", "25:30", "30:00", "30:00"],
                          "programme_title": ["A", "B", "C", "D"]})
    with_minutes = add_slot_minutes(slots.copy())
    measured = (list(with_minutes[config.START_MINUTE_COLUMN].astype(float)),
                list(with_minutes[config.DURATION_MINUTES_COLUMN].astype(float)))
    assert measured == ([360.0, 1380.0, 1530.0, 1799.0], [180.0, 150.0, 270.0, 1.0]), (
        "Programmes on the broadcast clock from 06:00 to 09:00, 23:00 to 25:30, 25:30 to 30:00 "
        "and 29:59 to 30:00 were given start minutes and durations of "
        + str(measured)
        + " where they start at minutes [360, 1380, 1530, 1799] of the clock and run for "
        "[180, 150, 270, 1] minutes. The broadcast clock runs on past 24:00, so a late programme "
        "starts later than an early one and none runs for a negative length."
    )
    assert (list(with_minutes[config.START_TIME_COLUMN]) == list(slots[config.START_TIME_COLUMN])
            and len(with_minutes) == len(slots)), (
        "add_slot_minutes changed the times or the rows it was given. It adds the start minute "
        "and the duration beside the times, which the schedule still shows."
    )
    for schema_name, frame, numeric_columns, category_columns in [
            ("amendment", processed_frames[config.CLEAN_SUBMISSIONS_EVAL_FILE].copy(),
             config.AMENDMENT_NUMERIC_FEATURES, config.AMENDMENT_CATEGORICAL_FEATURES),
            ("exposure",
             add_slot_minutes(processed_frames[config.CLEAN_AUDIENCES_EVAL_FILE].copy()),
             config.EXPOSURE_NUMERIC_FEATURES, config.EXPOSURE_CATEGORICAL_FEATURES)]:
        features = build_feature_frame(frame, numeric_columns, category_columns)
        built = list(features.columns)
        saved = list(saved_models[schema_name + "_schema"])
        assert built == saved, (
            "The "
            + schema_name
            + " features built from the cleaned held-out rows are "
            + str(built[:5])
            + " ... ("
            + str(len(built))
            + " columns) but the schema saved beside the model is "
            + str(saved[:5])
            + " ... ("
            + str(len(saved))
            + " columns). Same names in a different order score silently and wrongly."
        )
        assert len(features) == len(frame) and list(features.index) == list(frame.index), (
            "Building the "
            + schema_name
            + " inputs turned "
            + str(len(frame))
            + " rows into "
            + str(len(features))
            + ", or changed their order. Choosing columns must not drop, add or move rows."
        )
        not_numbers = [column_name for column_name in numeric_columns
                       if features[column_name].dtype.kind != "f"]
        assert len(not_numbers) == 0, (
            "The "
            + schema_name
            + " inputs carry "
            + str(not_numbers)
            + " as something other than floating-point numbers, so a gap cannot be told from a "
            "value."
        )
        written_blanks = frame.head(4).copy().reset_index(drop=True)
        written_blanks[numeric_columns[-1]] = pd.Series([2.5, None, pd.NA, np.nan], dtype=object)
        written_blanks[category_columns[0]] = pd.Series(["Planted", None, pd.NA, np.nan],
                                                        dtype=object)
        blank_features = build_feature_frame(written_blanks, numeric_columns, category_columns)
        numbers = list(blank_features[numeric_columns[-1]])
        categories = list(blank_features[category_columns[0]])
        assert (numbers[0] == 2.5 and all(np.isnan(value) for value in numbers[1:])
                and categories[0] == "Planted"
                and all(pd.isna(value) for value in categories[1:])), (
            "Written as [2.5, None, pd.NA, NaN], a number column came out of "
            "build_feature_frame as "
            + repr(numbers)
            + ", and written as ['Planted', None, pd.NA, NaN], a category column came out as "
            + repr(categories)
            + ". A value has to come out as itself and a gap as a gap: a live campaign reaches "
            "the model from a file, a database or the service, each writing a gap its own way, "
            "and the pipeline saved with the model handles a gap only if it still reads as one."
        )


def test_saved_models_score_rows_with_gaps_and_values_never_seen(processed_frames,
                                                                 saved_models):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import add_slot_minutes, build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    for model_name, frame, numeric_columns, category_columns in [
            ("amendment", processed_frames[config.CLEAN_SUBMISSIONS_EVAL_FILE].head(60).copy(),
             config.AMENDMENT_NUMERIC_FEATURES, config.AMENDMENT_CATEGORICAL_FEATURES),
            ("exposure",
             add_slot_minutes(processed_frames[config.CLEAN_AUDIENCES_EVAL_FILE].head(60).copy()),
             config.EXPOSURE_NUMERIC_FEATURES, config.EXPOSURE_CATEGORICAL_FEATURES)]:
        schema = saved_models[model_name + "_schema"]
        rows = build_feature_frame(frame, numeric_columns, category_columns)[schema].copy()
        rows.loc[rows.index[:20], numeric_columns[0]] = np.nan
        rows.loc[rows.index[20:40], category_columns[0]] = np.nan
        if isinstance(rows[category_columns[-1]].dtype, pd.CategoricalDtype):
            rows[category_columns[-1]] = rows[category_columns[-1]].cat.add_categories(
                ["A value no training row ever held"])
        rows.loc[rows.index[40:], category_columns[-1]] = "A value no training row ever held"
        model = saved_models[model_name]
        refusal = ""
        try:
            scores = (model.predict_proba(rows)[:, 1] if model_name == "amendment"
                      else model.predict(rows))
        except ValueError as error:
            scores, refusal = [], str(error)
        assert refusal == "", (
            "The saved "
            + model_name
            + " model refused rows with a gap or a value it never saw: "
            + refusal[:200]
            + ". A live campaign or programme arrives with gaps and new spellings, so the "
            "pipeline saved with the model has to fill every gap and ignore what it does not "
            "know."
        )
        assert len(scores) == len(rows), (
            "The saved "
            + model_name
            + " model returned "
            + str(len(scores))
            + " scores for "
            + str(len(rows))
            + " rows. Every row given has to get a score, or the scores are matched to the "
            "wrong rows."
        )
        assert np.isfinite(np.asarray(scores, dtype=float)).all(), (
            "The saved "
            + model_name
            + " model returned an empty or infinite score for a row with a gap or a value it "
            "never saw."
        )


def test_each_saved_model_was_fitted_on_the_columns_its_schema_names(saved_models):
    assert saved_models["problem"] is None, saved_models["problem"]
    for model_name in ["amendment", "exposure"]:
        model = saved_models[model_name]
        schema = [str(column_name) for column_name in saved_models[model_name + "_schema"]]
        fitted_on = getattr(model, "feature_names_in_", None)
        if fitted_on is None:
            assert getattr(model, "n_features_in_", None) == len(schema), (
                "The saved "
                + model_name
                + " model records neither the names nor the number of the columns it was "
                "fitted on, or the number is not the "
                + str(len(schema))
                + " its saved schema lists, so nothing shows that the schema describes it."
            )
        else:
            fitted_names = [str(column_name) for column_name in fitted_on]
            assert fitted_names == schema, (
                "The saved "
                + model_name
                + " model was fitted on "
                + str(fitted_names[:5])
                + " ... ("
                + str(len(fitted_names))
                + " columns) but the schema saved beside it lists "
                + str(schema[:5])
                + " ... ("
                + str(len(schema))
                + " columns). Live rows are built to the schema, so a model fitted on other "
                "columns, or on the same ones in another order, scores them wrongly or not at "
                "all."
            )
    classes = getattr(saved_models["amendment"], "classes_", None)
    assert classes is not None and list(classes) == [0, 1], (
        "The saved amendment model's classes are "
        + str(None if classes is None else list(classes))
        + ". Every tool reads the second column of its probabilities as the chance that the "
        "copy needs changes, which holds only when the classes are 0 and 1 in that order."
    )


def test_the_quality_report_accounts_for_every_row_it_received(raw_frames, processed_frames):
    import json
    import os
    import pathlib

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.quality_profiler import profile_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    planted_raw = pd.DataFrame({"record_id": ["A", "A", "B", "B", "C"],
                                "reading": ["1", "1", None, None, "3"],
                                "answer": ["Yes", "Yes", None, None, "No"]})
    planted_clean = pd.DataFrame({"record_id": ["A", "C"], "reading": ["1", None],
                                  "answer": ["Yes", "No"]})
    planted_expected = {"rows_received": 5, "rows_kept": 2, "rows_removed": 3,
                        "duplicate_rows": 2, "rows_missing_target": 1, "columns": 3,
                        "missing_cells_as_received": {"reading": 2, "answer": 2},
                        "missing_cells_after_cleaning": {"reading": 1}}
    planted_profile = profile_dataset(planted_raw, planted_clean, "answer")
    planted_found = {field_name: planted_profile.get(field_name)
                     for field_name in planted_expected}
    assert planted_found == planted_expected, (
        "Profiling a planted extract of five rows, two of them repeats of the row before and "
        "one first copy with no answer, against a cleaned table of two rows with one gap, gave "
        + str(planted_found)
        + " where the rows give "
        + str(planted_expected)
        + ". A repeat is counted for every copy after the first, a missing answer only on a "
        "first copy, and each gap in the column that holds it."
    )
    assert os.path.isfile(config.DATA_QUALITY_REPORT_FILE), (
        "There is no quality report at "
        + config.DATA_QUALITY_REPORT_FILE
        + ". The data pipeline writes one beside the cleaned files to say what cleaning did."
    )
    with open(config.DATA_QUALITY_REPORT_FILE, encoding="utf-8") as report_file:
        report = json.load(report_file)
    for raw_path, clean_path, target, removal_field, defect_fields, profiled_again in [
            (config.RAW_SUBMISSIONS_TRAIN_FILE, config.CLEAN_SUBMISSIONS_TRAIN_FILE,
             config.SUBMISSION_TARGET_COLUMN, None,
             ["legal_text_longer_than_copy", "amendments_exceeding_campaigns"], False),
            (config.RAW_SUBMISSIONS_EVAL_FILE, config.CLEAN_SUBMISSIONS_EVAL_FILE,
             config.SUBMISSION_TARGET_COLUMN, None,
             ["legal_text_longer_than_copy", "amendments_exceeding_campaigns"], True),
            (config.RAW_AUDIENCES_TRAIN_FILE, config.CLEAN_AUDIENCES_TRAIN_FILE,
             config.AUDIENCE_TARGET_COLUMN, "rows_with_impossible_target", [], False),
            (config.RAW_AUDIENCES_EVAL_FILE, config.CLEAN_AUDIENCES_EVAL_FILE,
             config.AUDIENCE_TARGET_COLUMN, "rows_with_impossible_target", [], True),
            (config.RAW_CLEARANCE_NOTES_FILE, config.CLEAN_CLEARANCE_NOTES_FILE,
             config.NOTE_TARGET_COLUMN, "rows_with_too_few_words",
             ["note_lengths_disagreeing_with_plain_text"], True)]:
        name = pathlib.Path(raw_path).stem
        assert name in report, (
            "The quality report has no entry for "
            + name
            + ", so nothing says what cleaning did to "
            + raw_path
            + "."
        )
        entry = report[name]
        clean_frame = processed_frames[clean_path]
        file_facts = {"rows_received": len(raw_frames[raw_path]),
                      "rows_kept": len(clean_frame),
                      "rows_removed": len(raw_frames[raw_path]) - len(clean_frame),
                      "columns": int(clean_frame.shape[1])}
        profiled = (profile_dataset(load_raw_dataset(raw_path), clean_frame, target)
                    if profiled_again else dict(entry, **file_facts))
        missing = sorted((set(planted_expected) - set(entry))
                         | (set(planted_expected) - set(profiled)))
        assert len(missing) == 0, (
            "The quality report entry for "
            + name
            + ", or the profile worked out for it, has no "
            + str(missing)
            + ", so what cleaning did to that extract cannot be read."
        )
        for field_name in planted_expected:
            wanted = file_facts.get(field_name, profiled[field_name])
            assert entry[field_name] == profiled[field_name] == wanted, (
                "The quality report says "
                + field_name
                + " is "
                + str(entry[field_name])[:200]
                + " for "
                + name
                + (" and profiling the extract again gives " + str(profiled[field_name])[:200]
                   if profiled_again else "")
                + (", but the files say " + str(wanted) if field_name in file_facts else "")
                + ". A report that does not match the data it describes cannot be used to "
                "defend what the pipeline did."
            )
        assert entry["duplicate_rows"] > 0 and entry["rows_missing_target"] > 0, (
            "The quality report gives "
            + name
            + " "
            + str(entry["duplicate_rows"])
            + " repeated rows and "
            + str(entry["rows_missing_target"])
            + " rows with no answer, where the extract arrives with both."
        )
        removed_for_other_reasons = 0
        if removal_field is not None:
            assert removal_field in entry and entry[removal_field] > 0, (
                "The quality report entry for "
                + name
                + " does not count "
                + removal_field
                + ", although rows of that kind are removed from it."
            )
            removed_for_other_reasons = entry[removal_field]
        assert entry["rows_removed"] == (entry["duplicate_rows"] + entry["rows_missing_target"]
                                         + removed_for_other_reasons), (
            "The quality report says "
            + str(entry["rows_removed"])
            + " rows were removed from "
            + name
            + " but gives reasons for "
            + str(entry["duplicate_rows"] + entry["rows_missing_target"]
                  + removed_for_other_reasons)
            + ". Every row removed is accounted for: repeated, no answer, an impossible answer, "
            "or a note too short to learn from."
        )
        for defect_field in defect_fields:
            as_received = entry.get(defect_field + "_as_received")
            after_cleaning = entry.get(defect_field + "_after_cleaning")
            assert as_received is not None and as_received > 0 and after_cleaning == 0, (
                "The quality report gives "
                + defect_field
                + "_as_received as "
                + str(as_received)
                + " and "
                + defect_field
                + "_after_cleaning as "
                + str(after_cleaning)
                + " for "
                + name
                + ". The extract arrives with the defect and the cleaned file carries none of "
                "it, and the report shows both under those two names."
            )


def test_amendment_model_ranks_held_out_submissions_above_the_configured_minimum(
        processed_frames, saved_models):
    import copy
    import json
    import os
    import re

    import numpy as np
    from sklearn.metrics import roc_auc_score

    from config import config
    from data_pipeline.feature_builder import build_feature_frame
    from modeling.model_evaluation import evaluate_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    train_frame = processed_frames[config.CLEAN_SUBMISSIONS_TRAIN_FILE].head(6000)
    held_out_frame = processed_frames[config.CLEAN_SUBMISSIONS_EVAL_FILE]
    schema = saved_models["amendment_schema"]
    train_features = build_feature_frame(train_frame.copy(), config.AMENDMENT_NUMERIC_FEATURES,
                                         config.AMENDMENT_CATEGORICAL_FEATURES)[schema]
    held_out_features = build_feature_frame(held_out_frame.copy(),
                                            config.AMENDMENT_NUMERIC_FEATURES,
                                            config.AMENDMENT_CATEGORICAL_FEATURES)[schema]
    train_answers = train_frame[config.SUBMISSION_TARGET_COLUMN].map(
        config.SUBMISSION_TARGET_ENCODING).astype(int)
    held_out_answers = held_out_frame[config.SUBMISSION_TARGET_COLUMN].map(
        config.SUBMISSION_TARGET_ENCODING).astype(int)
    model = saved_models["amendment"]
    scores = evaluate_classifier(copy.deepcopy(model), train_features, train_answers,
                                 held_out_features, held_out_answers)
    missing = sorted({"held_out_roc_auc", "train_roc_auc", "generalisation_gap"} - set(scores))
    assert len(missing) == 0, (
        "evaluate_classifier returned no "
        + str(missing)
        + ", so the amendment model's scores cannot be read."
    )
    assert scores["held_out_roc_auc"] >= config.MINIMUM_AMENDMENT_ROC_AUC, (
        "The amendment model scored "
        + str(scores["held_out_roc_auc"])
        + " ROC AUC on the held-out submissions, below the "
        + str(config.MINIMUM_AMENDMENT_ROC_AUC)
        + " the configuration sets. The bar sits well below what a sound model reaches."
    )
    probabilities = model.predict_proba(held_out_features)[:, 1]
    measured = roc_auc_score(held_out_answers, probabilities)
    assert abs(scores["held_out_roc_auc"] - measured) < 0.006, (
        "The evaluation reports a ROC AUC of "
        + str(scores["held_out_roc_auc"])
        + " for the held-out submissions, but the model's own probabilities give "
        + str(round(measured, 4))
        + ". evaluate_classifier scores the fitted model it is given, without fitting it "
        "again, on the held-out rows and from probabilities."
    )
    trained_on = roc_auc_score(train_answers, model.predict_proba(train_features)[:, 1])
    assert abs(scores["train_roc_auc"] - trained_on) < 0.006, (
        "The evaluation reports a training ROC AUC of "
        + str(scores["train_roc_auc"])
        + ", but the model's own probabilities on the training rows it was given give "
        + str(round(trained_on, 4))
        + ". The training score is measured on the training rows, or the gap cannot show how "
        "much the model memorised."
    )
    assert abs(scores["generalisation_gap"]
               - (scores["train_roc_auc"] - scores["held_out_roc_auc"])) < 0.011, (
        "The reported gap "
        + str(scores["generalisation_gap"])
        + " is not the training score "
        + str(scores["train_roc_auc"])
        + " less the held-out score "
        + str(scores["held_out_roc_auc"])
        + "."
    )
    answers = np.asarray(held_out_answers)
    amended = float(probabilities[answers == 1].mean())
    cleared = float(probabilities[answers == 0].mean())
    assert amended > cleared, (
        "Copies that needed changes average "
        + str(round(amended, 4))
        + " and copies cleared as submitted "
        + str(round(cleared, 4))
        + ". The model scores the wrong way round."
    )
    assert abs(float(probabilities.mean()) - float(answers.mean())) < 0.05, (
        "The held-out submissions average a probability of "
        + str(round(float(probabilities.mean()), 4))
        + " that the copy needs changes, where "
        + str(round(float(answers.mean()), 4))
        + " of them did. The score is read as a probability and banded at fixed thresholds, so "
        "a model that leans one way puts copies in the wrong band."
    )
    assert os.path.isfile(config.MODEL_METRICS_FILE), (
        "There is no metrics file at "
        + config.MODEL_METRICS_FILE
        + ". Training writes the scores it measured beside the models it saved."
    )
    with open(config.MODEL_METRICS_FILE, encoding="utf-8") as metrics_file:
        recorded = json.dumps(json.load(metrics_file))
    figures = [float(figure) for figure in re.findall(r"-?\d+\.\d+(?:[eE]-?\d+)?", recorded)]
    assert any(abs(figure - measured) < 0.0006 for figure in figures), (
        config.MODEL_METRICS_FILE
        + " does not record the held-out ROC AUC of "
        + str(round(measured, 4))
        + " that the saved amendment model scores, so the metrics describe some other model."
    )


def test_exposure_model_forecasts_held_out_audiences_better_than_their_average(
        processed_frames, saved_models):
    import copy

    import numpy as np
    from sklearn.metrics import r2_score

    from config import config
    from data_pipeline.feature_builder import add_slot_minutes, build_feature_frame
    from modeling.model_evaluation import evaluate_regressor

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    train_frame = add_slot_minutes(processed_frames[config.CLEAN_AUDIENCES_TRAIN_FILE].head(3000)
                                   .copy())
    held_out_frame = add_slot_minutes(processed_frames[config.CLEAN_AUDIENCES_EVAL_FILE].copy())
    schema = saved_models["exposure_schema"]
    train_features = build_feature_frame(train_frame, config.EXPOSURE_NUMERIC_FEATURES,
                                         config.EXPOSURE_CATEGORICAL_FEATURES)[schema]
    held_out_features = build_feature_frame(held_out_frame, config.EXPOSURE_NUMERIC_FEATURES,
                                            config.EXPOSURE_CATEGORICAL_FEATURES)[schema]
    train_viewers = train_frame[config.AUDIENCE_TARGET_COLUMN]
    held_out_viewers = held_out_frame[config.AUDIENCE_TARGET_COLUMN]
    model = saved_models["exposure"]
    scores = evaluate_regressor(copy.deepcopy(model), train_features, train_viewers,
                                held_out_features, held_out_viewers)
    missing = sorted({"held_out_r2", "train_r2", "generalisation_gap", "mean_absolute_error",
                      "average_estimate_mean_absolute_error", "lowest_prediction"}
                     - set(scores))
    assert len(missing) == 0, (
        "evaluate_regressor returned no "
        + str(missing)
        + ", so the exposure model's scores cannot be read."
    )
    assert scores["held_out_r2"] >= config.MINIMUM_EXPOSURE_R2, (
        "The exposure model scored "
        + str(scores["held_out_r2"])
        + " R squared on the held-out airings, below the "
        + str(config.MINIMUM_EXPOSURE_R2)
        + " the configuration sets."
    )
    predictions = np.asarray(model.predict(held_out_features), dtype=float)
    measured = r2_score(held_out_viewers, predictions)
    assert abs(scores["held_out_r2"] - measured) < 0.006, (
        "The evaluation reports R squared "
        + str(scores["held_out_r2"])
        + ", but the model's own forecasts for the held-out airings give "
        + str(round(measured, 4))
        + ". evaluate_regressor scores the fitted model it is given, without fitting it again."
    )
    trained_on = r2_score(train_viewers, np.asarray(model.predict(train_features), dtype=float))
    assert (abs(scores["train_r2"] - trained_on) < 0.006
            and abs(scores["generalisation_gap"]
                    - (scores["train_r2"] - scores["held_out_r2"])) < 0.011), (
        "The evaluation reports a training R squared of "
        + str(scores["train_r2"])
        + " and a gap of "
        + str(scores["generalisation_gap"])
        + ", but the model's own forecasts for the training rows it was given give "
        + str(round(trained_on, 4))
        + ". The training score is measured on the training rows and the gap is that less the "
        "held-out score."
    )
    average_error = float(np.mean(np.abs(np.asarray(held_out_viewers, dtype=float)
                                         - float(np.mean(np.asarray(train_viewers,
                                                                    dtype=float))))))
    assert abs(scores["average_estimate_mean_absolute_error"] - average_error) < 0.051, (
        "The evaluation reports that always answering the average misses by "
        + str(scores["average_estimate_mean_absolute_error"])
        + " thousand viewers, but always answering the average of the training audiences it was "
        "given misses the held-out airings by "
        + str(round(average_error, 2))
        + " thousand on average."
    )
    assert scores["mean_absolute_error"] < scores["average_estimate_mean_absolute_error"], (
        "The model misses by "
        + str(scores["mean_absolute_error"])
        + " thousand viewers on average, no better than the "
        + str(scores["average_estimate_mean_absolute_error"])
        + " thousand missed by always answering the training average."
    )
    assert abs(scores["lowest_prediction"] - float(predictions.min())) < 0.051, (
        "The evaluation reports a lowest forecast of "
        + str(scores["lowest_prediction"])
        + " thousand, but the model's lowest forecast for the held-out airings is "
        + str(round(float(predictions.min()), 2))
        + "."
    )


def test_amendment_score_rises_with_a_health_claim_and_with_offer_terms_left_off_screen(
        processed_frames, saved_models):
    import numpy as np

    from config import config
    from data_pipeline.feature_builder import build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    schema = saved_models["amendment_schema"]
    model = saved_models["amendment"]
    held_out = processed_frames[config.CLEAN_SUBMISSIONS_EVAL_FILE]
    as_submitted = model.predict_proba(build_feature_frame(
        held_out.head(3000).copy(), config.AMENDMENT_NUMERIC_FEATURES,
        config.AMENDMENT_CATEGORICAL_FEATURES)[schema])[:, 1]
    assert float(np.std(as_submitted)) > 0.05, (
        "The amendment model scores about the same for every copy, with a spread of "
        + str(round(float(np.std(as_submitted)), 4))
        + ". A score that ignores the copy cannot say which campaigns need an executive first."
    )
    with_offers = held_out[held_out["offer_terms_shown_flag"].notna()].head(2000)
    assert len(with_offers) > 200, (
        "Only "
        + str(len(with_offers))
        + " held-out copies make an offer, too few to compare."
    )
    for change, riskier, safer in [
            ("a health claim made rather than none",
             held_out.head(3000).assign(health_claim_flag=config.BOOLEAN_YES),
             held_out.head(3000).assign(health_claim_flag=config.BOOLEAN_NO)),
            ("an offer whose terms are left off screen rather than shown",
             with_offers.assign(offer_terms_shown_flag=config.BOOLEAN_NO),
             with_offers.assign(offer_terms_shown_flag=config.BOOLEAN_YES))]:
        averages = [float(model.predict_proba(build_feature_frame(
            frame.copy(), config.AMENDMENT_NUMERIC_FEATURES,
            config.AMENDMENT_CATEGORICAL_FEATURES)[schema])[:, 1].mean())
            for frame in [riskier, safer]]
        assert averages[0] > averages[1], (
            "With "
            + change
            + " the same copies average an amendment score of "
            + str(round(averages[0], 4))
            + ", against "
            + str(round(averages[1], 4))
            + " otherwise. A health claim needs evidence and an offer needs its significant "
            "terms on screen, so these are the copies a clearance desk most often sends back; "
            "the model has not learned what makes a copy need changes."
        )


def test_exposure_forecast_rises_in_school_holidays_and_falls_late_at_night(
        processed_frames, saved_models):
    import numpy as np

    from config import config
    from data_pipeline.feature_builder import add_slot_minutes, build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    schema = saved_models["exposure_schema"]
    model = saved_models["exposure"]
    airings = processed_frames[config.CLEAN_AUDIENCES_EVAL_FILE].head(2000)
    as_aired = model.predict(build_feature_frame(
        add_slot_minutes(airings.copy()), config.EXPOSURE_NUMERIC_FEATURES,
        config.EXPOSURE_CATEGORICAL_FEATURES)[schema])
    assert float(np.std(as_aired)) > 5, (
        "The exposure model forecasts about the same audience for every programme, with a "
        "spread of "
        + str(round(float(np.std(as_aired)), 2))
        + " thousand. A forecast that ignores the programme cannot rank campaigns by who they "
        "would reach."
    )
    for change, larger, smaller in [
            ("in the school holidays rather than in term time",
             airings.assign(school_holiday_flag=config.BOOLEAN_YES),
             airings.assign(school_holiday_flag=config.BOOLEAN_NO)),
            ("at 19:00 to 20:00 rather than at 25:00 to 26:00 on the broadcast clock",
             airings.assign(**{config.START_TIME_COLUMN: "19:00", config.END_TIME_COLUMN: "20:00"}),
             airings.assign(**{config.START_TIME_COLUMN: "25:00",
                               config.END_TIME_COLUMN: "26:00"})),
            ("with a forecast under-18 index fifty points higher",
             airings.assign(forecast_under18_index=airings["forecast_under18_index"] + 50),
             airings)]:
        averages = [float(np.mean(model.predict(build_feature_frame(
            add_slot_minutes(frame.copy()), config.EXPOSURE_NUMERIC_FEATURES,
            config.EXPOSURE_CATEGORICAL_FEATURES)[schema]))) for frame in [larger, smaller]]
        assert averages[0] > averages[1], (
            "Shown "
            + change
            + ", the same programmes average a forecast of "
            + str(round(averages[0], 2))
            + " thousand viewers under 18, against "
            + str(round(averages[1], 2))
            + " otherwise. Children watch more in the holidays, far fewer are up in the small "
            "hours, and the planners' index says who a programme draws, so the model has not "
            "learned what decides an under-18 audience."
        )


def test_amendment_facts_band_each_score_by_the_configured_thresholds(open_queue, saved_models):
    import numpy as np

    from config import config
    from data_pipeline.feature_builder import build_feature_frame
    from tools.prediction_tools import build_amendment_facts

    assert open_queue["problem"] is None, open_queue["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    campaigns = open_queue["campaigns"]
    facts = build_amendment_facts(saved_models["amendment"], saved_models["amendment_schema"],
                                  campaigns.copy())
    assert len(facts) == len(campaigns), (
        "Amendment facts were built for "
        + str(len(facts))
        + " of the "
        + str(len(campaigns))
        + " campaigns given, so the facts no longer line up with the campaigns they describe."
    )
    missing = sorted({"amendment_score", "amendment_band"} - set(facts[0]))
    assert len(missing) == 0, (
        "The amendment facts carry no "
        + str(missing)
        + ", so the agents and the queue cannot read them."
    )
    features = build_feature_frame(campaigns.copy(), config.AMENDMENT_NUMERIC_FEATURES,
                                   config.AMENDMENT_CATEGORICAL_FEATURES)[
        saved_models["amendment_schema"]]
    probabilities = saved_models["amendment"].predict_proba(features)[:, 1]
    bands_seen = set()
    for position, (fact, probability) in enumerate(zip(facts, probabilities)):
        reference = str(campaigns[config.CAMPAIGN_REFERENCE_COLUMN].iloc[position])
        assert abs(fact["amendment_score"] - float(probability)) < 0.00051, (
            reference
            + " was given an amendment score of "
            + str(fact["amendment_score"])
            + " where the saved model scores it "
            + str(round(float(probability), 4))
            + "."
        )
        expected_bands = {config.AMENDMENT_BAND_HIGH
                          if value >= config.AMENDMENT_BAND_HIGH_THRESHOLD
                          else config.AMENDMENT_BAND_MEDIUM
                          if value >= config.AMENDMENT_BAND_MEDIUM_THRESHOLD
                          else config.AMENDMENT_BAND_LOW
                          for value in [float(probability), fact["amendment_score"]]}
        assert fact["amendment_band"] in expected_bands, (
            reference
            + " scored "
            + str(fact["amendment_score"])
            + " and was banded "
            + str(fact["amendment_band"])
            + " where the thresholds "
            + str(config.AMENDMENT_BAND_MEDIUM_THRESHOLD)
            + " and "
            + str(config.AMENDMENT_BAND_HIGH_THRESHOLD)
            + " put it in "
            + " or ".join(sorted(expected_bands))
            + ". A score sitting on a threshold belongs to the band above it."
        )
        bands_seen.add(fact["amendment_band"])
    assert bands_seen == {config.AMENDMENT_BAND_HIGH, config.AMENDMENT_BAND_MEDIUM,
                          config.AMENDMENT_BAND_LOW}, (
        "Across the live queue only "
        + str(sorted(bands_seen))
        + " occur. The queue holds campaigns of every band, so the scores are not separating "
        "them."
    )
    assert float(np.std(probabilities)) > 0.05, (
        "Every live campaign scores about the same, so the scores cannot tell a copy likely to "
        "need changes from one that will be cleared as submitted."
    )


def test_issue_classifier_clears_the_configured_macro_f1(processed_frames, saved_models):
    import copy

    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    from config import config
    from modeling.model_evaluation import evaluate_text_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    notes = processed_frames[config.CLEAN_CLEARANCE_NOTES_FILE]
    train_text, held_out_text, train_answers, held_out_answers = train_test_split(
        notes[config.NOTE_TEXT_COLUMN], notes[config.NOTE_TARGET_COLUMN],
        test_size=config.NOTE_HELD_OUT_SHARE, stratify=notes[config.NOTE_TARGET_COLUMN],
        random_state=config.RANDOM_SEED)
    model = saved_models["issue"]
    scores = evaluate_text_classifier(copy.deepcopy(model), train_text.head(3000),
                                      train_answers.head(3000), held_out_text, held_out_answers)
    missing = sorted({"held_out_macro_f1", "train_macro_f1", "generalisation_gap",
                      "per_class_f1"} - set(scores))
    assert len(missing) == 0, (
        "evaluate_text_classifier returned no "
        + str(missing)
        + ", so the issue classifier's scores cannot be read."
    )
    assert scores["held_out_macro_f1"] >= config.MINIMUM_ISSUE_MACRO_F1, (
        "The issue classifier scored "
        + str(scores["held_out_macro_f1"])
        + " macro F1 on the held-out notes, below the "
        + str(config.MINIMUM_ISSUE_MACRO_F1)
        + " the configuration sets. Macro F1 weighs the rarest issue as much as the commonest, "
        "which is the point: the rare issues are the ones a busy desk misreads."
    )
    measured = f1_score(held_out_answers, model.predict(held_out_text), average="macro")
    assert abs(scores["held_out_macro_f1"] - measured) < 0.006, (
        "The evaluation reports macro F1 "
        + str(scores["held_out_macro_f1"])
        + ", but the classifier's own predictions on the held-out notes give "
        + str(round(measured, 4))
        + ". evaluate_text_classifier scores the fitted classifier it is given, without fitting "
        "it again."
    )
    trained_on = f1_score(train_answers.head(3000), model.predict(train_text.head(3000)),
                          average="macro")
    assert (abs(scores["train_macro_f1"] - trained_on) < 0.006
            and abs(scores["generalisation_gap"]
                    - (scores["train_macro_f1"] - scores["held_out_macro_f1"])) < 0.011), (
        "The evaluation reports a training macro F1 of "
        + str(scores["train_macro_f1"])
        + " and a gap of "
        + str(scores["generalisation_gap"])
        + ", but the classifier's own predictions on the training notes it was given give "
        + str(round(trained_on, 4))
        + ". The training score is measured on the training notes and the gap is that less the "
        "held-out score."
    )
    assert sorted(scores["per_class_f1"]) == sorted(config.GUIDE_DOCUMENT_BY_ISSUE), (
        "The per-issue scores cover "
        + str(sorted(scores["per_class_f1"]))
        + " where the desk codes to "
        + str(sorted(config.GUIDE_DOCUMENT_BY_ISSUE))
        + ". An issue left out is an issue whose score nobody sees."
    )


def test_issue_predictions_stay_inside_the_vocabulary_and_cover_every_issue(processed_frames,
                                                                            saved_models):
    from sklearn.model_selection import train_test_split

    from config import config

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    notes = processed_frames[config.CLEAN_CLEARANCE_NOTES_FILE]
    split = train_test_split(
        notes[config.NOTE_TEXT_COLUMN], notes[config.NOTE_TARGET_COLUMN],
        test_size=config.NOTE_HELD_OUT_SHARE, stratify=notes[config.NOTE_TARGET_COLUMN],
        random_state=config.RANDOM_SEED)
    held_out_text, held_out_answers = split[1], split[3]
    model = saved_models["issue"]
    issues = sorted(config.GUIDE_DOCUMENT_BY_ISSUE)
    assert sorted(str(label) for label in model.classes_) == issues, (
        "The issue classifier knows the issues "
        + str(sorted(str(label) for label in model.classes_))
        + " where the desk codes to "
        + str(issues)
        + "."
    )
    predicted = [str(label) for label in model.predict(held_out_text)]
    strays = sorted(set(predicted) - set(issues))
    assert len(strays) == 0, (
        "The issue classifier predicted "
        + str(strays)
        + ", which the configuration does not list. Every issue downstream is looked up by "
        "name, its guide included."
    )
    never = sorted(set(issues) - set(predicted))
    assert len(never) == 0, (
        "Across "
        + str(len(predicted))
        + " held-out notes the classifier never once predicted "
        + str(never)
        + ", so those campaigns are always given another issue's guide."
    )
    for issue in issues:
        given = [label for label, truth in zip(predicted, held_out_answers) if truth == issue]
        commonest = max(set(given), key=given.count)
        assert commonest == issue, (
            "Notes about "
            + issue
            + " are most often read as "
            + commonest
            + ". A classifier that confuses an issue with another more often than it names it "
            "has not learned that issue."
        )
    unseen = ["Zyxwv qrstu plmok wvbnm ghjkl asdfg.", "The the the the the the the."]
    read_unseen = [str(label) for label in model.predict(unseen)]
    assert len(read_unseen) == 2 and set(read_unseen) <= set(issues), (
        "Notes made of words the classifier never saw were read as "
        + str(read_unseen)
        + ". A note written in new words still comes back as one of the desk's issues."
    )


def test_note_cleaning_leaves_plain_words_whatever_markup_or_capitals_the_note_came_in(
        raw_frames, processed_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.note_cleaning import clean_note_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    raw_notes = raw_frames[config.RAW_CLEARANCE_NOTES_FILE].drop_duplicates(subset=["note_id"])
    written = raw_notes[config.NOTE_TEXT_COLUMN].str.strip()
    answered = ~raw_notes[config.NOTE_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    plain_notes = raw_notes[answered & ~written.str.contains("[<>&\xa0]", regex=True)
                            & (written != written.str.upper())
                            & (written.str.split().str.len() >= 8)].head(150)
    variants = []
    for note_row in plain_notes.to_dict("records"):
        words = note_row[config.NOTE_TEXT_COLUMN].split()
        marked_up = ("<p>" + "&nbsp;".join(words[:2]) + " <b>" + " ".join(words[2:4])
                     + "</b> <br/> " + " ".join(words[4:]).replace("'", "&#39;") + "</p>")
        note_id = str(note_row["note_id"]).strip()
        variants.append(dict(note_row, **{"note_id": "MARKED-" + note_id,
                                          config.NOTE_TEXT_COLUMN: marked_up}))
        variants.append(dict(note_row, **{"note_id": "CAPITALS-" + note_id,
                                          config.NOTE_TEXT_COLUMN:
                                              note_row[config.NOTE_TEXT_COLUMN].upper()}))
    first_row = plain_notes.iloc[0].to_dict()
    first_id = str(first_row["note_id"]).strip()
    for prefix, timestamp in [("THURSDAY-", "THU 09:05"), ("SATURDAY-", "sa. 17:45")]:
        variants.append(dict(first_row, **{"note_id": prefix + first_id,
                                           config.NOTE_TIMESTAMP_COLUMN: timestamp}))
    variants.append(dict(first_row, **{"note_id": "SHORT-" + first_id,
                                       config.NOTE_TEXT_COLUMN: "Chased the agency again today."}))
    sample_path = str(tmp_path / "notes_raw_sample.csv")
    cleaned_path = str(tmp_path / "notes_cleaned_sample.csv")
    pd.concat([plain_notes, pd.DataFrame(variants)]).to_csv(sample_path, index=False)
    clean_note_dataset(sample_path, cleaned_path)
    cleaned_sample = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False)
    by_id = cleaned_sample.set_index("note_id")
    changed = []
    for note_id in plain_notes["note_id"].astype(str).str.strip():
        as_written = (by_id.loc[note_id, config.NOTE_TEXT_COLUMN] if note_id in by_id.index
                      else None)
        for prefix, wanted in [("MARKED-", as_written),
                               ("CAPITALS-", None if as_written is None else as_written.lower())]:
            found = (by_id.loc[prefix + note_id, config.NOTE_TEXT_COLUMN]
                     if prefix + note_id in by_id.index else None)
            if found != wanted or wanted is None:
                changed.append((prefix + note_id, found, wanted))
    assert len(plain_notes) >= 100 and len(changed) == 0, (
        str(len(changed))
        + " of "
        + str(2 * len(plain_notes))
        + " notes rewritten with markup and escaped characters, or in capitals throughout, did "
        "not clean to what the same note written plainly cleans to, for example "
        + str(changed[:2])
        + ". Markup comes out, escaped characters are put back and a note written in capitals "
        "is kept in lower case, so the words the classifier reads do not depend on how the "
        "note was written."
    )
    for prefix, wanted in [("THURSDAY-", "Thursday 09:05"), ("SATURDAY-", "Saturday 17:45")]:
        found = (by_id.loc[prefix + first_id, config.NOTE_TIMESTAMP_COLUMN]
                 if prefix + first_id in by_id.index else "a removed note")
        assert found == wanted, (
            "A note time written "
            + ("'THU 09:05'" if prefix == "THURSDAY-" else "'sa. 17:45'")
            + " was cleaned to "
            + repr(found)
            + ", where every note time is the weekday written in full and the time, "
            + repr(wanted)
            + "."
        )
    assert "SHORT-" + first_id not in by_id.index, (
        "The note 'Chased the agency again today.' was kept. A note of fewer than "
        + str(config.MINIMUM_NOTE_WORDS)
        + " words is too short to learn from and is removed."
    )
    refused_path = str(tmp_path / "note_time_unreadable.csv")
    pd.DataFrame([dict(first_row, **{"note_id": "TIMELESS-" + first_id,
                                     config.NOTE_TIMESTAMP_COLUMN: "Funday 10:00"})]).to_csv(
        refused_path, index=False)
    refused = False
    try:
        clean_note_dataset(refused_path, refused_path.replace(".csv", "_clean.csv"))
    except ValueError:
        refused = True
    assert refused, (
        "A note written at 'Funday 10:00' was cleaned without a ValueError. A time whose "
        "weekday cannot be read is refused rather than guessed."
    )
    for source, notes in [("the sample just cleaned", cleaned_sample),
                          (config.CLEAN_CLEARANCE_NOTES_FILE,
                           processed_frames[config.CLEAN_CLEARANCE_NOTES_FILE])]:
        texts = notes[config.NOTE_TEXT_COLUMN].astype(str)
        for description, flagged in [
                ("markup", texts.str.contains(config.MARKUP_PATTERN, regex=True)),
                ("escaped characters", texts.str.contains(config.ENCODING_ARTIFACT_PATTERN,
                                                          regex=True)),
                ("runs of whitespace or leading and trailing spaces",
                 (texts != texts.str.strip())
                 | texts.str.contains(r"\s{2,}|[\t\n\r]", regex=True)),
                ("capitals throughout", (texts == texts.str.upper())
                 & (texts != texts.str.lower())),
                ("fewer than " + str(config.MINIMUM_NOTE_WORDS) + " words",
                 texts.str.split().str.len() < config.MINIMUM_NOTE_WORDS),
                ("a length that disagrees with the cleaned text",
                 notes[config.NOTE_LENGTH_COLUMN].astype(int) != texts.str.len())]:
            assert int(flagged.sum()) == 0, (
                str(int(flagged.sum()))
                + " notes in "
                + source
                + " still carry "
                + description
                + ", for example "
                + str(texts[flagged].head(1).tolist())
                + ". The classifier reads words, and a tag, an escape or a fragment is noise to "
                "it."
            )


def test_issue_facts_name_the_issue_the_notes_name_most_once_a_legal_review_counts_twice(
        open_queue, processed_frames, saved_models):
    from config import config
    from tools.campaign_data_tools import fetch_campaign_notes
    from tools.prediction_tools import build_issue_facts

    assert open_queue["problem"] is None, open_queue["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    model = saved_models["issue"]
    notes = open_queue["notes"]
    counts = notes.groupby(config.CAMPAIGN_REFERENCE_COLUMN).size()
    references = list(counts[counts >= 3].index[:40])
    assert len(references) >= 20, (
        "Fewer than twenty live campaigns have three notes or more, so weighted voting cannot "
        "be tried."
    )
    disagreement_seen = False
    for reference in references:
        campaign_notes = fetch_campaign_notes(notes, reference)
        facts = build_issue_facts(model, campaign_notes)
        missing = sorted({"issue_category", "weighted_votes", "weighted_agreement", "notes_read",
                          "notes_too_short_to_read", "note_readings"} - set(facts))
        assert len(missing) == 0, (
            "The issue facts for "
            + reference
            + " carry no "
            + str(missing)
            + ", so the issue analyst cannot read them."
        )
        readable = [note for note in campaign_notes
                    if len(note["note_text"].split()) >= config.MINIMUM_NOTE_WORDS]
        predicted = [str(model.classes_[position]) for position in model.predict_proba(
            [note["note_text"] for note in readable]).argmax(axis=1)]
        assert (facts["notes_read"] == len(readable)
                and facts["notes_too_short_to_read"] == len(campaign_notes) - len(readable)), (
            reference
            + " has "
            + str(len(readable))
            + " notes long enough to read and "
            + str(len(campaign_notes) - len(readable))
            + " too short, but the facts say "
            + str(facts["notes_read"])
            + " and "
            + str(facts["notes_too_short_to_read"])
            + "."
        )
        readings = [(reading["note_sequence"], reading["issue_category"])
                    for reading in facts["note_readings"]]
        assert readings == [(note["note_sequence"], label)
                            for note, label in zip(readable, predicted)], (
            reference
            + " lists the notes as "
            + str(readings)
            + " where the classifier reads them, in the order written, as "
            + str([(note["note_sequence"], label) for note, label in zip(readable, predicted)])
            + ". The facts handed to the issue analyst are the classifier's own reading, note by "
            "note."
        )
        weighted = {}
        for note, label in zip(readable, predicted):
            weighted[label] = weighted.get(label, 0) + config.NOTE_SOURCE_WEIGHTS[
                note["note_source"]]
        most = max(weighted.values())
        assert weighted.get(facts["issue_category"]) == most, (
            reference
            + " was put down as "
            + str(facts["issue_category"])
            + " where its notes, a legal review counting "
            + str(config.NOTE_SOURCE_WEIGHTS["Legal review"])
            + " times, give the weighted votes "
            + str(weighted)
            + ". The issue is the one the notes name most once each note carries its source's "
            "weight."
        )
        assert abs(facts["weighted_agreement"] - most / sum(weighted.values())) < 0.00051, (
            reference
            + " reports a weighted agreement of "
            + str(facts["weighted_agreement"])
            + " where the winning issue holds "
            + str(most)
            + " of "
            + str(sum(weighted.values()))
            + " weighted votes."
        )
        disagreement_seen = disagreement_seen or len(set(predicted)) > 1
    assert disagreement_seen, (
        "Not one of these campaigns has notes that disagree, so the vote was never tested."
    )

    training_notes = processed_frames[config.CLEAN_CLEARANCE_NOTES_FILE][
        config.NOTE_TEXT_COLUMN].astype(str).head(400).tolist()
    labels = [str(model.classes_[position])
              for position in model.predict_proba(training_notes).argmax(axis=1)]
    first_text = training_notes[0]
    second_position = next(position for position, label in enumerate(labels)
                           if label != labels[0])
    second_text = training_notes[second_position]
    for sources, winner in [(["Legal review", "Executive update"], labels[0]),
                            (["Agency reply", "Legal review"], labels[second_position])]:
        chosen = build_issue_facts(model, [
            {"note_sequence": 1, "note_source": sources[0], "note_text": first_text},
            {"note_sequence": 2, "note_source": sources[1], "note_text": second_text}])
        assert chosen["issue_category"] == winner and abs(
                chosen["weighted_agreement"] - 2 / 3) < 0.00051, (
            "A note from the "
            + sources[0]
            + " read as "
            + labels[0]
            + " and one from the "
            + sources[1]
            + " read as "
            + labels[second_position]
            + " were put down as "
            + str(chosen["issue_category"])
            + " with agreement "
            + str(chosen["weighted_agreement"])
            + ", where the legal review's issue wins two votes to one, an agreement of two "
            "thirds. The legal adviser's reading is the one that counts twice."
        )
    summed = dict(zip([str(label) for label in model.classes_],
                      model.predict_proba([first_text, second_text]).sum(axis=0)))
    tied = [labels[0], labels[second_position]]
    stronger = max(tied, key=summed.get)
    for order in [[first_text, second_text], [second_text, first_text]]:
        chosen = build_issue_facts(model, [
            {"note_sequence": 1, "note_source": "Executive first read", "note_text": order[0]},
            {"note_sequence": 2, "note_source": "Agency reply", "note_text": order[1]}])
        assert chosen["issue_category"] == stronger, (
            "Two notes of equal weight read as "
            + str(tied)
            + " were put down as "
            + str(chosen["issue_category"])
            + " where the classifier's summed confidence favours "
            + stronger
            + ". A tie is broken by how sure the classifier is across the notes, never by which "
            "note happens to come first."
        )
    with_short = build_issue_facts(model, [
        {"note_sequence": 1, "note_source": "Executive first read", "note_text": first_text},
        {"note_sequence": 2, "note_source": "Legal review", "note_text": "Agreed, see above."}])
    assert (with_short["notes_too_short_to_read"] == 1 and with_short["notes_read"] == 1
            and with_short["issue_category"] == labels[0]), (
        "A campaign with one readable note and a legal review reading 'Agreed, see above.' was "
        "read as "
        + str({key: with_short[key]
               for key in ["issue_category", "notes_read", "notes_too_short_to_read"]})
        + ". A note too short to learn from is set aside, as it was in training, whatever its "
        "weight."
    )
    for description, campaign_notes in [
            ("every note too short to read",
             [{"note_sequence": 1, "note_source": "Legal review", "note_text": "Agreed."},
              {"note_sequence": 2, "note_source": "Agency reply", "note_text": "Noted, thanks."}]),
            ("a note from a source the configuration gives no weight",
             [{"note_sequence": 1, "note_source": "Telephone message", "note_text": first_text}])]:
        refused = False
        try:
            build_issue_facts(model, campaign_notes)
        except ValueError:
            refused = True
        assert refused, (
            "A campaign with "
            + description
            + " was given an issue without a ValueError. With nothing to read, or a vote whose "
            "weight is unknown, an issue would be invented."
        )


def test_notes_are_read_for_one_campaign_in_the_order_they_were_written(open_queue):
    import pandas as pd

    from config import config
    from tools.campaign_data_tools import fetch_campaign_notes, fetch_campaign_profile

    assert open_queue["problem"] is None, open_queue["problem"]
    notes = open_queue["notes"]
    campaigns = open_queue["campaigns"]
    counts = notes.groupby(config.CAMPAIGN_REFERENCE_COLUMN).size()
    for reference in list(counts.sort_values(ascending=False).index[:15]):
        read = fetch_campaign_notes(notes.sample(frac=1.0, random_state=config.RANDOM_SEED),
                                    reference)
        own = notes[notes[config.CAMPAIGN_REFERENCE_COLUMN] == reference]
        own = own.assign(written_order=pd.to_numeric(own[config.NOTE_SEQUENCE_COLUMN]))
        in_order = own.sort_values("written_order")
        assert ([note["note_text"] for note in read] == list(in_order[config.NOTE_TEXT_COLUMN])
                and [note["note_source"] for note in read]
                == list(in_order[config.NOTE_SOURCE_COLUMN])), (
            "The notes of "
            + reference
            + " came back as "
            + str([note["note_text"][:30] for note in read])
            + " where, in the order they were written, they are "
            + str([text[:30] for text in in_order[config.NOTE_TEXT_COLUMN]])
            + ". The notes were handed over shuffled, and a later note answers an earlier one, "
            "so they are put back in the order they were written, each with its source."
        )
    reference = str(counts.index[0])
    own_rows = notes[notes[config.CAMPAIGN_REFERENCE_COLUMN] == reference].head(3).copy()
    own_rows[config.NOTE_SEQUENCE_COLUMN] = ["10", "2", "1"]
    own_rows[config.NOTE_TEXT_COLUMN] = ["Tenth note of the clearance history, written last.",
                                         "Second note of the clearance history, written next.",
                                         "First note of the clearance history, written first."]
    ordered = fetch_campaign_notes(own_rows, reference)
    assert [note["note_text"].split()[0] for note in ordered] == ["First", "Second", "Tenth"], (
        "Notes numbered 10, 2 and 1 came back in the order "
        + str([note["note_text"].split()[0] for note in ordered])
        + ". The sequence is a number, and 10 comes after 2."
    )
    profile = fetch_campaign_profile(campaigns, reference)
    assert (profile[config.CAMPAIGN_REFERENCE_COLUMN] == reference
            and set(campaigns.columns) <= set(profile)), (
        "The profile fetched for "
        + reference
        + " is not that campaign's row with every column the queue holds."
    )
    unknown = "CLR-ZZZ-9999999"
    repeated = pd.concat([campaigns, campaigns[campaigns[config.CAMPAIGN_REFERENCE_COLUMN]
                                               == reference]])
    shared_sequence = own_rows.assign(**{config.NOTE_SEQUENCE_COLUMN: ["1", "2", "2"]})
    blank_text = own_rows.assign(**{config.NOTE_TEXT_COLUMN: [None, "Second note of the "
                                                              "clearance history, written next.",
                                                              "First note of the clearance "
                                                              "history, written first."]})
    for description, reader, frame, asked in [
            ("a campaign that is not in the queue", fetch_campaign_profile, campaigns, unknown),
            ("a campaign with no notes", fetch_campaign_notes, notes, unknown),
            ("a reference the queue holds twice", fetch_campaign_profile, repeated, reference),
            ("notes sharing a sequence number", fetch_campaign_notes, shared_sequence, reference),
            ("a note whose text is blank", fetch_campaign_notes, blank_text, reference)]:
        refused = False
        try:
            reader(frame, asked)
        except ValueError:
            refused = True
        assert refused, (
            reader.__name__
            + " did not raise a ValueError for "
            + description
            + ". An assessment opened on a campaign that cannot be read for certain has to stop "
            "before it spends a language model call, and the service turns that ValueError into "
            "its refusal."
        )


def test_every_document_gives_up_its_text(document_passages, tmp_path):
    import os
    import shutil

    import pypdf

    from config import config
    from knowledge.pdf_extraction import extract_text_from_directory

    assert document_passages["problem"] is None, document_passages["problem"]
    in_folder = sorted(name for name in os.listdir(config.DOCS_DIRECTORY)
                       if name.lower().endswith(".pdf"))
    documents = document_passages["documents"]
    assert len(documents) > 0 and {"document_name", "document_text"} <= set(documents[0]), (
        "Reading the document folder returned "
        + str(len(documents))
        + " entries"
        + ("" if len(documents) == 0 else ", the first keyed " + str(sorted(documents[0])))
        + ", where every entry names its document and carries its text."
    )
    names = sorted(document["document_name"] for document in documents)
    assert names == in_folder, (
        "Reading the document folder returned "
        + str(names)
        + " where the folder holds "
        + str(in_folder)
        + ". A document never read is a question answered from general knowledge instead."
    )
    for document in documents:
        assert len(document["document_text"].split()) > 300, (
            document["document_name"]
            + " gave up only "
            + str(len(document["document_text"].split()))
            + " words. A PDF that will not give up its text is indexed as an empty passage."
        )
    total_words = sum(len(document["document_text"].split()) for document in documents)
    assert total_words >= 10000, (
        "The "
        + str(len(in_folder))
        + " documents gave up "
        + str(total_words)
        + " words between them, where every page read in full gives well over 10,000. A reader "
        "that stops early loses the later sections, and a rule in them is never read."
    )
    for required in (list(config.CATEGORY_CODE_DOCUMENTS.values())
                     + list(config.GUIDE_DOCUMENT_BY_ISSUE.values())):
        assert required in names, (
            required
            + " was not read, and the configuration sends questions to it by name."
        )
    folder = tmp_path / "mixed_folder"
    folder.mkdir()
    code_document = config.CATEGORY_CODE_DOCUMENTS["Alcohol"]
    guide_document = config.GUIDE_DOCUMENT_BY_ISSUE["Health claim"]
    for name in [code_document, guide_document]:
        shutil.copy(os.path.join(config.DOCS_DIRECTORY, name), str(folder / name))
    (folder / "reading-list.txt").write_text("Not a document of the clearance desk.")
    (folder / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    mixed = extract_text_from_directory(str(folder))
    assert sorted(entry["document_name"] for entry in mixed) == sorted([code_document,
                                                                        guide_document]), (
        "A folder holding two PDFs, a text file and an image was read as "
        + str(sorted(entry["document_name"] for entry in mixed))
        + ". Only the PDFs are the desk's documents; anything else in the folder is passed over."
    )
    blank_folder = tmp_path / "blank_page_folder"
    blank_folder.mkdir()
    writer = pypdf.PdfWriter()
    writer.add_page(pypdf.PdfReader(os.path.join(config.DOCS_DIRECTORY, code_document)).pages[0])
    writer.add_blank_page()
    with open(str(blank_folder / "Code-With-A-Blank-Page.pdf"), "wb") as blank_file:
        writer.write(blank_file)
    refused = False
    try:
        extract_text_from_directory(str(blank_folder))
    except ValueError:
        refused = True
    assert refused, (
        "A PDF whose second page gives up no text was read without a ValueError. A page that "
        "cannot be read may be the one that states a rule, so the document is refused rather "
        "than indexed with a hole in it."
    )


def test_documents_are_cut_into_passages_of_about_the_configured_size_without_page_headers(
        document_passages):
    import re

    from config import config
    from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections

    assert document_passages["problem"] is None, document_passages["problem"]
    documents = document_passages["documents"]
    sections = document_passages["sections"]
    chunks = document_passages["chunks"]
    assert len(chunks) > 0 and {"chunk_id", "chunk_text", "document_name"} <= set(chunks[0]), (
        "Cutting the documents returned "
        + str(len(chunks))
        + " passages"
        + ("" if len(chunks) == 0 else ", the first keyed " + str(sorted(chunks[0])))
        + ", where every passage carries its identifier, its text and its document."
    )
    assert len(sections) > 5 * len(documents), (
        str(len(documents))
        + " documents split into "
        + str(len(sections))
        + " sections. Each document is written in numbered sections, and a section is the unit "
        "a question is usually about."
    )
    with_header = [chunk["chunk_id"] for chunk in chunks
                   if re.search(config.DOCUMENT_HEADER_PATTERN, chunk["chunk_text"])]
    assert len(with_header) == 0, (
        "These passages still carry the document header: "
        + str(with_header[:5])
        + ". The header repeats the same words in every document, so it matches every question "
        "equally and pushes the passages that answer it out of the results."
    )
    lengths = [len(chunk["chunk_text"]) for chunk in chunks]
    average = sum(lengths) / len(lengths)
    assert config.CHUNK_SIZE_CHARACTERS / 5 <= average <= config.CHUNK_SIZE_CHARACTERS, (
        "Passages average "
        + str(int(average))
        + " characters where the configuration asks for passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + ". Passages far longer return a whole document for every question; far shorter lose "
        "the sentence that gave them their meaning."
    )
    assert max(lengths) <= 2 * config.CHUNK_SIZE_CHARACTERS, (
        "The longest passage runs to "
        + str(max(lengths))
        + " characters, far beyond the "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + " the configuration asks for."
    )
    identifiers = [chunk["chunk_id"] for chunk in chunks]
    assert len(identifiers) == len(set(identifiers)), (
        "Passage identifiers repeat, so storing them would overwrite one passage with another."
    )
    words_by_document = {document["document_name"]: set(document["document_text"].split())
                         for document in documents}
    for chunk in chunks:
        words = chunk["chunk_text"].split()
        foreign = [word for word in words
                   if word not in words_by_document.get(chunk["document_name"], set())]
        assert len(words) > 0, (
            "Passage "
            + str(chunk["chunk_id"])
            + " from "
            + str(chunk["document_name"])
            + " holds no words, and a passage with no text can answer nothing."
        )
        assert len(foreign) <= len(words) // 10, (
            "Passage "
            + str(chunk["chunk_id"])
            + " says it comes from "
            + str(chunk["document_name"])
            + " but "
            + str(len(foreign))
            + " of its "
            + str(len(words))
            + " words are not in that document, for example "
            + str(foreign[:5])
            + ". A passage has to cite the document it was cut from."
        )
    rule_patterns = ([config.RESTRICTED_HOURS_PATTERN, config.ANY_TIME_PATTERN,
                      config.LIVE_SPORT_EXCEPTION_PATTERN]
                     + list(config.CHILDREN_RULE_PATTERNS.values())
                     + list(config.CODE_FIGURE_PATTERNS.values()))
    for document in documents:
        if document["document_name"] not in config.CATEGORY_CODE_DOCUMENTS.values():
            continue
        flowing_text = re.sub(r"\s+", " ", document["document_text"])
        own_chunks = [re.sub(r"\s+", " ", chunk["chunk_text"]) for chunk in chunks
                      if chunk["document_name"] == document["document_name"]]
        stated = [found.group(0) for pattern in rule_patterns
                  for found in re.finditer(pattern, flowing_text)]
        broken = [sentence for sentence in stated
                  if not any(sentence in chunk_text for chunk_text in own_chunks)]
        assert len(stated) >= 5, (
            "Only "
            + str(len(stated))
            + " rule sentences that the patterns in config read were found in the text "
            "extracted from "
            + document["document_name"]
            + ", where the code states each of its five rules in a sentence of its own, so the "
            "extraction has broken them."
        )
        assert len(broken) == 0, (
            "These rule sentences of "
            + document["document_name"]
            + " are not whole inside any one of its passages: "
            + str(broken[:3])
            + ". A restricted hour, a threshold, a cap or a separation cut across two passages "
            "can no longer be read from either."
        )

    header = "Document Code 99 | Issue 1 | Owner: Clearance | Review cycle: every six months"
    lines = ["Reading " + first + second + " of the overlap check names break " + first + second
             + "." for first in "abcdefgh" for second in "abcdefgh"][:60]
    synthetic = [{"document_name": "Overlap-Check.pdf",
                  "document_text": "HARBOURLIGHT MEDIA\nOverlap Check\n" + header
                                   + "\n1. Overlap check\n" + " ".join(lines)}]
    pieces = [str(chunk["chunk_text"]) for chunk in build_chunk_records(
        split_documents_into_sections(synthetic))]
    assert len(pieces) >= 3, (
        "One numbered section of "
        + str(len(" ".join(lines)))
        + " characters was cut into "
        + str(len(pieces))
        + " passages, where passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + " characters need at least three."
    )
    unshared = [position for position in range(1, len(pieces))
                if pieces[position].strip()[:20] not in pieces[position - 1]]
    assert len(unshared) == 0, (
        "Passages "
        + str(unshared)
        + " of one long section open with text the passage before them does not carry. "
        "Neighbouring passages of a section overlap by up to "
        + str(config.CHUNK_OVERLAP_CHARACTERS)
        + " characters, so a sentence cut at a boundary is still whole in one of them."
    )
    for description, document_text in [
            ("no header line", "HARBOURLIGHT MEDIA\n1. Overlap check\n" + " ".join(lines)),
            ("two header lines", header + "\n" + header + "\n1. Overlap check\n"
             + " ".join(lines)),
            ("sections numbered from 2", header + "\n2. Overlap check\n" + " ".join(lines))]:
        refused = False
        try:
            split_documents_into_sections([{"document_name": "Broken-Layout.pdf",
                                            "document_text": document_text}])
        except ValueError:
            refused = True
        assert refused, (
            "A document with "
            + description
            + " was split without a ValueError. Every document carries one header line and "
            "sections numbered from 1, so a document laid out otherwise has been read wrongly "
            "and is refused."
        )


def test_the_index_holds_every_passage_once_with_the_document_it_came_from(
        knowledge_collection, document_passages):
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    assert document_passages["problem"] is None, document_passages["problem"]
    collection = knowledge_collection["collection"]
    documents = document_passages["documents"]
    chunks = document_passages["chunks"]
    stored = collection.get(include=["metadatas", "documents"])
    assert collection.count() == len(chunks) == len(stored["ids"]), (
        "The vector store holds "
        + str(collection.count())
        + " passages where cutting the documents gives "
        + str(len(chunks))
        + ". More means the index was added to twice and repeats itself in every result; fewer "
        "means passages no search can reach."
    )
    assert len(set(stored["ids"])) == len(stored["ids"]), (
        "Stored passage identifiers repeat, so one passage has overwritten another and its text "
        "can no longer be found."
    )
    names = {document["document_name"] for document in documents}
    cited = [metadata.get("document_name") for metadata in stored["metadatas"]]
    unnamed = [value for value in cited if value not in names]
    assert len(unnamed) == 0, (
        str(len(unnamed))
        + " stored passages name no document from the folder, for example "
        + str(unnamed[:3])
        + ". The name is what lets an answer cite its source and a code be read from its own "
        "document alone."
    )
    by_document = {}
    for chunk in chunks:
        by_document[chunk["document_name"]] = by_document.get(chunk["document_name"], 0) + 1
    stored_by_document = {name: cited.count(name) for name in names}
    assert stored_by_document == by_document, (
        "Passages stored per document "
        + str(stored_by_document)
        + " differ from passages cut per document "
        + str(by_document)
        + ". A document with passages missing answers less than it says."
    )
    assert sorted(stored["documents"]) == sorted(chunk["chunk_text"] for chunk in chunks), (
        "The texts in the vector store are not the passages the documents are cut into, so a "
        "rule read from the store is not the rule the code states."
    )


def test_a_search_narrowed_to_one_document_reads_only_that_document(knowledge_collection):
    from config import config
    from knowledge.vector_index import find_nearest_passage, read_document_chunks, search_documents

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    collection = knowledge_collection["collection"]
    question = "How many spots may go out, and how far apart?"
    everything = collection.get(include=["documents", "metadatas"])
    ranked = collection.query(query_texts=[question], n_results=collection.count(),
                              include=["documents", "metadatas"])
    for document_name in list(config.CATEGORY_CODE_DOCUMENTS.values())[:3] + list(
            config.GUIDE_DOCUMENT_BY_ISSUE.values())[:2]:
        passage = find_nearest_passage(collection, question, document_name)
        whole = read_document_chunks(collection, document_name)
        stored = [text for text, metadata in zip(everything["documents"], everything["metadatas"])
                  if metadata["document_name"] == document_name]
        assert sorted(whole) == sorted(stored), (
            "Reading every passage of "
            + document_name
            + " returned "
            + str(len(whole))
            + " passages where the store holds "
            + str(len(stored))
            + ". A code is read whole, never from the few passages a search ranks first."
        )
        nearest = next(text for text, metadata in zip(ranked["documents"][0],
                                                      ranked["metadatas"][0])
                       if metadata["document_name"] == document_name)
        assert passage in stored and passage == nearest, (
            "Asked "
            + repr(question)
            + " of "
            + document_name
            + " alone, the search returned "
            + repr(str(passage)[:80])
            + ", which is "
            + ("not a passage of that document" if passage not in stored
               else "not the passage of that document closest to the question")
            + ". The codes share their headings and their sentences, and a passage from another "
            "code states another category's figures."
        )
    for reader, arguments in [(read_document_chunks, (collection, "No-Such-Code.pdf")),
                              (find_nearest_passage, (collection, question, "No-Such-Code.pdf"))]:
        refused = False
        try:
            reader(*arguments)
        except ValueError:
            refused = True
        assert refused, (
            reader.__name__
            + " did not raise a ValueError for a document the index does not hold. An empty "
            "answer would be taken for a document that states nothing."
        )
    questions = ["Which past cases did the desk clear with restrictions, and why?",
                 "What does the executive check for a price claim?"]
    results = search_documents(collection, questions)
    assert len(results) == len(questions), (
        "Searching with "
        + str(len(questions))
        + " questions returned "
        + str(len(results))
        + " result lists, one per question."
    )
    for asked, passages in zip(questions, results):
        assert len(passages) == config.CHAT_RETRIEVAL_RESULT_COUNT and all(
                {"chunk_id", "chunk_text", "document_name", "distance"} <= set(found)
                for found in passages), (
            "Asking "
            + repr(asked)
            + " returned "
            + str(len(passages))
            + " passages, the first keyed "
            + str(sorted(passages[0]) if passages else [])
            + ", where "
            + str(config.CHAT_RETRIEVAL_RESULT_COUNT)
            + " are returned, each with its identifier, its text, its document and its distance."
        )
        distances = [found["distance"] for found in passages]
        assert distances == sorted(distances), (
            "Passages came back in the order "
            + str(distances)
            + ", not closest first. The first passage is the one an answer is grounded on."
        )
        excluded = sorted({found["document_name"] for found in passages}
                          & set(config.CHAT_EXCLUDED_DOCUMENTS))
        assert len(excluded) == 0, (
            "Asking "
            + repr(asked)
            + " returned passages of "
            + str(excluded)
            + ". The casebook records what happened to earlier campaigns and must never reach an "
            "answer, even to a question about earlier campaigns."
        )


def test_the_issue_guides_checks_and_the_codes_rebooking_steps_come_from_the_right_documents(
        knowledge_collection):
    from config import config
    from tools.retrieval_tools import (read_executive_checks, read_rebooking_steps,
                                       read_standing_restrictions)

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    collection = knowledge_collection["collection"]
    everything = collection.get(include=["documents", "metadatas"])
    for reader, question, documents in [
            (read_executive_checks, config.EXECUTIVE_CHECKS_QUESTION,
             config.GUIDE_DOCUMENT_BY_ISSUE),
            (read_rebooking_steps, config.REBOOKING_QUESTION, config.CATEGORY_CODE_DOCUMENTS),
            (read_standing_restrictions, config.STANDING_RESTRICTIONS_QUESTION,
             config.CATEGORY_CODE_DOCUMENTS)]:
        passages = {}
        ranked = collection.query(query_texts=[question], n_results=collection.count(),
                                  include=["documents", "metadatas"])
        for subject, document_name in documents.items():
            passage = reader(collection, subject)
            own = [text for text, metadata in zip(everything["documents"], everything["metadatas"])
                   if metadata["document_name"] == document_name]
            nearest = next(text for text, metadata in zip(ranked["documents"][0],
                                                          ranked["metadatas"][0])
                           if metadata["document_name"] == document_name)
            assert passage in own, (
                reader.__name__
                + " for "
                + subject
                + " returned "
                + repr(str(passage)[:80])
                + ", which is not a passage of "
                + document_name
                + ". Every issue has its own guide and every category its own code, and a "
                "passage from another tells the agency the wrong thing."
            )
            assert passage == nearest, (
                reader.__name__
                + " for "
                + subject
                + " returned a passage of "
                + document_name
                + " that is not the one closest to the configured question "
                + repr(question)
                + ". The passage is the document's answer to that question, not any passage of "
                "it."
            )
            passages[subject] = passage
        assert len(set(passages.values())) == len(documents), (
            reader.__name__
            + " gave two subjects the same passage word for word, so at least one was answered "
            "with text its document shares with others rather than with its own section."
        )
    for reader in [read_executive_checks, read_rebooking_steps, read_standing_restrictions]:
        refused = False
        try:
            reader(collection, "Cosmetics")
        except ValueError:
            refused = True
        assert refused, (
            reader.__name__
            + " did not raise a ValueError for 'Cosmetics', which the configuration gives no "
            "document. A passage from some other document would be passed off as its rules."
        )


def test_each_category_is_read_from_its_own_code(planted_codes, knowledge_collection):
    from config import config
    from knowledge.code_reading import read_category_code

    assert planted_codes["problem"] is None, planted_codes["problem"]
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    assert planted_codes["valid_count"] == planted_codes["valid_records"], (
        "build_vector_index stored "
        + str(planted_codes["valid_count"])
        + " passages of the "
        + str(planted_codes["valid_records"])
        + " it was given, so it does not report what the store holds."
    )
    for store_name, collection in [("the planted store", planted_codes["valid"]),
                                   ("the knowledge index", knowledge_collection["collection"])]:
        fingerprints = {}
        for category, document_name in config.CATEGORY_CODE_DOCUMENTS.items():
            reading = read_category_code(collection, category)
            missing = sorted({"document_name", "restricted_hours", "live_sport_exception",
                              "children_rule", "index_threshold", "daily_cap",
                              "separation_minutes"} - set(reading or {}))
            assert len(missing) == 0, (
                "Reading the code for "
                + category
                + " from "
                + store_name
                + " returned no "
                + str(missing)
                + ", so the spot check has nothing to test that rule against."
            )
            assert reading["document_name"] == document_name, (
                "The code for "
                + category
                + " was read from "
                + repr(reading["document_name"])
                + " where the configuration names "
                + document_name
                + ". Every rule belongs to the category whose own code states it."
            )
            figures = [reading["index_threshold"], reading["daily_cap"],
                       reading["separation_minutes"]]
            figures += list((reading["restricted_hours"] or {}).values())
            figures += list((reading["live_sport_exception"] or {}).values())
            fingerprints[category] = str(figures + [reading["children_rule"]])
            if store_name == "the planted store":
                borrowed = sorted(str(figure) for figure in figures
                                  if figure in planted_codes["decoy_figures"])
                assert len(borrowed) == 0, (
                    "Reading the planted code for "
                    + category
                    + " picked up "
                    + str(borrowed)
                    + ", which only a superseded notice in the same store states. A code is "
                    "read from its own document, never from whichever passage of the store "
                    "mentions the same rule."
                )
        assert len(set(fingerprints.values())) == len(fingerprints), (
            "Two categories were read with identical rules from "
            + store_name
            + ": "
            + str(fingerprints)
            + ". Each category's code states its own figures, so one code was read for another."
        )
    refused = False
    try:
        read_category_code(planted_codes["valid"], "Cosmetics")
    except ValueError:
        refused = True
    assert refused, (
        "Reading the code for 'Cosmetics', a category the configuration gives no code, did not "
        "raise a ValueError. Some other category's rules would be applied to it."
    )


def test_restricted_hours_are_read_or_else_the_any_time_sentence(planted_codes):
    from knowledge.code_reading import read_category_code

    assert planted_codes["problem"] is None, planted_codes["problem"]
    for category, expected in planted_codes["expected"].items():
        reading = read_category_code(planted_codes["valid"], category)
        assert reading["restricted_hours"] == expected["restricted_hours"], (
            "From a store in which the code for "
            + category
            + " states "
            + ("restricted hours from " + expected["restricted_hours"]["start"] + " to "
               + expected["restricted_hours"]["end"]
               if expected["restricted_hours"] else "that its spots may go out at any time")
            + ", the restricted hours were read as "
            + repr(reading["restricted_hours"])
            + ". The hours are read from the code's own sentence as the times it writes; a code "
            "that allows any time has none, and hours that cross midnight are still one span."
        )
    refused = False
    try:
        read_category_code(planted_codes["contradictory"], "Alcohol")
    except ValueError:
        refused = True
    assert refused, (
        "A code for Alcohol that states neither restricted hours nor that its spots may go out "
        "at any time was read without a ValueError. Neither reading is safe to guess: one "
        "withdraws spots the code allows, the other clears spots it forbids."
    )


def test_the_live_sport_exception_is_read_only_where_a_code_states_it(planted_codes):
    from knowledge.code_reading import read_category_code

    assert planted_codes["problem"] is None, planted_codes["problem"]
    for category, expected in planted_codes["expected"].items():
        reading = read_category_code(planted_codes["valid"], category)
        assert reading["live_sport_exception"] == expected["live_sport_exception"], (
            "From a store in which the code for "
            + category
            + (" excepts centre breaks in live sport more than "
               + str(expected["live_sport_exception"]["minutes_after_start"])
               + " minutes after the coverage starts and more than "
               + str(expected["live_sport_exception"]["minutes_before_end"])
               + " minutes before it ends"
               if expected["live_sport_exception"] else " states no live sport exception")
            + ", the exception was read as "
            + repr(reading["live_sport_exception"])
            + ". Each margin is read from the code's own sentence, and a code with restricted "
            "hours but no exception sentence has no exception, whatever another category's code "
            "allows."
        )
    refused = False
    try:
        read_category_code(planted_codes["contradictory"], "Consumer credit")
    except ValueError:
        refused = True
    assert refused, (
        "A code for Consumer credit that allows spots at any time and also states a live sport "
        "exception was read without a ValueError. An exception to restricted hours the code "
        "does not have means the code has been misread."
    )


def test_the_index_threshold_and_the_reach_of_the_childrens_rule_are_read(planted_codes):
    from config import config
    from knowledge.code_reading import read_category_code

    assert planted_codes["problem"] is None, planted_codes["problem"]
    for category, expected in planted_codes["expected"].items():
        reading = read_category_code(planted_codes["valid"], category)
        assert reading["index_threshold"] == expected["index_threshold"], (
            "From a store in which the code for "
            + category
            + " sets an under-18 index threshold of "
            + str(expected["index_threshold"])
            + ", the threshold was read as "
            + repr(reading["index_threshold"])
            + ". It is a whole number read from the code's own sentence."
        )
        assert reading["children_rule"] == expected["children_rule"], (
            "From a store in which the code for "
            + category
            + " forbids spots "
            + expected["children_rule"]
            + " a programme made for children, the rule was read as "
            + repr(reading["children_rule"])
            + ". The configuration names the two readings, "
            + repr(config.CHILDREN_IN_OR_ADJACENT)
            + " and "
            + repr(config.CHILDREN_IN_ONLY)
            + ", and an end break beside a children's programme is withdrawn under one and "
            "cleared under the other."
        )
    for category, contradiction in [("Betting and gaming", "two different index thresholds"),
                                    ("Less healthy food", "both readings of the children's rule")]:
        refused = False
        try:
            read_category_code(planted_codes["contradictory"], category)
        except ValueError:
            refused = True
        assert refused, (
            "A code for "
            + category
            + " that states "
            + contradiction
            + " was read without a ValueError. A code that contradicts itself is refused, never "
            "settled by picking one of the figures."
        )


def test_the_daily_cap_and_the_separation_are_read(planted_codes):
    from knowledge.code_reading import read_category_code

    assert planted_codes["problem"] is None, planted_codes["problem"]
    for category, expected in planted_codes["expected"].items():
        reading = read_category_code(planted_codes["valid"], category)
        read_as = (reading["daily_cap"], reading["separation_minutes"])
        assert read_as == (expected["daily_cap"], expected["separation_minutes"]), (
            "From a store in which the code for "
            + category
            + " allows "
            + str(expected["daily_cap"])
            + " spots a channel a broadcast day, at least "
            + str(expected["separation_minutes"])
            + " minutes apart, the cap and the separation were read as "
            + str(read_as)
            + ". Both are whole numbers read from the code's own sentences, wherever in the code "
            "they are written, and a sentence that appears in two passages is the same rule "
            "read twice."
        )
    refused = False
    try:
        read_category_code(planted_codes["contradictory"], "Medicines and health")
    except ValueError:
        refused = True
    assert refused, (
        "A code for Medicines and health that states two different separations was read "
        "without a ValueError. A code that contradicts itself is refused, never settled by "
        "picking one of the figures."
    )


def test_every_spot_is_placed_in_or_adjacent_on_the_broadcast_clock(planted_schedule, planted_code,
                                                                    open_queue):
    import pandas as pd

    from config import config
    from tools.campaign_data_tools import fetch_campaign_schedules, fetch_campaign_spots
    from tools.schedule_tools import check_campaign_spots, place_spot

    assert open_queue["problem"] is None, open_queue["problem"]
    columns = [config.SPOT_REFERENCE_COLUMN, config.CAMPAIGN_REFERENCE_COLUMN,
               config.CLOCK_NUMBER_COLUMN, config.CHANNEL_COLUMN, config.SCHEDULING_SYSTEM_COLUMN,
               config.TRANSMISSION_DAY_COLUMN, config.BREAK_TIME_COLUMN, config.SPOT_LENGTH_COLUMN]
    bookings = [
        ("SPT-0000001-01", "Harbourlight Drama", "Airtime Planner", "Fri", "09:00", "0:30",
         "Friday", "09:00", config.BREAK_ADJACENT, ["Morning Repeats", "Daytime Drama"]),
        ("SPT-0000001-02", "Harbourlight Drama", "TXLine", "FRIDAY", "24:15", '30"',
         "Friday", "24:15", config.BREAK_IN, ["Late Film"]),
        ("SPT-0000001-03", "Harbourlight Drama", "SlotMaster", "Sa.", "01:00", "30s",
         "Friday", "25:00", config.BREAK_IN, ["Late Film"]),
        ("SPT-0000001-04", "Harbourlight Drama", "Planit", "sat", "05:59", "00:00:30",
         "Friday", "29:59", config.BREAK_IN, ["Night Shop"]),
        ("SPT-0000001-05", "Harbourlight Drama", "Airtime Planner", "Fri", "07:30", "0:30",
         "Friday", "07:30", config.BREAK_IN, ["Morning Repeats"]),
        ("SPT-0000001-06", "HLD", "TXLine", "Friday", "25:30", '30"',
         "Friday", "25:30", config.BREAK_ADJACENT, ["Late Film", "Night Shop"]),
        ("SPT-0000001-07", "Harbourlight Drama", "Airtime Planner", "Saturday", "06:30", "0:30",
         "Saturday", "06:30", config.BREAK_IN, ["Weekend Breakfast"]),
        ("SPT-0000001-08", "h-drama", "SlotMaster", "SAT", "12:00", "30s",
         "Saturday", "12:00", config.BREAK_ADJACENT, ["Weekend Breakfast", "Saturday Marathon"]),
        ("SPT-0000001-09", "Harbourlight Drama", "TXLine", "Friday", "19:00", '30"',
         "Friday", "19:00", config.BREAK_ADJACENT, ["Teatime Cartoons", "Evening Drama"]),
        ("SPT-0000001-10", "Harbourlight Drama", "TXLine", "Friday", "18:00", '30"',
         "Friday", "18:00", config.BREAK_IN, ["Teatime Cartoons"]),
    ]
    booked = pd.DataFrame([(booking[0], "CLR-TST-0000001", "TST/PLNT001/030") + booking[1:6]
                           for booking in bookings], columns=columns)
    spots = fetch_campaign_spots(booked, "CLR-TST-0000001")
    schedules = fetch_campaign_schedules(planted_schedule, spots)
    placements = {placement["spot_reference"]: placement
                  for placement in [place_spot(spot, schedules) for spot in spots]}
    for booking in bookings:
        placement = placements.get(booking[0], {})
        placed_as = (placement.get("broadcast_day"), placement.get("break_time"),
                     placement.get("break_position"),
                     [programme.get("programme_title")
                      for programme in placement.get("programmes", [])])
        assert placed_as == tuple(booking[6:9]) + (booking[9],), (
            booking[0]
            + ", booked by "
            + booking[2]
            + " on "
            + booking[1]
            + " for "
            + booking[3]
            + " "
            + booking[4]
            + ", was placed as "
            + str(placed_as)
            + " where it is "
            + str(tuple(booking[6:9]) + (booking[9],))
            + ". A break is placed on the broadcast clock of its broadcast day, in the programme "
            "it falls inside, or adjacent to the programme that ends and the one that starts at "
            "that minute, in that order."
        )
    for children_rule, withdrawn in [(config.CHILDREN_IN_OR_ADJACENT,
                                      ["SPT-0000001-09", "SPT-0000001-10"]),
                                     (config.CHILDREN_IN_ONLY, ["SPT-0000001-10"])]:
        results = check_campaign_spots(list(placements.values()),
                                       dict(planted_code, children_rule=children_rule))
        caught = sorted(result["spot_reference"] for result in results
                        if config.BREACH_CHILDREN in result["breach_codes"])
        assert caught == withdrawn, (
            "Under a code that forbids spots "
            + children_rule
            + " a programme made for children, the spots caught by the rule were "
            + str(caught)
            + " where they are "
            + str(withdrawn)
            + ". The spot at 18:00 is in Teatime Cartoons and the spot at 19:00 is in the end "
            "break after it, adjacent to it and to the programme that follows."
        )
    refusals = [
        ("SPT-0000002-01", "Harbourlight Drama", "Airtime Planner", "Fri", "13:15", "0:30",
         "a break in the half hour the schedule leaves empty after Lunchtime Mystery"),
        ("SPT-0000002-02", "Harbourlight Drama", "Airtime Planner", "Fri", "13:00", "0:30",
         "a break at the end of Lunchtime Mystery, when no programme starts"),
        ("SPT-0000002-03", "Harbourlight Drama", "TXLine", "Friday", "06:00", '30"',
         "a break at the very start of the broadcast day"),
        ("SPT-0000002-04", "Harbourlight One", "TXLine", "Friday", "10:00", '30"',
         "a channel and day the schedule does not carry"),
        ("SPT-0000002-05", "Harbourlight Drama", "TXLine", "Friday", "10:00", "45s",
         "a spot 45 seconds long, which is not a standard length"),
    ]
    for refusal in refusals:
        frame = pd.DataFrame([(refusal[0], "CLR-TST-0000002",
                               "TST/PLNT002/045" if refusal[5] == "45s" else "TST/PLNT002/030")
                              + refusal[1:6]], columns=columns)
        refused = False
        try:
            refused_spots = fetch_campaign_spots(frame, "CLR-TST-0000002")
            refused_schedules = fetch_campaign_schedules(planted_schedule, refused_spots)
            for spot in refused_spots:
                place_spot(spot, refused_schedules)
        except ValueError:
            refused = True
        assert refused, (
            "A spot booked for "
            + refusal[3]
            + " "
            + refusal[4]
            + " on "
            + refusal[1]
            + " was placed without a ValueError. That is "
            + refusal[6]
            + ", and a spot that cannot be placed for certain is refused, never put somewhere "
            "near."
        )

    published = {
        "SPT-0042117-01": ("Saturday", "18:20", config.BREAK_IN, ["Saturday Scoreline"]),
        "SPT-0042117-02": ("Saturday", "19:05", config.BREAK_IN, ["Premier Football Live"]),
        "SPT-0042117-03": ("Saturday", "19:40", config.BREAK_IN, ["Premier Football Live"]),
        "SPT-0042117-04": ("Saturday", "20:20", config.BREAK_IN, ["Premier Football Live"]),
        "SPT-0042117-05": ("Saturday", "20:35", config.BREAK_IN, ["Premier Football Live"]),
        "SPT-0042117-06": ("Saturday", "21:15", config.BREAK_ADJACENT,
                           ["Premier Football Live", "Football Tonight"]),
        "SPT-0042117-07": ("Saturday", "22:30", config.BREAK_IN, ["Fight Night Preview"]),
        "SPT-0042117-08": ("Saturday", "25:10", config.BREAK_IN, ["Harbour Poker Club"]),
        "SPT-0042117-09": ("Saturday", "22:45", config.BREAK_IN, ["Harbour Crime"]),
        "SPT-0042117-10": ("Saturday", "29:40", config.BREAK_IN, ["Night Film"]),
    }
    example_spots = fetch_campaign_spots(open_queue["spots"], "CLR-QSC-0042117")
    example_schedules = fetch_campaign_schedules(open_queue["schedule"], example_spots)
    example = {placement["spot_reference"]: (placement["broadcast_day"], placement["break_time"],
                                             placement["break_position"],
                                             [programme["programme_title"]
                                              for programme in placement["programmes"]])
               for placement in [place_spot(spot, example_schedules) for spot in example_spots]}
    assert example == published, (
        "The worked example's ten spots were placed as "
        + str({reference: example.get(reference) for reference in published
               if example.get(reference) != published[reference]})
        + " where the question places them as "
        + str({reference: published[reference] for reference in published
               if example.get(reference) != published[reference]})
        + ". The tenth spot, booked for Sunday 05:40 on the calendar clock, is Saturday 29:40 on "
        "the broadcast clock, and the sixth sits in the end break between the match and "
        "Football Tonight."
    )


def test_restricted_hours_that_wrap_the_broadcast_day_run_from_their_first_minute_to_their_last(
        planted_schedule, planted_code):
    import pandas as pd

    from config import config
    from tools.campaign_data_tools import fetch_campaign_schedules, fetch_campaign_spots
    from tools.schedule_tools import check_campaign_spots, place_spot

    columns = [config.SPOT_REFERENCE_COLUMN, config.CAMPAIGN_REFERENCE_COLUMN,
               config.CLOCK_NUMBER_COLUMN, config.CHANNEL_COLUMN, config.SCHEDULING_SYSTEM_COLUMN,
               config.TRANSMISSION_DAY_COLUMN, config.BREAK_TIME_COLUMN, config.SPOT_LENGTH_COLUMN]
    bookings = [("SPT-0000003-01", "TXLine", "Friday", "20:59"),
                ("SPT-0000003-02", "TXLine", "Friday", "21:00"),
                ("SPT-0000003-03", "Airtime Planner", "Sat", "05:29"),
                ("SPT-0000003-04", "Airtime Planner", "Sat", "05:30"),
                ("SPT-0000003-05", "TXLine", "Friday", "06:10"),
                ("SPT-0000003-06", "TXLine", "Friday", "23:45"),
                ("SPT-0000003-07", "TXLine", "Friday", "21:59"),
                ("SPT-0000003-08", "TXLine", "Friday", "22:00"),
                ("SPT-0000003-09", "SlotMaster", "Sat", "01:59"),
                ("SPT-0000003-10", "SlotMaster", "Sat", "02:00")]
    booked = pd.DataFrame([(reference, "CLR-TST-0000003", "TST/PLNT003/030", "Harbourlight Drama",
                            system, day, time, "30s")
                           for reference, system, day, time in bookings], columns=columns)
    spots = fetch_campaign_spots(booked, "CLR-TST-0000003")
    schedules = fetch_campaign_schedules(planted_schedule, spots)
    placements = [place_spot(spot, schedules) for spot in spots]
    written = {booking[0]: booking[2] + " " + booking[3] for booking in bookings}
    for restricted_hours, withdrawn in [
            ({"start": "05:30", "end": "21:00"},
             ["SPT-0000003-01", "SPT-0000003-04", "SPT-0000003-05"]),
            ({"start": "22:00", "end": "02:00"},
             ["SPT-0000003-06", "SPT-0000003-08", "SPT-0000003-09"]),
            (None, [])]:
        results = check_campaign_spots(placements,
                                       dict(planted_code, restricted_hours=restricted_hours))
        caught = sorted(result["spot_reference"] for result in results
                        if config.BREACH_WINDOW in result["breach_codes"])
        assert caught == withdrawn, (
            "Under a code whose restricted hours are "
            + (restricted_hours["start"] + " to " + restricted_hours["end"]
               if restricted_hours else "none, its spots allowed at any time")
            + ", the spots caught by the restricted hours were "
            + str([written[reference] for reference in caught])
            + " where they are "
            + str([written[reference] for reference in withdrawn])
            + ". The hours are on the ordinary clock and can wrap round midnight and the end of "
            "the broadcast day; a break at their first minute is inside them and a break at "
            "their last minute is not, so 21:00 is clear and Saturday 05:30 on the calendar "
            "clock, the end of Friday's broadcast day, is not."
        )


def test_the_live_sport_exception_needs_both_of_its_margins_and_a_centre_break(planted_schedule,
                                                                               planted_code):
    import pandas as pd

    from config import config
    from tools.campaign_data_tools import fetch_campaign_schedules, fetch_campaign_spots
    from tools.schedule_tools import check_campaign_spots, place_spot

    columns = [config.SPOT_REFERENCE_COLUMN, config.CAMPAIGN_REFERENCE_COLUMN,
               config.CLOCK_NUMBER_COLUMN, config.CHANNEL_COLUMN, config.SCHEDULING_SYSTEM_COLUMN,
               config.TRANSMISSION_DAY_COLUMN, config.BREAK_TIME_COLUMN, config.SPOT_LENGTH_COLUMN]
    times = {"SPT-0000005-01": "17:10", "SPT-0000005-02": "17:11", "SPT-0000005-03": "18:50",
             "SPT-0000005-04": "18:49", "SPT-0000005-05": "19:00", "SPT-0000005-06": "19:05",
             "SPT-0000005-07": "20:30", "SPT-0000005-08": "20:40", "SPT-0000005-09": "20:50",
             "SPT-0000005-10": "21:10"}
    booked = pd.DataFrame([(reference, "CLR-TST-0000005", "TST/PLNT005/030", "Harbourlight Sport",
                            "TXLine", "Friday", time, '30"') for reference, time in times.items()],
                          columns=columns)
    spots = fetch_campaign_spots(booked, "CLR-TST-0000005")
    schedules = fetch_campaign_schedules(planted_schedule, spots)
    placements = [place_spot(spot, schedules) for spot in spots]
    restricted = {"start": "05:30", "end": "21:00"}
    for exception, withdrawn in [
            ({"minutes_after_start": 10, "minutes_before_end": 10},
             ["17:10", "18:50", "19:00", "19:05", "20:40", "20:50"]),
            (None, ["17:10", "17:11", "18:49", "18:50", "19:00", "19:05", "20:30", "20:40",
                    "20:50"]),
            ({"minutes_after_start": 20, "minutes_before_end": 5},
             ["17:10", "17:11", "19:00", "19:05", "20:40", "20:50"])]:
        results = check_campaign_spots(placements, dict(planted_code, restricted_hours=restricted,
                                                        live_sport_exception=exception))
        caught = sorted(times[result["spot_reference"]] for result in results
                        if config.BREACH_WINDOW in result["breach_codes"])
        assert caught == withdrawn, (
            "Under restricted hours of 05:30 to 21:00 with "
            + ("an exception for centre breaks in live sport more than "
               + str(exception["minutes_after_start"]) + " minutes after the coverage starts and "
               "more than " + str(exception["minutes_before_end"]) + " minutes before it ends"
               if exception else "no live sport exception")
            + ", the spots withdrawn for the restricted hours broke at "
            + str(caught)
            + " where they break at "
            + str(withdrawn)
            + ". Cup Football Live runs 17:00 to 19:00 and Second Half Live 19:00 to 20:45, each "
            "its own coverage; more than means a break exactly on a margin is not excepted, an "
            "end break is never within coverage, and the margins are the code's own."
        )


def test_the_cap_is_counted_per_channel_and_broadcast_day_in_transmission_order(planted_schedule,
                                                                                planted_code):
    import pandas as pd

    from config import config
    from tools.campaign_data_tools import fetch_campaign_schedules, fetch_campaign_spots
    from tools.schedule_tools import check_campaign_spots, place_spot

    columns = [config.SPOT_REFERENCE_COLUMN, config.CAMPAIGN_REFERENCE_COLUMN,
               config.CLOCK_NUMBER_COLUMN, config.CHANNEL_COLUMN, config.SCHEDULING_SYSTEM_COLUMN,
               config.TRANSMISSION_DAY_COLUMN, config.BREAK_TIME_COLUMN, config.SPOT_LENGTH_COLUMN]
    bookings = [
        ("SPT-0000004-01", "Harbourlight Drama", "TXLine", "Friday", "09:30", []),
        ("SPT-0000004-02", "Harbourlight Drama", "Airtime Planner", "Fri", "07:00", []),
        ("SPT-0000004-03", "Harbourlight Drama", "SlotMaster", "Sa.", "01:30",
         [config.BREACH_FREQUENCY]),
        ("SPT-0000004-04", "Harbourlight Drama", "Planit", "Friday", "20:00",
         [config.BREACH_INDEX]),
        ("SPT-0000004-05", "Harbourlight Drama", "TXLine", "Friday", "20:30",
         [config.BREACH_INDEX, config.BREACH_FREQUENCY]),
        ("SPT-0000004-06", "Harbourlight Drama", "Airtime Planner", "Saturday", "07:00", []),
        ("SPT-0000004-08", "Harbourlight Life", "TXLine", "Friday", "11:00", []),
        ("SPT-0000004-09", "Harbourlight Life", "TXLine", "Friday", "10:00", []),
        ("SPT-0000004-12", "Harbourlight Life", "TXLine", "Friday", "12:00",
         [config.BREACH_FREQUENCY]),
        ("SPT-0000004-10", "Harbourlight Life", "TXLine", "Friday", "12:00", []),
    ]
    booked = pd.DataFrame([(booking[0], "CLR-TST-0000004", "TST/PLNT004/030") + booking[1:5]
                           + ("0:30",) for booking in bookings], columns=columns)
    spots = fetch_campaign_spots(booked, "CLR-TST-0000004")
    schedules = fetch_campaign_schedules(planted_schedule, spots)
    results = check_campaign_spots([place_spot(spot, schedules) for spot in spots],
                                   dict(planted_code, index_threshold=100, daily_cap=3,
                                        children_rule=config.CHILDREN_IN_ONLY))
    found = {result["spot_reference"]: list(result["breach_codes"]) for result in results}
    assert len(results) == len(bookings), (
        "Checking "
        + str(len(bookings))
        + " booked spots returned "
        + str(len(results))
        + " results. Every booked spot is checked and reported, compliant or not."
    )
    for booking in bookings:
        assert found.get(booking[0]) == booking[5], (
            booking[0]
            + ", booked on "
            + booking[1]
            + " for "
            + booking[3]
            + " "
            + booking[4]
            + ", was given "
            + str(found.get(booking[0]))
            + " where it breaks "
            + str(booking[5])
            + " under a cap of 3 spots and an under-18 index threshold of 100. The cap counts a "
            "campaign's spots per channel and per broadcast day in the order they are "
            "transmitted, every booked spot counting whether or not it breaks another rule; "
            "Saturday 01:30 on the calendar clock belongs to Friday's broadcast day; two spots "
            "in one break go out in spot-reference order; a programme indexed exactly at the "
            "threshold breaks it; and the codes are listed in the order "
            + str(config.BREACH_CODES)
            + "."
        )


def test_the_separation_runs_from_the_previous_spot_and_settles_the_decisions_open(
        planted_schedule, planted_code, open_queue, knowledge_collection):
    import pandas as pd

    from config import config
    from knowledge.code_reading import read_category_code
    from tools.campaign_data_tools import fetch_campaign_schedules, fetch_campaign_spots
    from tools.schedule_tools import (check_campaign_spots, compute_clearance_outcome, place_spot,
                                      summarise_spot_checks)

    assert open_queue["problem"] is None, open_queue["problem"]
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    columns = [config.SPOT_REFERENCE_COLUMN, config.CAMPAIGN_REFERENCE_COLUMN,
               config.CLOCK_NUMBER_COLUMN, config.CHANNEL_COLUMN, config.SCHEDULING_SYSTEM_COLUMN,
               config.TRANSMISSION_DAY_COLUMN, config.BREAK_TIME_COLUMN, config.SPOT_LENGTH_COLUMN]
    bookings = [
        ("SPT-0000006-01", "Harbourlight Drama", "TXLine", "Friday", "10:00", False),
        ("SPT-0000006-02", "Harbourlight Drama", "TXLine", "Friday", "10:20", True),
        ("SPT-0000006-03", "Harbourlight Drama", "TXLine", "Friday", "10:45", True),
        ("SPT-0000006-04", "Harbourlight Drama", "TXLine", "Friday", "11:15", False),
        ("SPT-0000006-06", "Harbourlight Drama", "TXLine", "Friday", "11:50", True),
        ("SPT-0000006-05", "Harbourlight Drama", "TXLine", "Friday", "11:50", False),
        ("SPT-0000006-07", "Harbourlight Drama", "Airtime Planner", "Sat", "05:50", False),
        ("SPT-0000006-08", "Harbourlight Drama", "Airtime Planner", "Saturday", "06:10", False),
        ("SPT-0000006-09", "Harbourlight Life", "TXLine", "Friday", "10:10", False),
    ]
    booked = pd.DataFrame([(booking[0], "CLR-TST-0000006", "TST/PLNT006/030") + booking[1:5]
                           + ('30"',) for booking in bookings], columns=columns)
    spots = fetch_campaign_spots(booked, "CLR-TST-0000006")
    schedules = fetch_campaign_schedules(planted_schedule, spots)
    results = check_campaign_spots([place_spot(spot, schedules) for spot in spots],
                                   dict(planted_code, separation_minutes=30))
    found = {result["spot_reference"]: list(result["breach_codes"]) for result in results}
    for booking in bookings:
        wanted = [config.BREACH_SEPARATION] if booking[5] else []
        assert found.get(booking[0]) == wanted, (
            booking[0]
            + ", booked on "
            + booking[1]
            + " for "
            + booking[3]
            + " "
            + booking[4]
            + ", was given "
            + str(found.get(booking[0]))
            + " where it breaks "
            + str(wanted)
            + " under a separation of 30 minutes. A spot less than 30 minutes after the "
            "campaign's previous spot on the same channel in the same broadcast day breaks the "
            "rule, measured from that previous spot whatever its own result; 30 minutes exactly "
            "is enough; two spots in one break are no minutes apart and go out in "
            "spot-reference order; and another channel or the next broadcast day starts afresh."
        )
    summary = summarise_spot_checks(results)
    wanted_summary = {"spots_booked": 9, "spots_compliant": 6, "spots_breaching": 3,
                      "breaches_by_rule": {config.BREACH_WINDOW: 0, config.BREACH_INDEX: 0,
                                           config.BREACH_CHILDREN: 0, config.BREACH_FREQUENCY: 0,
                                           config.BREACH_SEPARATION: 3},
                      "decisions_open": [config.DECISION_CLEAR_WITH_RESTRICTIONS,
                                         config.DECISION_REJECT]}
    found_summary = {key: summary.get(key) for key in wanted_summary}
    first = summary.get("first_transmission") or {}
    assert found_summary == wanted_summary and (first.get("broadcast_day"),
                                                first.get("break_time")) == ("Friday", "10:00"), (
        "The planted campaign was summarised as "
        + str(found_summary)
        + " with its first transmission at "
        + str(first)
        + ", where nine spots were booked, six comply and three break the separation, the first "
        "goes out on Friday at 10:00, and with some spots breaking and some not only "
        + str(wanted_summary["decisions_open"])
        + " are open."
    )
    compliant = [result for result in results if not result["breach_codes"]]
    breaching = [result for result in results if result["breach_codes"]]
    for subset, open_decisions in [(compliant, [config.DECISION_CLEAR, config.DECISION_REJECT]),
                                   (breaching, [config.DECISION_REJECT])]:
        assert summarise_spot_checks(subset).get("decisions_open") == open_decisions, (
            "A campaign whose "
            + str(len(subset))
            + " spots all "
            + ("comply" if subset is compliant else "break a rule")
            + " was left with the decisions "
            + str(summarise_spot_checks(subset).get("decisions_open"))
            + " open, where only "
            + str(open_decisions)
            + " are. CLEAR needs every spot compliant, CLEAR WITH RESTRICTIONS needs some of each, "
            "and REJECT is always open."
        )
    outcome = compute_clearance_outcome(config.DECISION_CLEAR_WITH_RESTRICTIONS, results)
    assert (sorted(outcome["spots_cleared"]) == sorted(result["spot_reference"]
                                                       for result in compliant)
            and sorted(outcome["spots_withdrawn"]) == sorted(result["spot_reference"]
                                                             for result in breaching)
            and (outcome["spots_cleared_count"], outcome["spots_withdrawn_count"]) == (6, 3)), (
        "Clearing the planted campaign with restrictions cleared "
        + str(outcome.get("spots_cleared"))
        + " and withdrew "
        + str(outcome.get("spots_withdrawn"))
        + ", counted as "
        + str((outcome.get("spots_cleared_count"), outcome.get("spots_withdrawn_count")))
        + ". The six compliant spots are cleared and the three breaching spots withdrawn, and "
        "the counts say how many of each."
    )

    example_spots = fetch_campaign_spots(open_queue["spots"], "CLR-QSC-0042117")
    example_schedules = fetch_campaign_schedules(open_queue["schedule"], example_spots)
    example_results = check_campaign_spots(
        [place_spot(spot, example_schedules) for spot in example_spots],
        read_category_code(knowledge_collection["collection"], "Betting and gaming"))
    published = {"SPT-0042117-01": [config.BREACH_WINDOW], "SPT-0042117-02": [config.BREACH_WINDOW],
                 "SPT-0042117-03": [], "SPT-0042117-04": [],
                 "SPT-0042117-05": [config.BREACH_SEPARATION],
                 "SPT-0042117-06": [config.BREACH_INDEX], "SPT-0042117-07": [],
                 "SPT-0042117-08": [config.BREACH_FREQUENCY], "SPT-0042117-09": [],
                 "SPT-0042117-10": [config.BREACH_WINDOW]}
    example = {result["spot_reference"]: list(result["breach_codes"]) for result in example_results}
    assert example == published, (
        "The worked example's spots were found to break "
        + str({reference: example.get(reference) for reference in published
               if example.get(reference) != published[reference]})
        + " where the question finds "
        + str({reference: published[reference] for reference in published
               if example.get(reference) != published[reference]})
        + ". Four spots may air and six must be withdrawn, each for the rule the worked example "
        "names."
    )
    example_summary = summarise_spot_checks(example_results)
    example_first = example_summary.get("first_transmission") or {}
    assert (example_summary.get("decisions_open") == [config.DECISION_CLEAR_WITH_RESTRICTIONS,
                                                      config.DECISION_REJECT]
            and (example_first.get("broadcast_day"), example_first.get("break_time"))
            == ("Saturday", "18:20")), (
        "The worked example was summarised with the decisions "
        + str(example_summary.get("decisions_open"))
        + " open and its first transmission at "
        + str(example_first)
        + ", where six spots break the code and four do not, so only CLEAR WITH RESTRICTIONS "
        "and REJECT are open, and its first spot goes out on Saturday at 18:20."
    )
    example_outcome = compute_clearance_outcome(config.DECISION_CLEAR_WITH_RESTRICTIONS,
                                                example_results)
    rejected = compute_clearance_outcome(config.DECISION_REJECT, example_results)
    assert (sorted(example_outcome["spots_cleared"])
            == ["SPT-0042117-03", "SPT-0042117-04", "SPT-0042117-07", "SPT-0042117-09"]
            and rejected["spots_cleared"] == [] and rejected["spots_withdrawn_count"] == 10), (
        "Clearing the worked example with restrictions cleared "
        + str(example_outcome.get("spots_cleared"))
        + ", and rejecting it cleared "
        + str(rejected.get("spots_cleared"))
        + ", where the first clears spots 3, 4, 7 and 9 and the second clears none of the ten."
    )
    refused = False
    try:
        compute_clearance_outcome(config.DECISION_CLEAR, example_results)
    except ValueError:
        refused = True
    assert refused, (
        "Clearing every spot of the worked example was accepted without a ValueError. Six of "
        "its spots break the code, so CLEAR is not open, and a decision the spots do not allow "
        "is refused rather than recorded."
    )


def test_each_assessing_agent_was_called_once_and_in_order(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    listed = saved_run["calls_at_gate"]
    assert listed["status"] == 200 and isinstance(listed["body"], dict) and {
            "calls", "call_count", "expected_call_count"} <= set(listed["body"]), (
        "Reading the run's model calls at the pause returned "
        + str(listed["status"])
        + " with "
        + str(listed["body"])[:300]
        + ", where the audit area lists every call with its count and the count expected."
    )
    at_gate = listed["body"]
    assessing = [config.AGENT_NAME_PLANNER, config.AGENT_NAME_COPY_RISK, config.AGENT_NAME_ISSUE,
                 config.AGENT_NAME_SCHEDULE, config.AGENT_NAME_WRITER]
    called = [call["agent_name"] for call in at_gate["calls"]]
    assert at_gate["call_count"] == len(at_gate["calls"]) == config.LLM_CALLS_BEFORE_DECISION, (
        "Reaching the Clearance Controller recorded a call count of "
        + str(at_gate["call_count"])
        + " and listed "
        + str(len(at_gate["calls"]))
        + " language model calls where the workflow allows "
        + str(config.LLM_CALLS_BEFORE_DECISION)
        + ", one per assessing agent. More means an agent ran twice; fewer, that one never ran."
    )
    assert called == assessing, (
        "The agents ran in the order "
        + str(called)
        + ". Each one reads what the one before it concluded: the copy risk is read against the "
        "plan, the issue against the copy risk, the spots against the plan, and the write-up "
        "draws on all of them."
    )
    assert at_gate["expected_call_count"] == config.LLM_CALLS_BEFORE_DECISION, (
        "Before any decision the service expected "
        + str(at_gate["expected_call_count"])
        + " calls. Until a decision is taken only the assessing agents' calls are expected."
    )


def test_every_model_call_is_recorded_with_what_it_cost(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    listed = saved_run["final_calls"]
    assert listed["status"] == 200 and isinstance(listed["body"], dict) and {
            "calls", "total_latency_milliseconds", "run_id"} <= set(listed["body"]), (
        "Reading the finished run's model calls returned "
        + str(listed["status"])
        + " with "
        + str(listed["body"])[:300]
        + ", where the audit area lists every call, the run it belongs to and their total "
        "latency."
    )
    final = listed["body"]
    agents = {config.AGENT_NAME_PLANNER, config.AGENT_NAME_COPY_RISK, config.AGENT_NAME_ISSUE,
              config.AGENT_NAME_SCHEDULE, config.AGENT_NAME_WRITER,
              config.AGENT_NAME_RESTRICTION_NOTICE, config.AGENT_NAME_CLEARANCE_CERTIFICATE}
    assert len(final["calls"]) > 0 and final["run_id"] == saved_run["at_gate"]["run_id"], (
        "The calls listed belong to run "
        + str(final["run_id"])
        + " where the assessment was made by run "
        + str(saved_run["at_gate"]["run_id"])
        + ", or none was recorded at all, so nothing can say what the assessment cost."
    )
    for call in final["calls"]:
        assert call["agent_name"] in agents, (
            "A call was recorded against "
            + str(call["agent_name"])
            + ", which is not one of the seven agents."
        )
        assert str(call["model_name"]).strip(), (
            "The call from "
            + call["agent_name"]
            + " does not say which model answered it, so an audit asking what assessed the "
            "campaign has nothing to read."
        )
        assert call["prompt_characters"] > 0 and call["response_characters"] > 0, (
            "The call from "
            + call["agent_name"]
            + " records a prompt of "
            + str(call["prompt_characters"])
            + " and a reply of "
            + str(call["response_characters"])
            + " characters. A call that was made sent something and got something back."
        )
        assert call["latency_milliseconds"] > 0, (
            "The call from "
            + call["agent_name"]
            + " took "
            + str(call["latency_milliseconds"])
            + " milliseconds by its own record."
        )
    assert final["total_latency_milliseconds"] == sum(
            call["latency_milliseconds"] for call in final["calls"]), (
        "The run's total latency "
        + str(final["total_latency_milliseconds"])
        + " is not the sum of its calls."
    )
    assert saved_run["final"]["llm_call_count"] == len(final["calls"]), (
        "The run records "
        + str(saved_run["final"]["llm_call_count"])
        + " calls against the "
        + str(len(final["calls"]))
        + " logged for it. The run's own count is what the audit trail reads."
    )
    assert saved_run["at_gate"]["llm_call_count"] == config.LLM_CALLS_BEFORE_DECISION, (
        "At the pause the run recorded "
        + str(saved_run["at_gate"]["llm_call_count"])
        + " calls, where the five assessing agents had made "
        + str(config.LLM_CALLS_BEFORE_DECISION)
        + ". The count is kept up to date while the run waits, not only when it finishes."
    )


def test_the_assessment_reaches_the_controller_as_a_recommendation_among_the_decisions_open(
        saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    assert (at_gate["recommended_decision"] in config.DECISIONS
            and at_gate["recommended_decision"] in at_gate["decisions_open"]), (
        "The write-up recommended "
        + repr(at_gate["recommended_decision"])
        + " where the decisions open for this campaign are "
        + str(at_gate["decisions_open"])
        + ". A recommendation the spots do not allow, or one in other words, cannot be acted on."
    )
    assert (at_gate["executive_request"] == saved_run["executive_request"]
            and at_gate["executive_id"] == saved_run["opener"]["username"]), (
        "The record holds the request the assessment was opened with as "
        + repr(at_gate["executive_request"])
        + ", made by "
        + repr(at_gate["executive_id"])
        + ", where "
        + saved_run["opener"]["username"]
        + " asked "
        + repr(saved_run["executive_request"])
        + ". The plan is only readable against what was actually asked, and by whom."
    )
    plan = str(at_gate["assessment_plan_text"]).strip()
    assessment = str(at_gate["assessment_text"]).strip()
    assert len(plan) > 60 and len(assessment) > 60, (
        "The assessment plan stored runs to "
        + str(len(plan))
        + " characters and the assessment to "
        + str(len(assessment))
        + ", so an agent's reply never reached the record."
    )
    assert plan != assessment and assessment not in plan, (
        "The assessment stored is the plan over again, so the writer's reply was thrown away."
    )
    assert (at_gate["issue_category"] in config.GUIDE_DOCUMENT_BY_ISSUE
            and at_gate["amendment_band"] in [config.AMENDMENT_BAND_HIGH,
                                              config.AMENDMENT_BAND_MEDIUM,
                                              config.AMENDMENT_BAND_LOW]), (
        "The record names the issue as "
        + repr(at_gate["issue_category"])
        + " and the band as "
        + repr(at_gate["amendment_band"])
        + ", which the desk does not use."
    )


def test_a_reply_decorated_the_way_models_write_is_read_back_whole():
    from agentic.agents import parse_labelled_response
    from config import config

    labels = ["MAY AIR", "MAY NOT AIR", "SCHEDULE SUMMARY"]
    first_cleared = "1. SPT-0000001-03 on Harbourlight Sport, Saturday 19:40"
    second_cleared = "2. SPT-0000001-04 on Harbourlight Sport, Saturday 20:20"
    withdrawn = "1. SPT-0000001-01 on Harbourlight Sport, Saturday 18:20 breaks WINDOW"
    summary_start = "Two spots may air and one may not."
    summary_end = "The withdrawn spot can move to a break after 21:00."
    reply = ("Sure - here is my advice on the booked spots.\n"
             "\n"
             "**MAY AIR:**\n"
             + first_cleared + "\n"
             + second_cleared + "\n"
             "## MAY NOT AIR:\n"
             + withdrawn + "\n"
             "\n"
             "> SCHEDULE SUMMARY: " + summary_start + "\n"
             + summary_end)
    refusal = ""
    try:
        parsed = parse_labelled_response(reply, labels, "decoration_check")
    except ValueError as error:
        parsed, refusal = {}, str(error)
    assert refusal == "", (
        "A reply holding all three fields was refused: "
        + refusal
        + ". A model wraps its labels in markdown, headings and quotes, and a reader that cannot "
        "see past them stops every assessment at the first agent."
    )
    assert sorted(parsed) == sorted(labels), (
        "Reading the reply returned the fields "
        + str(sorted(parsed))
        + " where "
        + str(sorted(labels))
        + " were asked for."
    )
    for label in labels:
        value = str(parsed[label])
        assert value.strip() and value == value.strip(config.RESPONSE_DECORATION_CHARACTERS), (
            label
            + " was read back as "
            + repr(value)
            + ". The decoration a model puts around a label belongs to neither the label nor its "
            "value, and an empty value is no value."
        )
    may_air = parsed["MAY AIR"]
    assert (first_cleared in may_air and second_cleared in may_air and withdrawn not in may_air
            and summary_start not in may_air), (
        "MAY AIR was read back as "
        + repr(may_air)
        + ". A value that runs over several lines keeps every line until the next label, and "
        "nothing from any other field, even from MAY NOT AIR, whose label begins the same way."
    )
    assert withdrawn in parsed["MAY NOT AIR"] and first_cleared not in parsed["MAY NOT AIR"], (
        "MAY NOT AIR was read back as "
        + repr(parsed["MAY NOT AIR"])
        + " where the reply gives it as "
        + repr(withdrawn)
        + "."
    )
    assert summary_start in parsed["SCHEDULE SUMMARY"] and summary_end in parsed[
            "SCHEDULE SUMMARY"], (
        "SCHEDULE SUMMARY was read back as "
        + repr(parsed["SCHEDULE SUMMARY"])
        + ". A label behind a quote mark is still a label, and its value runs on to the next "
        "line."
    )
    try:
        written_twice = parse_labelled_response(
            "MAY AIR: " + first_cleared + "\nMAY NOT AIR: " + withdrawn
            + "\nMAY NOT AIR: SPT-0000001-05 on Harbourlight Sport, Saturday 20:35 breaks "
              "SEPARATION\nSCHEDULE SUMMARY: " + summary_start, labels,
            "decoration_check")["MAY NOT AIR"]
    except ValueError as error:
        written_twice = "refused: " + str(error)
    assert withdrawn in written_twice and "SPT-0000001-05" in written_twice, (
        "A reply that wrote the MAY NOT AIR label again on its second line was read back as "
        + repr(written_twice)
        + ". A label written again adds its line to the field and never throws away what came "
        "before it, or a notice written that way loses every withdrawn spot but the last."
    )
    for incomplete in ["MAY AIR: One spot\nMAY NOT AIR: Two spots",
                       "MAY AIR: One spot\nMAY NOT AIR:\nSCHEDULE SUMMARY: Three spots",
                       "MAY NOT AIR: Two spots\nSCHEDULE SUMMARY: Three spots"]:
        refused = False
        try:
            parse_labelled_response(incomplete, labels, "decoration_check")
        except ValueError:
            refused = True
        assert refused, (
            "A reply missing a field, or leaving one empty, was accepted: "
            + repr(incomplete)
            + ". The next agent reads every field, so a reply without one has to be refused with "
            "a ValueError rather than passed on."
        )


def test_the_restriction_notice_is_drafted_before_the_certificate_and_both_reach_the_record(
        saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    listed = {name: saved_run[name] for name in ["calls_at_gate", "final_calls"]}
    assert all(entry["status"] == 200 and isinstance(entry["body"], dict)
               and {"calls", "call_count", "expected_call_count"} <= set(entry["body"])
               for entry in listed.values()), (
        "Reading the run's model calls returned "
        + str({name: (entry["status"], str(entry["body"])[:150])
               for name, entry in listed.items()})
        + ", where the audit area lists every call with its count and the count expected."
    )
    before = listed["calls_at_gate"]["body"]["calls"]
    final = listed["final_calls"]["body"]
    drafting = [call["agent_name"] for call in final["calls"][len(before):]]
    assert drafting == [config.AGENT_NAME_RESTRICTION_NOTICE,
                        config.AGENT_NAME_CLEARANCE_CERTIFICATE], (
        "After a decision of "
        + saved_run["decision"]
        + " the agents called were "
        + str(drafting)
        + ". Clearing with restrictions means the restriction notice is written first and the "
        "clearance certificate second, so the certificate can say the agency has been sent the "
        "notice."
    )
    assert (final["call_count"] == len(final["calls"]) == final["expected_call_count"]
            == config.LLM_CALLS_BY_DECISION[saved_run["decision"]]), (
        "A decision of "
        + saved_run["decision"]
        + " ended with "
        + str(final["call_count"])
        + " calls recorded, "
        + str(len(final["calls"]))
        + " listed and "
        + str(final["expected_call_count"])
        + " expected, where the workflow allows "
        + str(config.LLM_CALLS_BY_DECISION[saved_run["decision"]])
        + ". The expected figure follows the decision taken."
    )
    notice = str(saved_run["final"]["restriction_notice_text"] or "").strip()
    certificate = str(saved_run["final"]["clearance_certificate_text"] or "").strip()
    assert len(notice) > 60 and len(certificate) > 60, (
        "Clearing with restrictions stored a restriction notice of "
        + str(len(notice))
        + " characters and a clearance certificate of "
        + str(len(certificate))
        + ", so a drafting agent's reply never reached the record."
    )
    assert notice != certificate and certificate not in notice and notice not in certificate, (
        "The restriction notice and the clearance certificate stored are one text, so one "
        "document was written into the other's place."
    )
    assessment = str(saved_run["final"]["assessment_text"]).strip()
    assert assessment not in (notice, certificate) and assessment == str(
            saved_run["at_gate"]["assessment_text"]).strip(), (
        "After the decision the stored assessment is no longer the one the Controller decided "
        "on, or it stands in for one of the documents sent to the agency."
    )


def test_the_workflow_has_ten_reachable_nodes_and_sends_each_decision_down_its_own_path(tmp_path):
    from agentic.graph import build_workflow
    from agentic.state import ClearanceAssessmentState
    from config import config

    compiled = build_workflow(str(tmp_path / "workflow_check.db"))
    node_names = sorted(str(name) for name in compiled.builder.nodes)
    expected_nodes = sorted(["load_campaign_context", "plan_assessment", "assess_copy_risk",
                             "identify_issues", "check_spot_schedule",
                             "compose_clearance_assessment", "request_controller_decision",
                             "draft_restriction_notice", "draft_clearance_certificate",
                             "record_clearance_outcome"])
    assert node_names == expected_nodes, (
        "The workflow declares the nodes "
        + str(node_names)
        + " where the design has "
        + str(expected_nodes)
        + "."
    )
    branches = compiled.builder.branches.get("request_controller_decision", {})
    assert len(branches) == 1, (
        "The decision node has "
        + str(len(branches))
        + " conditional branches where it has one, choosing the path from the decision."
    )
    branch = list(branches.values())[0]
    destinations = {}
    for decision in config.DECISIONS:
        state = {key: None for key in ClearanceAssessmentState.__annotations__}
        state["controller_decision"] = {"decision": decision, "decided_by": "Routing Check",
                                        "decided_by_role": config.ROLE_CONTROLLER,
                                        "decision_reason": "Routing is being checked."}
        routed = branch.path.invoke(state)
        if isinstance(routed, (list, tuple)) and len(routed) == 1:
            routed = routed[0]
        destinations[decision] = branch.ends[routed] if branch.ends else routed
    expected_paths = {config.DECISION_CLEAR: "draft_clearance_certificate",
                      config.DECISION_CLEAR_WITH_RESTRICTIONS: "draft_restriction_notice",
                      config.DECISION_REJECT: "record_clearance_outcome"}
    assert destinations == expected_paths, (
        "The decisions route to "
        + str(destinations)
        + ". Clearing writes the certificate alone, clearing with restrictions writes the "
        "restriction notice and then the certificate, and rejecting writes neither."
    )
    edges = {(str(source), str(target)) for source, target in compiled.builder.edges}
    edges |= {("request_controller_decision", str(target)) for target in destinations.values()}
    reached = {"__start__"}
    grew = True
    while grew:
        grew = False
        for source, target in edges:
            if source in reached and target not in reached:
                reached.add(target)
                grew = True
    unreachable = sorted(set(node_names) - reached)
    assert len(unreachable) == 0, (
        "These nodes cannot be reached from the start: "
        + str(unreachable)
        + ". A node with no way in never runs, whatever it contains."
    )
    for source, target, reason in [
            ("compose_clearance_assessment", "request_controller_decision",
             "the assessment is stored before the run waits for the Controller"),
            ("draft_restriction_notice", "draft_clearance_certificate",
             "the certificate says the agency has been sent the notice, so it is written after "
             "it"),
            ("draft_clearance_certificate", "record_clearance_outcome",
             "what was written has to reach the record"),
            ("record_clearance_outcome", "__end__", "the run ends once the outcome is recorded")]:
        assert (source, target) in edges, (
            "Nothing leads from " + source + " to " + target + ": " + reason + "."
        )


def test_the_run_waited_for_another_controller_before_any_spot_was_cleared(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    assert at_gate["status"] == config.CAMPAIGN_STATUS_PENDING, (
        "When the five assessing agents finished, the campaign stood at "
        + str(at_gate["status"])
        + " where it waits at "
        + config.CAMPAIGN_STATUS_PENDING
        + ". The Advertising Clearance Policy gives the decision to a Clearance Controller, so "
        "no spot is cleared and nothing is sent to the agency until one decides."
    )
    pending = saved_run["pending_at_gate"]
    assert pending["status"] == 200 and isinstance(pending["body"], list), (
        "Listing the campaigns pending a decision returned "
        + str(pending["status"])
        + " with "
        + str(pending["body"])[:300]
        + ", so the Clearance Controllers' queue cannot be read."
    )
    assert at_gate["assessment_id"] in [str(row.get("assessment_id")) for row in pending["body"]], (
        "The assessment waiting for a decision is missing from the list of campaigns pending a "
        "decision, so the decision queue would not show it."
    )
    nothing_yet = {field_name: at_gate[field_name] for field_name in [
        "decision", "decided_by", "spots_cleared", "spots_withdrawn", "spots_cleared_count",
        "spots_withdrawn_count", "restriction_notice_text", "clearance_certificate_text"]
        if at_gate[field_name] is not None}
    assert len(nothing_yet) == 0, (
        "Before anyone decided, the record already holds "
        + str(nothing_yet)
        + ". Nothing may be decided, cleared, withdrawn or written to the agency while the run "
        "waits."
    )
    assert saved_run["run_record"]["problem"] is None, saved_run["run_record"]["problem"]
    status_at_gate = saved_run["run_record"]["status_at_gate"]
    assert status_at_gate == config.AGENT_RUN_STATUS_AWAITING_DECISION, (
        "While it waited the run was recorded as "
        + str(status_at_gate)
        + " where it should read "
        + config.AGENT_RUN_STATUS_AWAITING_DECISION
        + ". A paused run that still reads as running tells nobody it is waiting for them."
    )
    refused = saved_run["own_decision"]
    assert (refused["status"] == 403 and refused["still_pending"] is True
            and refused["calls_after"] == refused["calls_before"]), (
        "The Clearance Controller who opened "
        + at_gate["assessment_id"]
        + " was answered "
        + str(refused["status"])
        + " with "
        + str(refused["body"])[:200]
        + " when they tried to decide it; afterwards the campaign "
        + ("was still" if refused["still_pending"] else "was no longer")
        + " waiting, with "
        + str(refused["calls_after"])
        + " model calls recorded where "
        + str(refused["calls_before"])
        + " were recorded before. The Advertising Clearance Policy separates reviewing a "
        "campaign from deciding it, so the person who opened an assessment is refused with 403 "
        "before anything resumes, and another Clearance Controller decides it."
    )


def test_resuming_finished_the_run_from_its_checkpoint_without_repeating_an_agent(saved_run):
    from agentic.graph import build_workflow
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_run["run_record"]["problem"] is None, saved_run["run_record"]["problem"]
    listed = {name: saved_run[name] for name in ["calls_at_gate", "final_calls"]}
    assert all(entry["status"] == 200 and isinstance(entry["body"], dict)
               and "calls" in entry["body"] for entry in listed.values()), (
        "Reading the run's model calls returned "
        + str({name: (entry["status"], str(entry["body"])[:150])
               for name, entry in listed.items()})
        + ", so what ran before and after the pause cannot be compared."
    )
    before = listed["calls_at_gate"]["body"]["calls"]
    after = listed["final_calls"]["body"]["calls"]
    agents_after = [call["agent_name"] for call in after]
    assert len(agents_after) == len(set(agents_after)), (
        "The finished run called "
        + str(agents_after)
        + ". An agent that appears twice ran again on resume, so the paused state was not "
        "restored and the assessment was paid for twice."
    )
    assert after[:len(before)] == before and len(after) > len(before), (
        "Resuming turned the calls made before the pause, "
        + str([call["agent_name"] for call in before])
        + ", into "
        + str(agents_after)
        + ". Resuming continues from the pause: the calls already made stay exactly as they were "
        "recorded, and only the drafting agents the decision needs are added after them."
    )
    assert saved_run["run_record"]["status_final"] == config.AGENT_RUN_STATUS_COMPLETE, (
        "The run finished at "
        + str(saved_run["run_record"]["status_final"])
        + " rather than "
        + config.AGENT_RUN_STATUS_COMPLETE
        + ". A finished run that still reads as waiting stays in somebody's queue."
    )
    reopened = build_workflow(config.CHECKPOINT_DATABASE_FILE).get_state(
        {"configurable": {"thread_id": saved_run["run_record"]["thread_id"]}})
    assert reopened.values.get("assessment_id") == saved_run["at_gate"]["assessment_id"], (
        "A workflow built afresh on "
        + config.CHECKPOINT_DATABASE_FILE
        + " does not find the run's state under its thread, so a paused run would not survive "
        "the service being restarted overnight."
    )
    assert len(reopened.next) == 0 and (
            (reopened.values.get("controller_decision") or {}).get("decision")
            == saved_run["decision"]), (
        "The checkpoint of the finished run still waits at "
        + str(reopened.next)
        + " or holds a different decision, so the run on disk is not the one that finished."
    )


def test_the_decision_is_recorded_once_with_the_spots_it_clears_and_a_closed_one_refused(
        saved_run, service_client, staff):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert staff["problem"] is None, staff["problem"]
    at_gate = saved_run["at_gate"]
    closed = saved_run["closed_decision"]
    expected_refusal = config.DECISION_NOT_OPEN_MESSAGE.format(
        decision=config.DECISION_CLEAR, spots_breaching=at_gate["spots_breaching"],
        spots_booked=at_gate["spots_booked"], spots_compliant=at_gate["spots_compliant"])
    assert (closed["status"] == 409 and isinstance(closed["body"], dict)
            and closed["body"].get("detail") == expected_refusal
            and closed["calls_after"] == closed["calls_before"]), (
        "Deciding CLEAR on "
        + at_gate["assessment_id"]
        + ", where "
        + str(at_gate["spots_breaching"])
        + " of its spots break the code, was answered "
        + str(closed["status"])
        + " with "
        + str(closed["body"])[:300]
        + ", and the calls went from "
        + str(closed["calls_before"])
        + " to "
        + str(closed["calls_after"])
        + ". A decision the spots do not allow is refused with 409 and the configuration's "
        "message, "
        + repr(expected_refusal)
        + ", before the run resumes."
    )
    final = saved_run["final"]
    assert final["status"] == config.CAMPAIGN_STATUS_BY_DECISION[saved_run["decision"]], (
        "After a decision of "
        + saved_run["decision"]
        + " the campaign stands at "
        + str(final["status"])
        + " where the configuration maps it to "
        + config.CAMPAIGN_STATUS_BY_DECISION[saved_run["decision"]]
        + "."
    )
    assert (final["decision"] == saved_run["decision"]
            and final["decided_by"] == saved_run["decider"]["display_name"]
            and final["decided_by_role"] == config.ROLE_CONTROLLER
            and final["decision_reason"] == saved_run["decision_reason"]
            and final["decided_at"] is not None), (
        "The decision is recorded as "
        + str({key: final[key] for key in ["decision", "decided_by", "decided_by_role",
                                           "decision_reason", "decided_at"]})
        + ". It has to name the decision, the person, their role, their reason as written, and "
        "when."
    )
    detail = saved_run["detail"]
    assert detail["status"] == 200 and isinstance(detail["body"], dict), (
        "The detail of "
        + saved_run["reference"]
        + " returned "
        + str(detail["status"])
        + ", so the spots the tools found compliant cannot be compared with the record."
    )
    results = detail["body"]["spot_results"]
    compliant = sorted(result["spot_reference"] for result in results
                       if not result["breach_codes"])
    breaching = sorted(result["spot_reference"] for result in results if result["breach_codes"])
    assert (sorted(final["spots_cleared"] or []) == compliant
            and sorted(final["spots_withdrawn"] or []) == breaching
            and final["spots_cleared_count"] == len(compliant)
            and final["spots_withdrawn_count"] == len(breaching)), (
        "Clearing with restrictions recorded "
        + str(final["spots_cleared"])
        + " as cleared and "
        + str(final["spots_withdrawn"])
        + " as withdrawn, counted as "
        + str((final["spots_cleared_count"], final["spots_withdrawn_count"]))
        + ", where the tools find "
        + str(compliant)
        + " compliant and "
        + str(breaching)
        + " breaching. The spots cleared are worked out from the tools' results the moment the "
        "decision arrives, never chosen by an agent."
    )
    client = service_client["client"]
    again = client.post(
        "/clearances/" + at_gate["assessment_id"] + config.DECISION_PATH_ENDING,
        json={"decision": config.DECISION_REJECT,
              "decision_reason": "A second decision on the same assessment."},
        headers=saved_run["decider"]["headers"])
    assert again.status_code == 409, (
        "Deciding an assessment that was already decided returned "
        + str(again.status_code)
        + " where it is refused with 409. A second decision would withdraw cleared spots or "
        "overwrite the first."
    )
    listed = client.get("/clearances", params={"status_value": config.CAMPAIGN_STATUS_OPTIONS},
                        headers=staff["executive"]["headers"])
    same_assessment = ([record for record in listed.json()
                        if record.get("campaign_reference") == saved_run["reference"]]
                       if listed.status_code == 200 else [])
    assert (len(same_assessment) == 1
            and same_assessment[0]["assessment_id"] == at_gate["assessment_id"]
            and same_assessment[0]["decision"] == saved_run["decision"]
            and same_assessment[0]["status"] == final["status"]), (
        "After the second decision was refused, the campaign is listed as "
        + str([{key: record.get(key) for key in ["assessment_id", "status", "decision"]}
               for record in same_assessment])
        + ". It is listed once, with its one assessment, the first decision and its status."
    )


def test_the_stored_assessment_agrees_with_the_tools_that_produced_it(saved_run):
    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["detail"]["status"] == 200 and isinstance(saved_run["detail"]["body"],
                                                               dict), (
        "The detail of "
        + saved_run["reference"]
        + " returned "
        + str(saved_run["detail"]["status"])
        + " with "
        + str(saved_run["detail"]["body"])[:300]
        + ", so the facts the tools work out cannot be compared with the record."
    )
    stored = saved_run["at_gate"]
    detail = saved_run["detail"]["body"]
    summary = detail["summary"]
    breaching = [result for result in detail["spot_results"] if result["breach_codes"]]
    comparisons = [
        ("amendment_score", stored["amendment_score"], detail["amendment"]["amendment_score"]),
        ("amendment_band", stored["amendment_band"], detail["amendment"]["amendment_band"]),
        ("issue_category", stored["issue_category"], detail["issue"]["issue_category"]),
        ("issue_agreement", stored["issue_agreement"], detail["issue"]["weighted_agreement"]),
        ("code_document", stored["code_document"], detail["code_rules"]["document_name"]),
        ("spots_booked", stored["spots_booked"], summary["spots_booked"]),
        ("spots_compliant", stored["spots_compliant"], summary["spots_compliant"]),
        ("spots_breaching", stored["spots_breaching"], summary["spots_breaching"]),
        ("breaches_by_rule", stored["breaches_by_rule"], summary["breaches_by_rule"]),
        ("decisions_open", stored["decisions_open"], summary["decisions_open"]),
        ("first transmission", (stored["first_transmission"]["broadcast_day"],
                                stored["first_transmission"]["break_time"]),
         (summary["first_transmission"]["broadcast_day"],
          summary["first_transmission"]["break_time"])),
        ("exposure_thousands", stored["exposure_thousands"], detail["exposure"]),
        ("breaching spots", sorted((spot["spot_reference"], tuple(spot["breach_codes"]))
                                   for spot in stored["breaching_spots"]),
         sorted((result["spot_reference"], tuple(result["breach_codes"]))
                for result in breaching)),
    ]
    for field_name, stored_value, worked_out in comparisons:
        agrees = (abs(float(stored_value) - float(worked_out)) < 0.00051
                  if isinstance(worked_out, float) else stored_value == worked_out)
        assert agrees, (
            "The stored "
            + field_name
            + " is "
            + str(stored_value)
            + " where the tools, run again on the same campaign for its detail, give "
            + str(worked_out)
            + ". The record holds the facts the decision was taken on, the tools' own and never "
            "an agent's."
        )


def test_signing_in_and_out_decides_who_the_service_will_serve(service_client):
    from access.password_security import hash_password
    from config import config
    from persistence.database import session_factory
    from persistence.models import UserAccount
    from persistence.repository import fetch_user_account

    assert service_client["problem"] is None, service_client["problem"]
    client = service_client["client"]
    executives = [account for account in config.SEED_USER_ACCOUNTS
                  if account["role"] == config.ROLE_EXECUTIVE]
    account = executives[-1]
    credentials = {"username": account["username"], "password": account["password"]}
    signed_in = client.post("/auth/signin", json=credentials)
    body = signed_in.json() if signed_in.status_code == 200 else {}
    assert ({"session_token", "display_name", "role", "assigned_agencies", "pages",
             "areas"} <= set(body) and str(body["session_token"]).strip()), (
        "Signing in as "
        + account["username"]
        + " with the seeded password returned "
        + str(signed_in.status_code)
        + " with "
        + signed_in.text[:200]
        + ", where a sign-in returns a session token with the user's name, role, agencies, "
        "pages and areas."
    )
    assert (body["role"] == account["role"] and body["display_name"] == account["display_name"]
            and sorted(body["pages"]) == sorted(config.ROLE_PAGE_ACCESS[account["role"]])
            and sorted(body["areas"]) == sorted(config.ROLE_AREA_ACCESS[account["role"]])), (
        "Signing in as a "
        + account["role"]
        + " returned the name "
        + repr(body["display_name"])
        + ", the role "
        + str(body["role"])
        + ", the pages "
        + str(body["pages"])
        + " and the areas "
        + str(body["areas"])
        + ", which are not the account's name and what the configuration grants that role."
    )
    wrong = client.post("/auth/signin", json={"username": account["username"],
                                              "password": "wrong-" + account["password"]})
    stranger = client.post("/auth/signin", json={"username": "no-such-person",
                                                 "password": account["password"]})
    assert wrong.status_code == 401 and stranger.status_code == 401, (
        "A wrong password returned "
        + str(wrong.status_code)
        + " and an unknown username "
        + str(stranger.status_code)
        + ". Both are refused."
    )
    assert wrong.json() == stranger.json(), (
        "A wrong password is refused with "
        + repr(wrong.json())
        + " and an unknown username with "
        + repr(stranger.json())
        + ". Two different refusals tell a stranger which usernames exist."
    )
    parameters = {"product_category": config.QUEUE_CATEGORY_ALL}
    first_headers = {config.SESSION_TOKEN_HEADER: body["session_token"]}
    accepted = client.get("/queue", params=parameters, headers=first_headers)
    assert accepted.status_code == 200, (
        "The token just issued to "
        + account["username"]
        + " was not accepted by the queue. A fresh session token opens the areas its role is "
        "granted."
    )
    for headers in [None, {config.SESSION_TOKEN_HEADER: "not-a-token-anyone-was-given"}]:
        refused = client.get("/queue", params=parameters, headers=headers)
        assert refused.status_code == 401, (
            "The queue answered "
            + str(refused.status_code)
            + " to a request "
            + ("with no token" if headers is None else "with an invented token")
            + ". Only a signed-in user may read it."
        )
    again = client.post("/auth/signin", json=credentials)
    second_headers = {config.SESSION_TOKEN_HEADER: again.json().get("session_token", "")}
    replaced = client.get("/queue", params=parameters, headers=first_headers)
    current = client.get("/queue", params=parameters, headers=second_headers)
    assert (again.status_code == 200 and replaced.status_code == 401
            and current.status_code == 200), (
        "After "
        + account["username"]
        + " signed in a second time, answered "
        + str(again.status_code)
        + ", the first token got "
        + str(replaced.status_code)
        + " and the second "
        + str(current.status_code)
        + " from the queue. An account holds one session: signing in somewhere else ends the "
        "session before it."
    )
    signed_out = client.post("/auth/signout", headers=second_headers)
    after = client.get("/queue", params=parameters, headers=second_headers)
    assert signed_out.status_code == 200 and after.status_code == 401, (
        "Signing out answered "
        + str(signed_out.status_code)
        + ", and the old token then got "
        + str(after.status_code)
        + " from the queue. Signing out answers 200 and ends the session."
    )
    with session_factory() as session:
        stored = fetch_user_account(session, UserAccount.username == account["username"])
        other = fetch_user_account(session, UserAccount.username == executives[0]["username"])
    assert (stored["password_hash"] != account["password"]
            and stored["password_hash"] == hash_password(account["password"],
                                                         stored["password_salt"])), (
        "The password stored for "
        + account["username"]
        + " is not the salted hash of the password. A password kept as written is read by "
        "anyone who can open the database."
    )
    assert (stored["password_salt"] != other["password_salt"]
            and stored["password_hash"] != other["password_hash"]), (
        account["username"]
        + " and "
        + executives[0]["username"]
        + " share a password and are stored with the same salt or the same hash, so one "
        "cracked hash opens both accounts."
    )


def test_every_role_reaches_exactly_the_areas_it_is_granted(service_client, staff):
    from config import config

    assert staff["problem"] is None, staff["problem"]
    client = service_client["client"]
    liaison = [entry for entry in config.SEED_USER_ACCOUNTS
               if entry["role"] == config.ROLE_LIAISON][1]
    reply = client.post("/auth/signin", json={"username": liaison["username"],
                                              "password": liaison["password"]})
    assert reply.status_code == 200, (
        "Signing in as "
        + liaison["username"]
        + " returned "
        + str(reply.status_code)
        + ", so what an Agency Liaison may reach cannot be probed."
    )
    headers = {config.ROLE_CONTROLLER: staff["controller"]["headers"],
               config.ROLE_EXECUTIVE: staff["executive"]["headers"],
               config.ROLE_LIAISON: {config.SESSION_TOKEN_HEADER: reply.json()["session_token"]}}
    unknown_campaign = "CLR-ZZZ-9999999"
    unknown_assessment = config.ASSESSMENT_ID_PREFIX + "DOES-NOT-EXIST"
    probes = [
        (config.AREA_INTAKE, "GET", "/queue", {"product_category": config.QUEUE_CATEGORY_ALL},
         None, 200),
        (config.AREA_INTAKE, "GET", "/queue/" + unknown_campaign, None, None, 404),
        (config.AREA_ASSESSMENT, "POST", "/assessments", None,
         {"campaign_reference": unknown_campaign,
          "executive_request": "Check every booked spot."}, 404),
        (config.AREA_AUDIT, "GET", "/clearances",
         {"status_value": [config.CAMPAIGN_STATUS_PENDING]}, None, 200),
        (config.AREA_AUDIT, "GET", "/clearances/" + unknown_assessment + "/calls", None, None,
         404),
        (config.AREA_AUDIT, "GET", "/audit/chat-turns", {"turn_limit": 1}, None, 200),
        (config.AREA_DECISION, "POST",
         "/clearances/" + unknown_assessment + config.DECISION_PATH_ENDING, None,
         {"decision": config.DECISION_REJECT,
          "decision_reason": "Nothing further is needed here."}, 404),
        (config.AREA_CHAT, "GET", "/chat/turns", {"turn_limit": 1}, None, 200),
    ]
    for role_name, role_headers in headers.items():
        for area_name, method, path, parameters, payload, status_when_granted in probes:
            reply = client.request(method, path, params=parameters, json=payload,
                                   headers=role_headers)
            granted = area_name in config.ROLE_AREA_ACCESS[role_name]
            assert (reply.status_code == 403) != granted and reply.status_code != 401, (
                "A "
                + role_name
                + " calling "
                + method
                + " "
                + path
                + " got "
                + str(reply.status_code)
                + ". The "
                + area_name
                + " area is "
                + ("granted" if granted else "not granted")
                + " to that role, and the guard runs before the endpoint: a role without the "
                "area is refused with 403 whether or not the thing asked for exists."
            )
            assert not granted or reply.status_code == status_when_granted, (
                "A "
                + role_name
                + " calling "
                + method
                + " "
                + path
                + " got "
                + str(reply.status_code)
                + " where it is "
                + str(status_when_granted)
                + (". A request that names a campaign or an assessment that does not exist is "
                   "answered 404, and an assessment opened on a campaign not in the queue stops "
                   "before it spends a language model call."
                   if status_when_granted == 404 else
                   ". A role granted an area is served by every endpoint in it.")
            )
    refused = client.get("/queue", params={"product_category": config.QUEUE_CATEGORY_ALL},
                         headers=headers[config.ROLE_LIAISON])
    expected = config.AREA_REFUSED_MESSAGE.format(role=config.ROLE_LIAISON,
                                                  area=config.AREA_INTAKE)
    assert refused.status_code == 403 and refused.json().get("detail") == expected, (
        "An Agency Liaison reading the queue was refused with "
        + refused.text[:200]
        + " where the refusal says, in the configuration's words, "
        + repr(expected)
        + "."
    )


def test_the_queue_puts_the_largest_exposure_first(service_client, staff, open_queue):
    from config import config

    assert staff["problem"] is None, staff["problem"]
    assert open_queue["problem"] is None, open_queue["problem"]
    client = service_client["client"]
    headers = staff["executive"]["headers"]
    reply = client.get("/queue", params={"product_category": config.QUEUE_CATEGORY_ALL},
                       headers=headers)
    assert reply.status_code == 200 and isinstance(reply.json(), list), (
        "The queue returned "
        + str(reply.status_code)
        + " with "
        + reply.text[:200]
        + " to a signed-in executive."
    )
    everything = reply.json()
    missing = sorted({"queue_position", config.CAMPAIGN_REFERENCE_COLUMN,
                      config.PRODUCT_CATEGORY_COLUMN, "spots_booked", "spots_breaching",
                      "first_transmission", "exposure", "amendment_score", "amendment_band"}
                     - set(everything[0] if everything else {}))
    assert len(missing) == 0, (
        "A queue row carries no "
        + str(missing)
        + ", so the queue cannot be ordered, filtered or read."
    )
    references = [row[config.CAMPAIGN_REFERENCE_COLUMN] for row in everything]
    live_references = open_queue["campaigns"][config.CAMPAIGN_REFERENCE_COLUMN]
    assert sorted(references) == sorted(live_references), (
        "The queue holds "
        + str(len(everything))
        + " campaigns where the live extract holds "
        + str(len(open_queue["campaigns"]))
        + ". Every open campaign is in the queue exactly once."
    )
    keys = [(-float(row["exposure"]),
             config.WEEKDAYS.index(row["first_transmission"]["broadcast_day"]),
             int(row["first_transmission"]["break_time"][:2]) * 60
             + int(row["first_transmission"]["break_time"][3:]),
             row[config.CAMPAIGN_REFERENCE_COLUMN]) for row in everything]
    out_of_order = [(references[position], references[position + 1])
                    for position in range(len(keys) - 1) if keys[position] > keys[position + 1]]
    assert len(out_of_order) == 0, (
        "The queue is not ordered by exposure, largest first, ties by the earliest first "
        "transmission in the booking week and then by campaign reference: "
        + str(out_of_order[:3])
        + ". The desk works the queue from the top."
    )
    assert [row["queue_position"] for row in everything] == list(range(1, len(everything) + 1)), (
        "The queue positions run "
        + str([row["queue_position"] for row in everything][:10])
        + " ... where they count from 1 down the queue."
    )
    inconsistent = [row[config.CAMPAIGN_REFERENCE_COLUMN] for row in everything
                    if (float(row["exposure"]) > 0) != (row["spots_breaching"] > 0)
                    or not 0 <= row["spots_breaching"] <= row["spots_booked"]]
    assert len(inconsistent) == 0, (
        str(len(inconsistent))
        + " campaigns, for example "
        + str(inconsistent[:3])
        + ", show an exposure that does not follow from their breaching spots. The exposure is "
        "the forecast under-18 audience of the programmes the breaching spots would reach: none "
        "for a campaign whose every spot complies, and some for one with a breaching spot."
    )
    worked = [row for row in everything
              if row[config.CAMPAIGN_REFERENCE_COLUMN] == "CLR-QSC-0042117"]
    assert (len(worked) == 1 and worked[0]["spots_booked"] == 10
            and worked[0]["spots_breaching"] == 6
            and (worked[0]["first_transmission"]["broadcast_day"],
                 worked[0]["first_transmission"]["break_time"]) == ("Saturday", "18:20")), (
        "The published worked example CLR-QSC-0042117 is in the queue as "
        + str(worked[:1])
        + ", where it has 10 spots booked, 6 of them breaching, and its first spot goes out on "
        "Saturday at 18:20."
    )
    for category in config.QUEUE_CATEGORY_OPTIONS[1:]:
        filtered = client.get("/queue", params={"product_category": category},
                              headers=headers).json()
        assert filtered == [row for row in everything
                            if row[config.PRODUCT_CATEGORY_COLUMN] == category], (
            "Filtering the queue to "
            + category
            + " did not return exactly the "
            + category
            + " campaigns of the full queue, in the same order and with the same positions."
        )
    missing_detail = client.get("/queue/CLR-ZZZ-9999999", headers=headers)
    assert missing_detail.status_code == 404, (
        "The detail of a campaign that is not in the queue returned "
        + str(missing_detail.status_code)
        + " rather than 404."
    )
    first = everything[0]
    detail = client.get("/queue/" + first[config.CAMPAIGN_REFERENCE_COLUMN],
                        headers=headers).json()
    assert (detail.get("queue_position") == first["queue_position"]
            and abs(float(detail.get("exposure", -1)) - float(first["exposure"])) < 0.051
            and detail["summary"]["spots_breaching"] == first["spots_breaching"]
            and detail["summary"]["spots_booked"] == first["spots_booked"]), (
        "The detail of "
        + first[config.CAMPAIGN_REFERENCE_COLUMN]
        + " gives its position, exposure and spots as "
        + str({key: detail.get(key) for key in ["queue_position", "exposure"]})
        + " and "
        + str(detail.get("summary"))[:200]
        + " where the queue shows "
        + str(first)
        + ". The panel and the queue work from the same tools."
    )


def test_a_code_rules_question_is_answered_from_the_documents_and_remembered(chat_replies,
                                                                              tmp_path):
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from chat.conversation_memory import recall_conversation
    from config import config
    from persistence.database import session_factory
    from persistence.models import Base
    from persistence.repository import insert_chat_turn

    assert chat_replies["problem"] is None, chat_replies["problem"]
    for name, meaning in [("rules", "Asking the clearance assistant how many alcohol spots one "
                                    "channel may carry in a broadcast day"),
                          ("turns", "Reading back the executive's own chat turns")]:
        assert chat_replies[name]["status"] == 200, (
            meaning
            + " returned "
            + str(chat_replies[name]["status"])
            + " with "
            + str(chat_replies[name]["body"])[:300]
            + "."
        )
    reply = chat_replies["rules"]["body"]
    missing = sorted({"question_type", "campaign_reference", "answer", "source_documents",
                      "was_refused"} - set(reply))
    assert len(missing) == 0, (
        "The assistant's reply carries no "
        + str(missing)
        + ", so what it answered and where from cannot be read."
    )
    assert reply["was_refused"] is False and reply["question_type"] in config.CHAT_ANSWERED_TYPES, (
        "Asking how many alcohol spots one channel may carry in a broadcast day was answered as "
        + str(reply["question_type"])
        + (" and refused" if reply["was_refused"] else "")
        + ". The alcohol code answers it, so a refusal means the search never reached it."
    )
    folder = {name for name in os.listdir(config.DOCS_DIRECTORY) if name.lower().endswith(".pdf")}
    cited = reply["source_documents"]
    assert len(cited) > 0 and set(cited) <= folder, (
        "The answer cites "
        + str(cited)
        + ". An answer that cannot name the documents it came from is an answer from general "
        "knowledge."
    )
    assert not set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS), (
        "The answer cites "
        + str(sorted(set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS)))
        + ", which the assistant must not read: it records what happened to earlier campaigns."
    )
    assert len(reply["answer"].strip()) > 0 and reply["answer"] != config.CHAT_REFUSAL_MESSAGE, (
        "The answer given was "
        + repr(reply["answer"][:120])
        + ", which answers nothing."
    )
    turns = chat_replies["turns"]["body"]
    asked = [turn for turn in turns if turn["question_text"] == chat_replies["rules_question"]
             and turn["turn_id"] > chat_replies["latest_turn_before"]]
    assert len(asked) > 0, (
        "The executive's latest turns read back as "
        + str([turn["question_text"] for turn in turns])
        + ", which do not include the question just answered, so the audit trail and the chat "
        "page lose it."
    )
    latest = asked[0]
    assert (latest["question_type"] == reply["question_type"] and latest["was_refused"] is False
            and latest["answer_text"] == reply["answer"]), (
        "The executive's stored turn is "
        + str({key: latest[key] for key in ["question_text", "question_type", "was_refused"]})
        + ", which is not how the question was just answered."
    )
    assert latest["closest_distance"] <= config.CHAT_MAX_RELEVANCE_DISTANCE, (
        "An answered question is stored with its closest passage "
        + str(latest["closest_distance"])
        + " away, beyond the "
        + str(config.CHAT_MAX_RELEVANCE_DISTANCE)
        + " within which an answer counts as grounded."
    )
    assert all(name in latest["source_documents"] for name in cited), (
        "The stored turn cites "
        + repr(latest["source_documents"])
        + " where the answer cited "
        + str(cited)
        + ". The audit trail has to show where each answer came from."
    )
    follow_up = "And how far apart must those spots be?"
    with session_factory() as session:
        remembered = recall_conversation(session, chat_replies["asker"], follow_up)
        someone_else = recall_conversation(session, chat_replies["someone_else"], follow_up)
    assert chat_replies["rules_question"] in remembered["history_text"], (
        "The conversation recalled for the executive's next question does not include the "
        "question just asked, so a follow-up such as "
        + repr(follow_up)
        + " would be read with no idea what it follows."
    )
    assert (follow_up in remembered["search_text"]
            and chat_replies["rules_question"] in remembered["search_text"]), (
        "The text searched for the follow-up is "
        + repr(remembered["search_text"][:200])
        + ". It has to carry the new question and the question asked just before it."
    )
    assert (chat_replies["rules_question"] not in someone_else["history_text"]
            and chat_replies["rules_question"] not in someone_else["search_text"]), (
        "The conversation recalled for an Agency Liaison carries the executive's question "
        + repr(chat_replies["rules_question"])
        + ". Each user's memory is their own last exchanges: a desk answer can carry what an "
        "agency must never hear."
    )
    status_answer = ("CLR-QSC-0042117 is under review. Record ASM-PLANTED00001, opened by tom; "
                     "score 0.4321, band LOW; young audience 123.4.")
    planted_engine = create_engine("sqlite:///" + str(tmp_path / "planted_chat.db"))
    Base.metadata.create_all(planted_engine)
    with sessionmaker(bind=planted_engine)() as session:
        insert_chat_turn(session, {
            "username": chat_replies["asker"], "user_role": config.ROLE_EXECUTIVE,
            "question_text": "Where has CLR-QSC-0042117 got to?",
            "question_type": config.CHAT_TYPE_CAMPAIGN_STATUS,
            "campaign_reference": "CLR-QSC-0042117", "answer_text": status_answer,
            "source_documents": "", "closest_distance": 0.5, "was_refused": False})
        after_status = recall_conversation(session, chat_replies["asker"],
                                           "And has the agency been told?")
    carried = [fact for fact in ["ASM-PLANTED00001", "0.4321", "123.4"]
               if fact in after_status["history_text"]]
    assert len(carried) == 0, (
        "After an executive was told a campaign's record, the conversation recalled for their "
        "next question carries "
        + str(carried)
        + ". The recalled conversation goes into the next prompt, and nothing from a campaign's "
        "record is ever put in front of the language model, so a campaign status answer is "
        "never replayed."
    )
    assert config.CHAT_STATUS_ANSWER_IN_HISTORY in after_status["history_text"], (
        "After an executive was told a campaign's status, the conversation recalled for their "
        "next question does not carry "
        + repr(config.CHAT_STATUS_ANSWER_IN_HISTORY)
        + " where that answer stood, so the next prompt cannot tell that the question was "
        "answered from the record."
    )


def test_an_agency_liaison_hears_only_about_their_own_agencies_campaigns(saved_run, open_queue,
                                                                         staff):
    from datetime import datetime

    from chat.campaign_status_answer import answer_campaign_status
    from config import config
    from persistence.database import session_factory
    from persistence.models import ClearanceAssessment
    from persistence.repository import fetch_assessment_records

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_run["heard_at_gate"]["problem"] is None, saved_run["heard_at_gate"]["problem"]
    assert saved_run["heard_after"]["problem"] is None, saved_run["heard_after"]["problem"]
    reference = saved_run["reference"]
    at_gate = saved_run["at_gate"]
    final = saved_run["final"]
    pending_line = config.CHAT_STATUS_TEMPLATE.format(
        campaign_reference=reference,
        status_phrase=config.CHAT_STATUS_PHRASES[config.CAMPAIGN_STATUS_PENDING],
        changed_at=datetime.fromisoformat(str(at_gate["status_changed_at"])).strftime(
            config.STATUS_CHANGE_TIME_FORMAT))
    decided_line = config.CHAT_STATUS_TEMPLATE.format(
        campaign_reference=reference,
        status_phrase=config.CHAT_STATUS_PHRASES[final["status"]],
        changed_at=datetime.fromisoformat(str(final["status_changed_at"])).strftime(
            config.STATUS_CHANGE_TIME_FORMAT))
    decided_lines = " ".join([
        decided_line,
        config.CHAT_DECIDED_TEMPLATE.format(decision=final["decision"],
                                            spots_cleared=final["spots_cleared_count"],
                                            spots_withdrawn=final["spots_withdrawn_count"]),
        config.CHAT_DOCUMENTS_SENT[final["status"]]])
    private = {"the assessment identifier": final["assessment_id"],
               "who opened it": final["executive_id"], "who decided it": final["decided_by"],
               "the reason for the decision": final["decision_reason"],
               "the issue the notes name": final["issue_category"],
               "a cleared spot's reference": (final["spots_cleared"] or ["no spot"])[0]}
    for moment, heard, expected in [
            ("while the run waited for a Clearance Controller", saved_run["heard_at_gate"],
             pending_line),
            ("after the decision", saved_run["heard_after"], decided_lines)]:
        answer = str(heard["assigned"])
        leaked = sorted(description for description, fact in private.items()
                        if str(fact) in answer)
        assert len(leaked) == 0, (
            "The Agency Liaison assigned to "
            + reference
            + "'s agency, asking "
            + moment
            + ", was told "
            + repr(answer[:300])
            + ", which gives away "
            + ", ".join(leaked)
            + ". A liaison hears where the campaign has got to, and what it means for the "
            "agency, never the desk's record."
        )
        assert answer == expected, (
            "The Agency Liaison assigned to "
            + reference
            + "'s agency, asking "
            + moment
            + ", was told "
            + repr(answer[:300])
            + " where they hear exactly "
            + repr(expected)
            + ": the status and when it last changed, in the configuration's own words, and once "
            "it is decided the decision, how many spots were cleared and withdrawn, and the "
            "documents the agency has been sent."
        )
        assert heard["rival"] == config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE, (
            "A liaison assigned to other agencies, asking about "
            + reference
            + " "
            + moment
            + ", was told "
            + repr(str(heard["rival"])[:200])
            + " where they hear exactly what they would hear about a reference that does not "
            "exist, "
            + repr(config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE)
            + ". Telling a rival agency that a campaign exists is already telling them too much."
        )
    desk = str(saved_run["heard_after"]["desk"])
    absent = [description for description, fact in private.items() if str(fact) not in desk]
    assert len(absent) == 0 and desk.startswith(decided_line), (
        "A Clearance Executive asking about "
        + reference
        + " after the decision was told "
        + repr(desk[:300])
        + ", which does not give "
        + ", ".join(absent or ["the status first"])
        + ". The desk hears the status and then everything the record holds."
    )
    campaigns = open_queue["campaigns"]
    liaison = saved_run["liaison"]
    with session_factory() as session:
        untouched = next(
            candidate for candidate in campaigns.loc[
                campaigns[config.AGENCY_CODE_COLUMN].isin(liaison["assigned_agencies"]),
                config.CAMPAIGN_REFERENCE_COLUMN]
            if not fetch_assessment_records(
                session, ClearanceAssessment.campaign_reference == candidate))
        unassessed = answer_campaign_status(session, campaigns, liaison, untouched)
        unknown_heard = answer_campaign_status(session, campaigns, liaison, "CLR-ZZZ-9999999")
        desk_unknown = answer_campaign_status(session, campaigns, staff["executive"],
                                              "CLR-ZZZ-9999999")
    expected_unassessed = config.CHAT_NOT_ASSESSED_TEMPLATE.format(campaign_reference=untouched)
    assert unassessed == expected_unassessed, (
        "The liaison asking about "
        + untouched
        + ", one of their agencies' campaigns that nobody has assessed, was told "
        + repr(str(unassessed)[:200])
        + " where they hear exactly "
        + repr(expected_unassessed)
        + "."
    )
    assert (unknown_heard == config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE
            and desk_unknown == config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE), (
        "A question about CLR-ZZZ-9999999, which is not in the queue, was answered "
        + repr(str(unknown_heard)[:120])
        + " for a liaison and "
        + repr(str(desk_unknown)[:120])
        + " for an executive, where both hear "
        + repr(config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE)
        + "."
    )
