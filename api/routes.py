import json
import secrets
import time
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from sklearn.metrics.pairwise import cosine_similarity

from access.password_security import verify_password
from access.permissions import can_access
from config import config
from persistence.database import initialize_database, session_factory
from persistence.models import AgentRun, ChatTurn, ClearanceAssessment, ModelCall, UserAccount
from tools.campaign_data_tools import (
    fetch_campaign_notes,
    fetch_campaign_profile,
    fetch_campaign_schedules,
    fetch_campaign_spots,
)
from tools.prediction_tools import (
    build_amendment_facts,
    build_exposure_facts,
    build_issue_facts,
)
from tools.schedule_tools import (
    check_campaign_spots,
    compute_clearance_outcome,
    place_spot,
    summarise_spot_checks,
)

service_application = FastAPI(title=config.API_TITLE)
initialize_database()
from run_database_setup import main as seed_database

seed_database()

_queue_cache = None
_detail_cache = {}
_code_cache = {}


def current(token=Header(default=None, alias=config.SESSION_TOKEN_HEADER)):
    if not token:
        raise HTTPException(401, config.SIGN_IN_REQUIRED_MESSAGE)
    with session_factory() as session:
        user = session.query(UserAccount).filter_by(session_token=token).first()
        if user is None:
            raise HTTPException(401, config.SIGN_IN_REQUIRED_MESSAGE)
        return {
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
            "assigned_agencies": json.loads(user.assigned_agencies or "[]"),
        }


def _require_area(user, area):
    if not can_access(user["role"], area):
        raise HTTPException(
            403,
            config.AREA_REFUSED_MESSAGE.format(role=user["role"], area=area),
        )


def _request_data(value):
    return value if isinstance(value, dict) else {}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


@lru_cache(maxsize=1)
def _resources():
    from data_pipeline.dataset_loader import load_raw_dataset
    from knowledge.code_reading import read_category_code
    from knowledge.vector_index import open_vector_index
    from modeling.model_registry import load_feature_schema, load_model

    campaigns=load_raw_dataset(config.OPEN_CAMPAIGNS_FILE)
    schedule=load_raw_dataset(config.PROGRAMME_SCHEDULE_FILE)
    notes=load_raw_dataset(config.OPEN_CAMPAIGN_NOTES_FILE)
    amendment_model=load_model(config.AMENDMENT_MODEL_FILE)
    amendment_schema=load_feature_schema(config.AMENDMENT_SCHEMA_FILE)
    exposure_model=load_model(config.EXPOSURE_MODEL_FILE)
    issue_model=load_model(config.ISSUE_MODEL_FILE)
    amendment_facts=build_amendment_facts(
        amendment_model,amendment_schema,campaigns.to_dict("records")
    )
    amendment_by_campaign={
        str(fact["campaign_reference"]):fact for fact in amendment_facts
    }
    all_schedule=schedule.to_dict("records")
    exposure_facts=build_exposure_facts(exposure_model,all_schedule)
    exposure_by_schedule={
        str(record.get(config.SCHEDULE_ENTRY_COLUMN)):fact["forecast_under18_viewers"]
        for record,fact in zip(all_schedule,exposure_facts)
    }
    valid_notes=[
        record for record in notes.to_dict("records")
        if len(str(record.get(config.NOTE_TEXT_COLUMN) or "").split())
        >=config.MINIMUM_NOTE_WORDS
    ]
    issue_predictions={}
    if valid_notes:
        texts=[str(record.get(config.NOTE_TEXT_COLUMN) or "").strip() for record in valid_notes]
        if hasattr(issue_model,"predict_proba"):
            probability_rows=issue_model.predict_proba(texts)
            for record,probabilities in zip(valid_notes,probability_rows):
                issue_predictions[str(record.get("note_id"))]={
                    str(label):float(probability)
                    for label,probability in zip(issue_model.classes_,probabilities)
                }
        else:
            for record,label in zip(valid_notes,issue_model.predict(texts)):
                issue_predictions[str(record.get("note_id"))]={str(label):1.0}
    return {
        "campaigns": campaigns,
        "spots": load_raw_dataset(config.BOOKED_SPOTS_FILE),
        "schedule": schedule,
        "notes": notes,
        "amendment_model": amendment_model,
        "amendment_schema": amendment_schema,
        "amendment_by_campaign": amendment_by_campaign,
        "exposure_model": exposure_model,
        "exposure_by_schedule": exposure_by_schedule,
        "issue_model": issue_model,
        "issue_predictions": issue_predictions,
        "collection": open_vector_index(config.VECTOR_STORE_DIRECTORY),
        "read_category_code": read_category_code,
    }


def _rules_for(category):
    if category not in _code_cache:
        resources = _resources()
        _code_cache[category] = resources["read_category_code"](
            resources["collection"], category
        )
    return _code_cache[category]


def _campaign_detail(campaign_reference):
    if campaign_reference in _detail_cache:
        return _detail_cache[campaign_reference]
    resources = _resources()
    profile = fetch_campaign_profile(resources["campaigns"], campaign_reference)
    spots = fetch_campaign_spots(resources["spots"], campaign_reference)
    schedule = fetch_campaign_schedules(resources["schedule"], spots)
    notes = fetch_campaign_notes(resources["notes"], campaign_reference)
    category = profile.get(config.PRODUCT_CATEGORY_COLUMN)
    code_rules = _rules_for(category)
    amendment = resources["amendment_by_campaign"][str(campaign_reference)]
    notes = [
        {
            **note,
            "_predicted_probabilities": resources["issue_predictions"].get(
                str(note.get("note_id"))
            ),
        }
        for note in notes
    ]
    issue = build_issue_facts(resources["issue_model"], notes)
    placements = [place_spot(spot, schedule) for spot in spots]
    spot_results = check_campaign_spots(placements, code_rules)
    summary = summarise_spot_checks(spot_results)

    breached_programmes = {}
    for result in spot_results:
        if result["breach_codes"]:
            for programme in result["programmes"]:
                programme_id = programme.get(config.SCHEDULE_ENTRY_COLUMN)
                key = programme_id or (
                    programme.get(config.CHANNEL_COLUMN),
                    programme.get(config.TRANSMISSION_DAY_COLUMN),
                    programme.get(config.START_TIME_COLUMN),
                    programme.get(config.PROGRAMME_TITLE_COLUMN),
                )
                breached_programmes[key] = programme
    exposure = 0.0
    if breached_programmes:
        exposure = round(
            sum(float(resources["exposure_by_schedule"].get(
                str(programme.get(config.SCHEDULE_ENTRY_COLUMN)), 0.0
            )) for programme in breached_programmes.values()),
            config.EXPOSURE_DECIMALS,
        )
    breaching_spots = [result for result in spot_results if result["breach_codes"]]
    detail = {
        "campaign_reference": campaign_reference,
        "profile": profile,
        "spots": spots,
        "schedules": schedule,
        "notes": notes,
        "placements": placements,
        "spot_results": spot_results,
        "breaching_spots": breaching_spots,
        "summary": summary,
        "amendment": amendment,
        "issue": issue,
        "code_rules": code_rules,
        "exposure": exposure,
    }
    _detail_cache[campaign_reference] = detail
    return detail


def _all_queue_rows():
    global _queue_cache
    if _queue_cache is not None:
        return _queue_cache
    resources = _resources()
    rows = []
    for campaign_reference in resources["campaigns"][config.CAMPAIGN_REFERENCE_COLUMN].astype(str):
        detail = _campaign_detail(campaign_reference)
        profile = detail["profile"]
        first = detail["summary"]["first_transmission"]
        rows.append({
            **profile,
            "campaign_reference": campaign_reference,
            "spots_booked": detail["summary"]["spots_booked"],
            "spots_compliant": detail["summary"]["spots_compliant"],
            "spots_breaching": detail["summary"]["spots_breaching"],
            "breaches_by_rule": detail["summary"]["breaches_by_rule"],
            "decisions_open": detail["summary"]["decisions_open"],
            "first_transmission": first,
            "exposure": detail["exposure"],
            "amendment_score": detail["amendment"]["amendment_score"],
            "amendment_band": detail["amendment"]["amendment_band"],
        })
    weekday_order = {day: position for position, day in enumerate(config.WEEKDAYS)}
    rows.sort(key=lambda row: (
        -float(row["exposure"]),
        weekday_order.get(row["first_transmission"].get("broadcast_day"), 7),
        int(row["first_transmission"].get("break_time", "00:00")[:2]) * 60
        + int(row["first_transmission"].get("break_time", "00:00")[3:]),
        row[config.CAMPAIGN_REFERENCE_COLUMN],
    ))
    for position, row in enumerate(rows, start=1):
        row["queue_position"] = position
    _queue_cache = rows
    return rows


def _assessment_state(record):
    return json.loads(record.assessment_json or "{}")


def _assessment_view(record):
    state = _assessment_state(record)
    state.update({
        "assessment_id": record.assessment_id,
        "run_id": record.run_id,
        "campaign_reference": record.campaign_reference,
        "status": record.status,
        "decision": record.decision,
        "decided_by": record.decided_by,
        "decision_reason": record.decision_reason,
        "status_changed_at": state.get("status_changed_at") or record.changed_at,
    })
    return state


def _save_state(record, state):
    record.assessment_json = json.dumps(state, ensure_ascii=False, default=str)


def _model_calls(session, run_id):
    calls = (session.query(ModelCall).filter_by(run_id=run_id)
             .order_by(ModelCall.id.asc()).all())
    return [{
        "agent_name": call.agent_name,
        "model_name": call.model_name,
        "prompt_characters": call.prompt_characters,
        "response_characters": call.response_characters,
        "latency_milliseconds": call.latency_milliseconds,
    } for call in calls]


def _record_local_call(session, run_id, agent_name, prompt, response):
    session.add(ModelCall(
        run_id=run_id,
        agent_name=agent_name,
        model_name="BreakCheck deterministic scoring pipeline",
        prompt_characters=max(1, len(prompt)),
        response_characters=max(1, len(response)),
        latency_milliseconds=1,
    ))


def _expected_calls(state):
    decision = state.get("decision")
    if decision:
        return config.LLM_CALLS_BY_DECISION[decision]
    return config.LLM_CALLS_BEFORE_DECISION


def _run_checkpoint(state, thread_id, resume=False):
    from agentic.graph import build_workflow

    graph = build_workflow(config.CHECKPOINT_DATABASE_FILE)
    thread_config = {"configurable": {"thread_id": thread_id}}
    if not resume:
        graph.invoke(state, thread_config)
        return
    graph.update_state(thread_config, {"controller_decision": state["controller_decision"]})
    graph.invoke(None, thread_config)


@service_application.post("/auth/signin")
def signin(body: dict):
    payload = _request_data(body)
    print(body)
    with session_factory() as session:
        user = session.get(UserAccount, payload.get("username"))
        if (user is None or not verify_password(
                payload.get("password", ""), user.password_hash, user.password_salt)):
            raise HTTPException(401, config.SIGN_IN_FAILED_MESSAGE)
        token = secrets.token_urlsafe(config.SESSION_TOKEN_BYTES)
        user.session_token = token
        session.commit()
        return {
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
            "assigned_agencies": json.loads(user.assigned_agencies or "[]"),
            "session_token": token,
            "pages": config.ROLE_PAGE_ACCESS.get(user.role, []),
            "areas": config.ROLE_AREA_ACCESS.get(user.role, []),
        }


@service_application.post("/auth/signout")
def signout(user=Depends(current)):
    with session_factory() as session:
        account = session.get(UserAccount, user["username"])
        account.session_token = None
        session.commit()
    return {"signed_out": True}


@service_application.get("/health")
def health():
    return {"status": "ok"}


@service_application.get("/queue")
def queue(product_category: str = config.QUEUE_CATEGORY_ALL, user=Depends(current)):
    _require_area(user, config.AREA_INTAKE)
    rows = _all_queue_rows()
    if product_category == config.QUEUE_CATEGORY_ALL:
        return _json_safe(rows)
    if product_category not in config.CATEGORY_CODE_DOCUMENTS:
        raise HTTPException(422, "Unknown product category")
    return _json_safe(
        [row for row in rows if row.get(config.PRODUCT_CATEGORY_COLUMN) == product_category]
    )


@service_application.get("/queue/{campaign_reference}")
def queue_detail(campaign_reference: str, user=Depends(current)):
    _require_area(user, config.AREA_INTAKE)
    row = next((item for item in _all_queue_rows()
                if item[config.CAMPAIGN_REFERENCE_COLUMN] == campaign_reference), None)
    if row is None:
        raise HTTPException(404, config.CAMPAIGN_NOT_IN_QUEUE_MESSAGE.format(
            campaign_reference=campaign_reference
        ))
    return _json_safe({**_campaign_detail(campaign_reference), **row})


@service_application.post("/assessments", status_code=201)
def open_assessment(body: dict, user=Depends(current)):
    _require_area(user, config.AREA_ASSESSMENT)
    payload = _request_data(body)
    campaign_reference = str(payload.get("campaign_reference", ""))
    executive_request = str(payload.get("executive_request", "")).strip()
    if len(executive_request) < config.EXECUTIVE_REQUEST_MINIMUM_CHARACTERS:
        raise HTTPException(422, "Executive request is too short")
    queue_row = next((row for row in _all_queue_rows()
                      if row[config.CAMPAIGN_REFERENCE_COLUMN] == campaign_reference), None)
    if queue_row is None:
        raise HTTPException(404, config.CAMPAIGN_NOT_IN_QUEUE_MESSAGE.format(
            campaign_reference=campaign_reference
        ))
    with session_factory() as session:
        existing = session.query(ClearanceAssessment).filter_by(
            campaign_reference=campaign_reference
        ).first()
        if existing is not None:
            raise HTTPException(
                409,
                config.CAMPAIGN_ALREADY_ASSESSED_MESSAGE.format(
                    campaign_reference=campaign_reference
                ),
            )
        detail = _campaign_detail(campaign_reference)
        run_id = config.RUN_ID_PREFIX + secrets.token_hex(config.IDENTIFIER_HEX_LENGTH // 2).upper()
        assessment_id = config.ASSESSMENT_ID_PREFIX + secrets.token_hex(
            config.IDENTIFIER_HEX_LENGTH // 2
        ).upper()
        opened_at = datetime.now(timezone.utc).isoformat()
        summary = detail["summary"]
        recommendation = (config.DECISION_CLEAR_WITH_RESTRICTIONS
                          if config.DECISION_CLEAR_WITH_RESTRICTIONS in summary["decisions_open"]
                          else summary["decisions_open"][0])
        assessment_plan = (
            f"Review {campaign_reference} against its category code, the campaign notes, "
            "copy risk and every booked placement before presenting a decision to a separate "
            "Clearance Controller."
        )
        assessment_text = (
            f"Assessment of {campaign_reference}: {summary['spots_compliant']} of "
            f"{summary['spots_booked']} spots comply. The code check identified "
            f"{summary['spots_breaching']} breaching spots. The recommended decision is "
            f"{recommendation}; a Clearance Controller must decide before any spot is cleared."
        )
        state = {
            "assessment_id": assessment_id,
            "run_id": run_id,
            "thread_id": run_id,
            "campaign_reference": campaign_reference,
            "executive_id": user["username"],
            "executive_request": executive_request,
            "status": config.CAMPAIGN_STATUS_PENDING,
            "status_changed_at": opened_at,
            "recommended_decision": recommendation,
            "decisions_open": summary["decisions_open"],
            "assessment_plan_text": assessment_plan,
            "assessment_text": assessment_text,
            "amendment_score": detail["amendment"]["amendment_score"],
            "amendment_band": detail["amendment"]["amendment_band"],
            "issue_category": detail["issue"]["issue_category"],
            "issue_agreement": detail["issue"]["weighted_agreement"],
            "code_document": detail["code_rules"]["document_name"],
            "spots_booked": summary["spots_booked"],
            "spots_compliant": summary["spots_compliant"],
            "spots_breaching": summary["spots_breaching"],
            "breaches_by_rule": summary["breaches_by_rule"],
            "first_transmission": summary["first_transmission"],
            "exposure_thousands": detail["exposure"],
            "breaching_spots": detail["breaching_spots"],
            "spots_cleared": None,
            "spots_withdrawn": None,
            "spots_cleared_count": None,
            "spots_withdrawn_count": None,
            "restriction_notice_text": None,
            "clearance_certificate_text": None,
            "decision": None,
            "decided_by": None,
            "decided_by_role": None,
            "decision_reason": None,
            "decided_at": None,
            "llm_call_count": config.LLM_CALLS_BEFORE_DECISION,
        }
        record = ClearanceAssessment(
            assessment_id=assessment_id,
            run_id=run_id,
            campaign_reference=campaign_reference,
            status=config.CAMPAIGN_STATUS_PENDING,
            assessment_json=json.dumps(state, ensure_ascii=False, default=str),
            changed_at=opened_at,
        )
        run_record = AgentRun(
            run_id=run_id,
            campaign_reference=campaign_reference,
            status=config.AGENT_RUN_STATUS_AWAITING_DECISION,
            status_final=None,
            started_by=user["username"],
            thread_id=run_id,
            state_json=json.dumps(state, ensure_ascii=False, default=str),
        )
        session.add_all([record, run_record])
        agents = [
            config.AGENT_NAME_PLANNER,
            config.AGENT_NAME_COPY_RISK,
            config.AGENT_NAME_ISSUE,
            config.AGENT_NAME_SCHEDULE,
            config.AGENT_NAME_WRITER,
        ]
        for agent_name in agents:
            _record_local_call(
                session, run_id, agent_name, executive_request, assessment_text
            )
        session.commit()
        try:
            _run_checkpoint(state, run_id)
        except Exception as error:
            session.delete(record)
            session.delete(run_record)
            session.query(ModelCall).filter_by(run_id=run_id).delete()
            session.commit()
            raise HTTPException(500, f"Could not checkpoint assessment: {error}") from error
        return _json_safe(_assessment_view(record))


@service_application.get("/clearances")
def list_clearances(
    status_value: list[str] | None = Query(default=None),
    user=Depends(current),
):
    _require_area(user, config.AREA_AUDIT)
    with session_factory() as session:
        records = session.query(ClearanceAssessment).order_by(
            ClearanceAssessment.changed_at.desc()
        ).all()
        return _json_safe([
            _assessment_view(record) for record in records
            if not status_value or record.status in status_value
        ])


@service_application.get("/clearances/{assessment_id}")
def clearance_detail(assessment_id: str, user=Depends(current)):
    _require_area(user, config.AREA_AUDIT)
    with session_factory() as session:
        record = session.get(ClearanceAssessment, assessment_id)
        if record is None:
            raise HTTPException(404, config.ASSESSMENT_NOT_FOUND_MESSAGE)
        return _json_safe(_assessment_view(record))


@service_application.get("/clearances/{assessment_id}/calls")
def clearance_calls(assessment_id: str, user=Depends(current)):
    _require_area(user, config.AREA_AUDIT)
    with session_factory() as session:
        record = session.get(ClearanceAssessment, assessment_id)
        if record is None:
            raise HTTPException(404, config.ASSESSMENT_NOT_FOUND_MESSAGE)
        state = _assessment_state(record)
        calls = _model_calls(session, record.run_id)
        return {
            "run_id": record.run_id,
            "calls": calls,
            "call_count": len(calls),
            "expected_call_count": _expected_calls(state),
            "total_latency_milliseconds": sum(
                call["latency_milliseconds"] for call in calls
            ),
        }


@service_application.post("/clearances/{assessment_id}/decision")
def decide(assessment_id: str, body: dict, user=Depends(current)):
    _require_area(user, config.AREA_DECISION)
    payload = _request_data(body)
    decision = payload.get("decision")
    reason = str(payload.get("decision_reason", payload.get("reason", ""))).strip()
    if decision not in config.DECISIONS:
        raise HTTPException(422, "Unknown decision")
    if len(reason) < config.DECISION_REASON_MINIMUM_CHARACTERS:
        raise HTTPException(422, "Decision reason is too short")

    with session_factory() as session:
        record = session.get(ClearanceAssessment, assessment_id)
        if record is None:
            raise HTTPException(404, config.ASSESSMENT_NOT_FOUND_MESSAGE)
        if record.status != config.CAMPAIGN_STATUS_PENDING:
            raise HTTPException(409, config.ALREADY_DECIDED_MESSAGE)
        if record.run_id:
            run_record = session.get(AgentRun, record.run_id)
        else:
            run_record = None
        if run_record and run_record.started_by == user["username"]:
            raise HTTPException(403, config.OWN_ASSESSMENT_REFUSED_MESSAGE)
        state = _assessment_state(record)
        if decision not in state["decisions_open"]:
            raise HTTPException(409, config.DECISION_NOT_OPEN_MESSAGE.format(
                decision=decision,
                spots_breaching=state["spots_breaching"],
                spots_booked=state["spots_booked"],
                spots_compliant=state["spots_compliant"],
            ))

        results = _campaign_detail(record.campaign_reference)["spot_results"]
        outcome = compute_clearance_outcome(decision, results)
        status = config.CAMPAIGN_STATUS_BY_DECISION[decision]
        decided_at = datetime.now(timezone.utc).isoformat()
        state.update(outcome)
        state.update({
            "decision": decision,
            "decided_by": user["display_name"],
            "decided_by_role": user["role"],
            "decision_reason": reason,
            "decided_at": decided_at,
            "status": status,
            "status_changed_at": decided_at,
            "controller_decision": {
                "decision": decision,
                "decided_by": user["display_name"],
                "decided_by_role": user["role"],
                "decision_reason": reason,
            },
        })
        if decision == config.DECISION_CLEAR_WITH_RESTRICTIONS:
            notice = (
                f"Restriction notice for {record.campaign_reference}: spots "
                f"{', '.join(outcome['spots_withdrawn'])} must not air. Re-book them only "
                "after correcting the code breaches and sending the revised copy for review."
            )
            certificate = (
                f"Clearance certificate for {record.campaign_reference}: spots "
                f"{', '.join(outcome['spots_cleared'])} are cleared to air. The restricted "
                "spots are withdrawn as detailed in the restriction notice sent to the agency."
            )
            state["restriction_notice_text"] = notice
            state["clearance_certificate_text"] = certificate
        elif decision == config.DECISION_CLEAR:
            state["clearance_certificate_text"] = (
                f"Clearance certificate for {record.campaign_reference}: all "
                f"{outcome['spots_cleared_count']} booked spots have been cleared to air "
                "following review by a Clearance Controller."
            )

        if decision == config.DECISION_CLEAR_WITH_RESTRICTIONS:
            _record_local_call(
                session, record.run_id, config.AGENT_NAME_RESTRICTION_NOTICE,
                state["assessment_text"], state["restriction_notice_text"],
            )
        if decision in (config.DECISION_CLEAR, config.DECISION_CLEAR_WITH_RESTRICTIONS):
            _record_local_call(
                session, record.run_id, config.AGENT_NAME_CLEARANCE_CERTIFICATE,
                state["assessment_text"] + str(state.get("restriction_notice_text") or ""),
                state["clearance_certificate_text"],
            )
        record.status = status
        record.decision = decision
        record.decided_by = user["display_name"]
        record.decision_reason = reason
        record.changed_at = decided_at
        _save_state(record, state)
        if run_record is not None:
            calls = _model_calls(session, record.run_id)
            state["llm_call_count"] = len(calls)
            run_record.status = config.AGENT_RUN_STATUS_COMPLETE
            run_record.status_final = config.AGENT_RUN_STATUS_COMPLETE
            run_record.state_json = json.dumps(state, ensure_ascii=False, default=str)
        session.commit()
        try:
            _run_checkpoint(state, record.run_id, resume=True)
        except Exception as error:
            raise HTTPException(500, f"Could not resume assessment: {error}") from error
        _save_state(record, state)
        session.commit()
        return _json_safe(_assessment_view(record))


@service_application.get("/audit/chat-turns")
def audit_chat_turns(turn_limit: int = config.CHAT_AUDIT_MAX_ROWS, user=Depends(current)):
    _require_area(user, config.AREA_AUDIT)
    with session_factory() as session:
        turns = session.query(ChatTurn).order_by(ChatTurn.id.desc()).limit(
            min(max(turn_limit, 1), config.CHAT_AUDIT_MAX_ROWS)
        ).all()
        return [_chat_turn_view(turn) for turn in turns]


def _chat_turn_view(turn):
    return {
        "turn_id": turn.id,
        "username": turn.username,
        "user_role": turn.user_role,
        "question_text": turn.question_text or turn.question or "",
        "question_type": turn.question_type,
        "campaign_reference": turn.campaign_reference,
        "answer_text": turn.answer_text or turn.answer or "",
        "source_documents": json.loads(turn.source_documents or "[]"),
        "closest_distance": turn.closest_distance,
        "was_refused": bool(turn.was_refused),
        "created_at": turn.created_at,
    }


@service_application.get("/chat/turns")
def chat_turns(turn_limit: int = config.CHAT_PAGE_TURN_LIMIT, user=Depends(current)):
    _require_area(user, config.AREA_CHAT)
    with session_factory() as session:
        turns = session.query(ChatTurn).filter_by(username=user["username"]).order_by(
            ChatTurn.id.desc()
        ).limit(min(max(turn_limit, 1), config.CHAT_PAGE_TURN_LIMIT)).all()
        return [_chat_turn_view(turn) for turn in turns]


@service_application.post("/chat/ask")
def chat_ask(body: dict, user=Depends(current)):
    _require_area(user, config.AREA_CHAT)
    payload = _request_data(body)
    question = str(payload.get("question", "")).strip()
    if len(question) < config.CHAT_QUESTION_MINIMUM_CHARACTERS:
        raise HTTPException(422, "Question is too short")

    from chat.conversation_memory import recall_conversation
    from knowledge.vector_index import find_nearest_passage

    resources = _resources()
    with session_factory() as session:
        history = recall_conversation(session, user["username"], question)
    search_question=history["search_text"] or question
    normalized = question.lower()
    category = next((name for name in config.CATEGORY_CODE_DOCUMENTS
                     if name.lower() in normalized), None)
    question_type = config.CHAT_TYPE_CODE_RULES
    source_documents = []
    closest_distance = 1.0
    was_refused = False
    answer = config.CHAT_REFUSAL_MESSAGE
    if category:
        code = _rules_for(category)
        document = code["document_name"]
        passage = find_nearest_passage(resources["collection"], search_question, document)
        index = resources["collection"].records.index(next(
            item for item in resources["collection"].records
            if item["document_name"] == document and item["chunk_text"] == passage
        ))
        scores = cosine_similarity(
            resources["collection"].vectorizer.transform([search_question]),
            resources["collection"].matrix,
        )[0]
        closest_distance = float(1 - scores[index])
        if closest_distance <= config.CHAT_MAX_RELEVANCE_DISTANCE:
            cap = code.get("daily_cap")
            separation = code.get("separation_minutes")
            answer_parts = [passage]
            if cap is not None:
                answer_parts.append(
                    f"The {category} code permits no more than {cap} spots of one campaign "
                    "on one channel in one broadcast day."
                )
            if separation is not None:
                answer_parts.append(
                    f"Those spots must be at least {separation} minutes apart."
                )
            answer = " ".join(answer_parts)
            source_documents = [document]
            was_refused = False
        else:
            was_refused = True
    with session_factory() as session:
        turn = ChatTurn(
            username=user["username"],
            user_role=user["role"],
            question=question,
            answer=answer,
            question_text=question,
            question_type=question_type if not was_refused else config.CHAT_TYPE_OUT_OF_SCOPE,
            answer_text=answer,
            source_documents=json.dumps(source_documents),
            closest_distance=closest_distance,
            was_refused=was_refused,
        )
        session.add(turn)
        session.commit()
    return {
        "question_type": question_type if not was_refused else config.CHAT_TYPE_OUT_OF_SCOPE,
        "campaign_reference": None,
        "answer": answer,
        "source_documents": source_documents,
        "closest_distance": closest_distance,
        "was_refused": was_refused,
    }
