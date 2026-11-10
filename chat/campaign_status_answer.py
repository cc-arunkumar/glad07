import json
from datetime import datetime

from config import config
from persistence.models import ClearanceAssessment


def _format_changed(value):
    if not value:
        value=datetime.utcnow().isoformat()
    parsed=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if parsed.tzinfo is None:
        parsed=parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.strftime(config.STATUS_CHANGE_TIME_FORMAT)


def answer_campaign_status(session, campaigns, user, campaign_reference):
    matches=campaigns[campaigns[config.CAMPAIGN_REFERENCE_COLUMN].astype(str)
                     ==str(campaign_reference)]
    if matches.empty:
        return config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE

    profile=matches.iloc[0]
    if user.get("role")==config.ROLE_LIAISON:
        agency=user.get("assigned_agencies",[])
        if str(profile.get(config.AGENCY_CODE_COLUMN)) not in agency:
            return config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE

    record=(session.query(ClearanceAssessment)
            .filter_by(campaign_reference=campaign_reference)
            .order_by(ClearanceAssessment.changed_at.desc()).first())
    if record is None:
        return config.CHAT_NOT_ASSESSED_TEMPLATE.format(
            campaign_reference=campaign_reference
        )

    state=json.loads(record.assessment_json or "{}")
    changed_at=_format_changed(state.get("status_changed_at") or record.changed_at)
    status_line=config.CHAT_STATUS_TEMPLATE.format(
        campaign_reference=campaign_reference,
        status_phrase=config.CHAT_STATUS_PHRASES[record.status],
        changed_at=changed_at,
    )
    if user.get("role")==config.ROLE_LIAISON:
        if record.status==config.CAMPAIGN_STATUS_PENDING:
            return status_line
        return " ".join([
            status_line,
            config.CHAT_DECIDED_TEMPLATE.format(
                decision=record.decision,
                spots_cleared=state.get("spots_cleared_count",0),
                spots_withdrawn=state.get("spots_withdrawn_count",0),
            ),
            config.CHAT_DOCUMENTS_SENT[record.status],
        ])

    if record.status==config.CAMPAIGN_STATUS_PENDING:
        return status_line
    details=[
        status_line,
        f"Assessment {record.assessment_id}; opened by {state.get('executive_id')}.",
        f"Amendment score {state.get('amendment_score')}; band {state.get('amendment_band')}.",
        f"Issue category: {state.get('issue_category')}.",
        f"Decision by {record.decided_by}: {record.decision}.",
        f"Reason: {record.decision_reason}.",
        f"Spots cleared: {', '.join(state.get('spots_cleared') or [])}.",
        f"Spots withdrawn: {', '.join(state.get('spots_withdrawn') or [])}.",
    ]
    return " ".join(details)
