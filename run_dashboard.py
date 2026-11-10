import requests
import streamlit as st

from config import config


class DashboardAPIError(Exception):
    pass


def _request(api_url, path, auth=None, method="GET", body=None, params=None):
    headers = {}
    if auth:
        headers[config.SESSION_TOKEN_HEADER] = auth["session_token"]
    try:
        response = requests.request(
            method,
            f"{api_url.rstrip('/')}{path}",
            headers=headers,
            json=body,
            params=params,
            timeout=config.API_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise DashboardAPIError(
            f"Could not connect to the API at {api_url}. Check the URL and confirm "
            "the REST service is running."
        ) from error

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise DashboardAPIError(f"API error ({response.status_code}): {detail}")
    if not response.content:
        return None
    return response.json()


def _show_request_error(error):
    st.error(str(error))


def _metric_row(values):
    columns = st.columns(len(values))
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)


def _display_campaign_identity(campaign):
    reference = campaign.get("campaign_reference", "Campaign")
    advertiser = campaign.get(config.ADVERTISER_NAME_COLUMN) or "Advertiser not listed"
    agency = campaign.get(config.AGENCY_NAME_COLUMN) or "Agency not listed"
    st.subheader(f"{advertiser} · {reference}")
    st.caption(
        f"{agency} · {campaign.get(config.PRODUCT_CATEGORY_COLUMN, 'Category not listed')}"
    )


def _render_queue(api_url, auth):
    st.header(config.PAGE_QUEUE)
    try:
        rows = _request(api_url, "/queue", auth)
    except DashboardAPIError as error:
        _show_request_error(error)
        return

    if isinstance(rows, dict):
        rows = rows.get("campaigns", [])
    if not rows:
        st.info("There are no campaigns in the clearance queue.")
        return

    categories = sorted({
        str(row.get(config.PRODUCT_CATEGORY_COLUMN))
        for row in rows
        if row.get(config.PRODUCT_CATEGORY_COLUMN)
    })
    selected_category = st.selectbox("Product category", ["All categories", *categories])
    visible_rows = rows
    if selected_category != "All categories":
        visible_rows = [
            row for row in rows
            if row.get(config.PRODUCT_CATEGORY_COLUMN) == selected_category
        ]

    exposure_total = sum(float(row.get("exposure") or 0) for row in visible_rows)
    breach_total = sum(int(row.get("spots_breaching") or 0) for row in visible_rows)
    _metric_row([
        ("Campaigns", len(visible_rows)),
        ("Booked spots", sum(int(row.get("spots_booked") or 0) for row in visible_rows)),
        ("Potentially breaching spots", breach_total),
        ("Under-18 exposure (thousands)", f"{exposure_total:,.1f}"),
    ])
    st.caption(
        "Queue order is set by the service using forecast under-18 exposure. "
        "Model scores support review; they are not clearance decisions."
    )

    columns = [
        "queue_position",
        "campaign_reference",
        config.ADVERTISER_NAME_COLUMN,
        config.AGENCY_NAME_COLUMN,
        config.PRODUCT_CATEGORY_COLUMN,
        "spots_booked",
        "spots_breaching",
        "exposure",
        "amendment_band",
    ]
    available_columns = [column for column in columns if any(column in row for row in visible_rows)]
    st.dataframe(
        [{key: row.get(key) for key in available_columns} for row in visible_rows],
        use_container_width=True,
        hide_index=True,
    )


def _render_campaign_detail(api_url, auth, areas):
    st.header(config.PAGE_CAMPAIGN)
    try:
        queue_rows = _request(api_url, "/queue", auth)
    except DashboardAPIError as error:
        _show_request_error(error)
        return
    if isinstance(queue_rows, dict):
        queue_rows = queue_rows.get("campaigns", [])
    if not queue_rows:
        st.info("There are no campaigns to review.")
        return

    by_reference = {
        str(row["campaign_reference"]): row
        for row in queue_rows
        if row.get("campaign_reference")
    }
    references = list(by_reference)
    reference = st.selectbox(
        "Campaign",
        references,
        format_func=lambda value: (
            f"{value} · {by_reference[value].get(config.ADVERTISER_NAME_COLUMN, 'Unknown advertiser')}"
        ),
    )
    try:
        detail = _request(api_url, f"/queue/{reference}", auth)
    except DashboardAPIError as error:
        _show_request_error(error)
        return

    _display_campaign_identity(detail)
    _metric_row([
        ("Booked spots", detail.get("spots_booked", 0)),
        ("Compliant", detail.get("spots_compliant", 0)),
        ("Breaching", detail.get("spots_breaching", 0)),
        ("Under-18 exposure (thousands)", f"{float(detail.get('exposure') or 0):,.1f}"),
    ])

    overview, spots_tab, notes_tab, rules_tab = st.tabs(
        ["Assessment overview", "Booked spots", "Campaign notes", "Code rules"]
    )
    with overview:
        st.write("**Amendment risk**")
        st.write(
            f"{detail.get('amendment_band', detail.get('amendment', {}).get('amendment_band', 'Not available'))}"
            f" · score {detail.get('amendment_score', detail.get('amendment', {}).get('amendment_score', 'Not available'))}"
        )
        st.write("**Issue review**")
        st.json(detail.get("issue", {}))
        st.write("**First transmission**")
        st.json(detail.get("first_transmission", detail.get("summary", {}).get("first_transmission", {})))
        st.write("**Breaches by rule**")
        st.json(detail.get("breaches_by_rule", detail.get("summary", {}).get("breaches_by_rule", {})))
    with spots_tab:
        results = detail.get("spot_results", [])
        if results:
            st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.info("No spot-level results are available.")
        with st.expander("Placement and programme details"):
            st.json(detail.get("placements", []))
    with notes_tab:
        notes = detail.get("notes", [])
        if notes:
            for note in notes:
                with st.expander(str(note.get("note_id") or "Campaign note")):
                    st.write(note.get(config.NOTE_TEXT_COLUMN, note))
        else:
            st.info("No campaign notes are available.")
    with rules_tab:
        st.write(f"**Source document:** {detail.get('code_rules', {}).get('document_name', 'Not available')}")
        st.json(detail.get("code_rules", {}))

    if config.AREA_ASSESSMENT not in areas:
        return
    st.divider()
    st.subheader("Start an assessment")
    st.warning(
        "An assessment is a recommendation for a separate Clearance Controller. "
        "It does not authorize transmission."
    )
    with st.form("assessment_form"):
        executive_request = st.text_area(
            "Assessment request",
            placeholder="Describe what should be reviewed for this campaign.",
            help="The request must contain at least "
            f"{config.EXECUTIVE_REQUEST_MINIMUM_CHARACTERS} characters.",
        )
        submitted = st.form_submit_button("Run assessment", type="primary")
    if submitted:
        try:
            assessment = _request(
                api_url,
                "/assessments",
                auth,
                method="POST",
                body={
                    "campaign_reference": reference,
                    "executive_request": executive_request,
                },
            )
        except DashboardAPIError as error:
            _show_request_error(error)
        else:
            st.success("Assessment saved and queued for an independent controller decision.")
            st.json(assessment)


def _render_decisions(api_url, auth):
    st.header(config.PAGE_DECISION)
    try:
        assessments = _request(
            api_url,
            "/clearances",
            auth,
            params={"status_value": config.CAMPAIGN_STATUS_PENDING},
        )
    except DashboardAPIError as error:
        _show_request_error(error)
        return
    if not assessments:
        st.info("There are no assessments awaiting a controller decision.")
        return

    by_id = {item["assessment_id"]: item for item in assessments}
    assessment_id = st.selectbox(
        "Pending assessment",
        list(by_id),
        format_func=lambda value: (
            f"{by_id[value].get('campaign_reference', 'Campaign')} · {value}"
        ),
    )
    assessment = by_id[assessment_id]
    _display_campaign_identity({
        "campaign_reference": assessment.get("campaign_reference"),
    })
    _metric_row([
        ("Booked spots", assessment.get("spots_booked", 0)),
        ("Compliant", assessment.get("spots_compliant", 0)),
        ("Breaching", assessment.get("spots_breaching", 0)),
        ("Recommended decision", assessment.get("recommended_decision", "Not available")),
    ])
    st.write(assessment.get("assessment_text", "No assessment summary is available."))
    with st.expander("Assessment evidence", expanded=True):
        st.write(f"**Assessment plan:** {assessment.get('assessment_plan_text', 'Not available')}")
        st.write(f"**Category-code document:** {assessment.get('code_document', 'Not available')}")
        st.write(f"**Amendment risk:** {assessment.get('amendment_band', 'Not available')}")
        st.write(f"**Issue category:** {assessment.get('issue_category', 'Not available')}")
        st.write(f"**Breaches by rule:**")
        st.json(assessment.get("breaches_by_rule", {}))
        st.write("**Breaching placements:**")
        st.dataframe(assessment.get("breaching_spots", []), use_container_width=True, hide_index=True)

    open_decisions = assessment.get("decisions_open", [])
    if not open_decisions:
        st.error("This assessment has no valid open decision options.")
        return
    with st.form(f"decision_form_{assessment_id}"):
        decision = st.selectbox("Controller decision", open_decisions)
        reason = st.text_area(
            "Decision rationale",
            help=f"Enter at least {config.DECISION_REASON_MINIMUM_CHARACTERS} characters.",
        )
        confirmed = st.checkbox(
            "I am acting as the independent controller; this is not my own assessment."
        )
        submitted = st.form_submit_button("Record controller decision", type="primary")
    if submitted:
        if not confirmed:
            st.error("Confirm the independent controller check before submitting.")
            return
        try:
            result = _request(
                api_url,
                f"/clearances/{assessment_id}/decision",
                auth,
                method="POST",
                body={"decision": decision, "decision_reason": reason},
            )
        except DashboardAPIError as error:
            _show_request_error(error)
        else:
            st.success(f"Decision recorded: {result.get('status', decision)}")
            for label, key in [
                ("Clearance certificate", "clearance_certificate_text"),
                ("Restriction notice", "restriction_notice_text"),
            ]:
                if result.get(key):
                    st.info(f"**{label}**\n\n{result[key]}")


def _render_audit(api_url, auth):
    st.header(config.PAGE_AUDIT)
    try:
        assessments = _request(api_url, "/clearances", auth)
        chat_turns = _request(api_url, "/audit/chat-turns", auth)
    except DashboardAPIError as error:
        _show_request_error(error)
        return
    st.subheader("Clearance assessments")
    if not assessments:
        st.info("No assessments have been recorded.")
    else:
        st.dataframe(
            [{
                "assessment_id": item.get("assessment_id"),
                "campaign_reference": item.get("campaign_reference"),
                "status": item.get("status"),
                "recommended_decision": item.get("recommended_decision"),
                "decision": item.get("decision"),
                "decided_by": item.get("decided_by"),
                "status_changed_at": item.get("status_changed_at"),
            } for item in assessments],
            use_container_width=True,
            hide_index=True,
        )
        by_id = {item["assessment_id"]: item for item in assessments}
        selected_id = st.selectbox(
            "Assessment audit detail",
            list(by_id),
            format_func=lambda value: (
                f"{by_id[value].get('campaign_reference', 'Campaign')} · {value}"
            ),
        )
        selected = by_id[selected_id]
        with st.expander("Assessment record"):
            st.json(selected)
        try:
            call_audit = _request(api_url, f"/clearances/{selected_id}/calls", auth)
        except DashboardAPIError as error:
            _show_request_error(error)
        else:
            st.write(
                f"Recorded pipeline events: {call_audit.get('call_count', 0)} "
                f"of {call_audit.get('expected_call_count', 0)}"
            )
            st.dataframe(call_audit.get("calls", []), use_container_width=True, hide_index=True)

    st.subheader("Chat audit")
    if chat_turns:
        st.dataframe(chat_turns, use_container_width=True, hide_index=True)
    else:
        st.info("No chat turns have been recorded.")


def _render_chat(api_url, auth):
    st.header(config.PAGE_CHAT)
    st.caption(
        "Ask about campaign status or clearance guidance. Responses are informational and "
        "do not replace policy review or controller approval."
    )
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Your question")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        try:
            answer = _request(
                api_url,
                "/chat/ask",
                auth,
                method="POST",
                body={"question": question},
            )
        except DashboardAPIError as error:
            _show_request_error(error)
        else:
            st.markdown(f"**{answer.get('question_type', 'Response')}**")
            st.write(answer.get("answer", "No answer was returned."))
            if answer.get("source_documents"):
                st.caption("Sources: " + ", ".join(answer["source_documents"]))
            if answer.get("was_refused"):
                st.warning("The service could not verify this question from the available sources.")

    try:
        turns = _request(api_url, "/chat/turns", auth)
    except DashboardAPIError as error:
        _show_request_error(error)
        return
    if turns:
        st.subheader("Your recent conversation")
        for turn in reversed(turns):
            with st.chat_message("user"):
                st.write(turn.get("question_text", ""))
            with st.chat_message("assistant"):
                st.write(turn.get("answer_text", ""))
                if turn.get("source_documents"):
                    st.caption("Sources: " + ", ".join(turn["source_documents"]))


def _render_login(api_url):
    st.info("Sign in to open the role-scoped clearance workspace.")
    with st.form("signin_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        try:
            auth = _request(
                api_url,
                "/auth/signin",
                method="POST",
                body={"username": username, "password": password},
            )
        except DashboardAPIError as error:
            _show_request_error(error)
        else:
            st.session_state["auth"] = auth
            st.rerun()


def main():
    st.set_page_config(
        page_title=config.DASHBOARD_TITLE,
        page_icon=":material/fact_check:",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(20, 67, 82, .10), rgba(20, 67, 82, .03));
            border: 1px solid rgba(20, 67, 82, .12);
            padding: 1rem;
            border-radius: .75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(config.DASHBOARD_TITLE)
    st.caption(config.DASHBOARD_SUBTITLE)

    api_url = st.sidebar.text_input("API URL", config.API_BASE_URL, key="api_url")
    auth = st.session_state.get("auth")
    if not auth:
        _render_login(api_url)
        return

    st.sidebar.success(f"{auth['display_name']} · {auth['role']}")
    if st.sidebar.button("Sign out", use_container_width=True):
        try:
            _request(api_url, "/auth/signout", auth, method="POST")
        except DashboardAPIError as error:
            _show_request_error(error)
        else:
            st.session_state.pop("auth", None)
            st.rerun()

    pages = auth.get("pages", [])
    if not pages:
        st.error("This account has no dashboard pages assigned.")
        return
    selected_page = st.sidebar.radio("Workspace", pages)
    areas = set(auth.get("areas", []))
    if selected_page == config.PAGE_QUEUE:
        _render_queue(api_url, auth)
    elif selected_page == config.PAGE_CAMPAIGN:
        _render_campaign_detail(api_url, auth, areas)
    elif selected_page == config.PAGE_DECISION:
        _render_decisions(api_url, auth)
    elif selected_page == config.PAGE_AUDIT:
        _render_audit(api_url, auth)
    elif selected_page == config.PAGE_CHAT:
        _render_chat(api_url, auth)
    else:
        st.error("The selected dashboard page is not supported.")


if __name__ == "__main__":
    main()
