import json

from persistence.database import session_factory
from persistence.models import ChatTurn
from config import config


def recall_conversation(session_or_username, username=None, question=None, limit=None):
    if hasattr(session_or_username, "query"):
        session = session_or_username
        account_name = username
        follow_up = question or ""
        close = False
    else:
        session = session_factory()
        account_name = session_or_username
        follow_up = ""
        if isinstance(username, int):
            limit = username
        close = True
    try:
        turns = (session.query(ChatTurn).filter_by(username=account_name)
                 .order_by(ChatTurn.id.desc()).limit(limit or config.CHAT_HISTORY_TURN_LIMIT)
                 .all())
        turns.reverse()
        history_lines = []
        search_lines = [follow_up] if follow_up else []
        for turn in turns:
            question_text = turn.question_text or turn.question or ""
            if turn.question_type == config.CHAT_TYPE_CAMPAIGN_STATUS:
                history_lines.append(config.CHAT_STATUS_ANSWER_IN_HISTORY)
                search_lines.append(config.CHAT_STATUS_ANSWER_IN_HISTORY)
                continue
            history_lines.append(question_text)
            if turn.answer_text or turn.answer:
                history_lines.append(turn.answer_text or turn.answer)
            search_lines.append(question_text)
            if turn.answer_text or turn.answer:
                search_lines.append(turn.answer_text or turn.answer)
        return {
            "history_text": "\n".join(line for line in history_lines if line),
            "search_text": "\n".join(line for line in search_lines if line),
            "turns": turns,
        }
    finally:
        if close:
            session.close()
