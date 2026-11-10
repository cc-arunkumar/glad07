from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UserAccount(Base):
    __tablename__ = "user_accounts"

    username: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    password_hash: Mapped[str] = mapped_column(String)
    password_salt: Mapped[str] = mapped_column(String, default="")
    assigned_agencies: Mapped[str] = mapped_column(Text, default="[]")
    session_token: Mapped[str | None] = mapped_column(String, nullable=True)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    campaign_reference: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String)
    started_by: Mapped[str] = mapped_column(String)
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    thread_id: Mapped[str] = mapped_column(String, default="")
    status_final: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )


class ClearanceAssessment(Base):
    __tablename__ = "clearance_assessments"

    assessment_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String)
    campaign_reference: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String)
    assessment_json: Mapped[str] = mapped_column(Text)
    decided_by: Mapped[str | None] = mapped_column(String, nullable=True)
    decision: Mapped[str | None] = mapped_column(String, nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    changed_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )


class ModelCall(Base):
    __tablename__ = "model_calls"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    agent_name: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String)
    prompt_characters: Mapped[int] = mapped_column()
    response_characters: Mapped[int] = mapped_column()
    latency_milliseconds: Mapped[int] = mapped_column()
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, index=True)
    user_role: Mapped[str | None] = mapped_column(String, nullable=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    question_text: Mapped[str] = mapped_column(Text, default="")
    question_type: Mapped[str] = mapped_column(String, default="")
    campaign_reference: Mapped[str | None] = mapped_column(String, nullable=True)
    answer_text: Mapped[str] = mapped_column(Text, default="")
    source_documents: Mapped[str] = mapped_column(Text, default="[]")
    closest_distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    was_refused: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )
