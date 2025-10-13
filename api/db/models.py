from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    func,
    JSON,
    UniqueConstraint,
    Float,
)
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tmdb_id: Mapped[int] = mapped_column(
        Integer, unique=True, index=True, nullable=False
    )
    media_type: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # 'movie' or 'tv'
    title: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    overview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    runtime: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # minutes
    original_language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    genres: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    poster_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    release_year: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, index=True
    )
    popularity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vote_average: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vote_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    popular_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    trending_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    top_rated_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ItemEmbedding(Base):
    __tablename__ = "item_embeddings"
    __table_args__ = (
        UniqueConstraint("item_id", "version", name="uq_item_embedding_item_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(
        ForeignKey("items.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    vector: Mapped[List[float]] = mapped_column(Vector(384), nullable=False)


class Availability(Base):
    __tablename__ = "availability"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(
        ForeignKey("items.id", ondelete="CASCADE"), index=True
    )
    country: Mapped[str] = mapped_column(String(8), index=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    offer_type: Mapped[Optional[str]] = mapped_column(String(16))  # SVOD/TVOD/Free
    deeplink: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    web_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    last_checked: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    long_vec: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)
    short_vec: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)
    genre_prefs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    neighbors: Mapped[Optional[List[dict]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserHistory(Base):
    __tablename__ = "user_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    item_id: Mapped[int] = mapped_column(Integer, index=True)
    event_type: Mapped[str] = mapped_column(String(32))  # watched, liked, rated
    weight: Mapped[int] = mapped_column(Integer, default=1)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(128), index=True, nullable=True
    )
    item_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    type: Mapped[str] = mapped_column(String(32))  # impression, click, complete
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
