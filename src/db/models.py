"""SQLAlchemy ORM models for the three database tables."""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Player IDs used in the stat.ink CSV
# ---------------------------------------------------------------------------
PLAYER_IDS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
PLAYER_STATS = ["weapon", "kill", "assist", "death", "special", "inked"]

# ---------------------------------------------------------------------------
# Table 1: raw battle records
# ---------------------------------------------------------------------------

class Battle(Base):
    __tablename__ = "battles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Match metadata
    period = Column(String, nullable=True)
    game_ver = Column(String, nullable=True)
    lobby = Column(String, nullable=True)
    mode = Column(String, nullable=True)
    stage = Column(String, nullable=True)
    win = Column(String, nullable=True)       # "alpha" or "bravo"
    knockout = Column(String, nullable=True)
    rank = Column(String, nullable=True)
    power = Column(Float, nullable=True)

    # Team ink totals
    alpha_inked = Column(Integer, nullable=True)
    alpha_ink_percent = Column(Float, nullable=True)
    bravo_inked = Column(Integer, nullable=True)
    bravo_ink_percent = Column(Float, nullable=True)

    # Per-player columns (A1-A4, B1-B4) — weapon + 5 stats each
    a1_weapon = Column(String, nullable=True)
    a1_kill = Column(Integer, nullable=True)
    a1_assist = Column(Integer, nullable=True)
    a1_death = Column(Integer, nullable=True)
    a1_special = Column(Integer, nullable=True)
    a1_inked = Column(Integer, nullable=True)

    a2_weapon = Column(String, nullable=True)
    a2_kill = Column(Integer, nullable=True)
    a2_assist = Column(Integer, nullable=True)
    a2_death = Column(Integer, nullable=True)
    a2_special = Column(Integer, nullable=True)
    a2_inked = Column(Integer, nullable=True)

    a3_weapon = Column(String, nullable=True)
    a3_kill = Column(Integer, nullable=True)
    a3_assist = Column(Integer, nullable=True)
    a3_death = Column(Integer, nullable=True)
    a3_special = Column(Integer, nullable=True)
    a3_inked = Column(Integer, nullable=True)

    a4_weapon = Column(String, nullable=True)
    a4_kill = Column(Integer, nullable=True)
    a4_assist = Column(Integer, nullable=True)
    a4_death = Column(Integer, nullable=True)
    a4_special = Column(Integer, nullable=True)
    a4_inked = Column(Integer, nullable=True)

    b1_weapon = Column(String, nullable=True)
    b1_kill = Column(Integer, nullable=True)
    b1_assist = Column(Integer, nullable=True)
    b1_death = Column(Integer, nullable=True)
    b1_special = Column(Integer, nullable=True)
    b1_inked = Column(Integer, nullable=True)

    b2_weapon = Column(String, nullable=True)
    b2_kill = Column(Integer, nullable=True)
    b2_assist = Column(Integer, nullable=True)
    b2_death = Column(Integer, nullable=True)
    b2_special = Column(Integer, nullable=True)
    b2_inked = Column(Integer, nullable=True)

    b3_weapon = Column(String, nullable=True)
    b3_kill = Column(Integer, nullable=True)
    b3_assist = Column(Integer, nullable=True)
    b3_death = Column(Integer, nullable=True)
    b3_special = Column(Integer, nullable=True)
    b3_inked = Column(Integer, nullable=True)

    b4_weapon = Column(String, nullable=True)
    b4_kill = Column(Integer, nullable=True)
    b4_assist = Column(Integer, nullable=True)
    b4_death = Column(Integer, nullable=True)
    b4_special = Column(Integer, nullable=True)
    b4_inked = Column(Integer, nullable=True)

    # Dedup guard: treat these four fields as a composite unique key
    __table_args__ = (
        UniqueConstraint(
            "period", "a1_weapon", "a1_kill", "a1_death",
            name="uq_battle_dedup",
        ),
    )

    feature_row = relationship("FeatureRow", back_populates="battle", uselist=False)


# ---------------------------------------------------------------------------
# Table 2: engineered feature rows (one-to-one with battles)
# ---------------------------------------------------------------------------

# Weapon classes used to generate column names
WEAPON_CLASSES = [
    "Shooter", "Blaster", "Roller", "Brush", "Charger",
    "Splatling", "Dualie", "Brella", "Slosher", "Stringer", "Splatana",
]

TEAM_STATS = ["kill", "death", "assist", "special", "inked"]


class FeatureRow(Base):
    __tablename__ = "feature_rows"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    battle_id = Column(BigInteger, ForeignKey("battles.id"), unique=True, nullable=False)
    target = Column(Integer, nullable=False)  # 1 = alpha wins, 0 = bravo wins

    # Team aggregate stats
    alpha_kill_sum = Column(Float)
    alpha_death_sum = Column(Float)
    alpha_assist_sum = Column(Float)
    alpha_special_sum = Column(Float)
    alpha_inked_sum = Column(Float)
    bravo_kill_sum = Column(Float)
    bravo_death_sum = Column(Float)
    bravo_assist_sum = Column(Float)
    bravo_special_sum = Column(Float)
    bravo_inked_sum = Column(Float)

    # Differentials
    kill_differential = Column(Float)
    death_differential = Column(Float)
    assist_differential = Column(Float)
    special_differential = Column(Float)
    inked_differential = Column(Float)

    # KD
    alpha_kd = Column(Float)
    bravo_kd = Column(Float)
    kd_ratio_differential = Column(Float)

    # Ink
    ink_differential = Column(Float)
    ink_percent_differential = Column(Float)

    # Matchmaking power
    power = Column(Float)

    # Weapon class counts per team (11 classes × 2 teams = 22 columns)
    alpha_shooter_count = Column(Float)
    alpha_blaster_count = Column(Float)
    alpha_roller_count = Column(Float)
    alpha_brush_count = Column(Float)
    alpha_charger_count = Column(Float)
    alpha_splatling_count = Column(Float)
    alpha_dualie_count = Column(Float)
    alpha_brella_count = Column(Float)
    alpha_slosher_count = Column(Float)
    alpha_stringer_count = Column(Float)
    alpha_splatana_count = Column(Float)

    bravo_shooter_count = Column(Float)
    bravo_blaster_count = Column(Float)
    bravo_roller_count = Column(Float)
    bravo_brush_count = Column(Float)
    bravo_charger_count = Column(Float)
    bravo_splatling_count = Column(Float)
    bravo_dualie_count = Column(Float)
    bravo_brella_count = Column(Float)
    bravo_slosher_count = Column(Float)
    bravo_stringer_count = Column(Float)
    bravo_splatana_count = Column(Float)

    # One-hot categoricals stored as Float (0.0 / 1.0)
    # Mode — populated dynamically during engineer step; stored as JSON here
    # to avoid a fixed schema for every possible mode/stage/lobby value.
    # The engineer module will expand these into real columns in the DataFrame.
    mode_onehot = Column(JSON, nullable=True)   # {"Turf War": 1, "Splat Zones": 0, ...}
    stage_onehot = Column(JSON, nullable=True)
    lobby_onehot = Column(JSON, nullable=True)

    battle = relationship("Battle", back_populates="feature_row")


# ---------------------------------------------------------------------------
# Table 3: prediction log (used by backend)
# ---------------------------------------------------------------------------

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    input_features = Column(JSON, nullable=False)   # feature dict submitted by user
    predicted_outcome = Column(Integer, nullable=False)  # 0 or 1
    confidence = Column(Float, nullable=False)           # probability of predicted class
    model_version = Column(String, nullable=True)        # e.g. "xgboost_v1"
