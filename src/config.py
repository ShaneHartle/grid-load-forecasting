from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    timestamp_col: str = "Timestamp"
    target_col: str = "Load Demand (kW)"
    freq_minutes: int = 15               # dataset resolution
    daily_season_steps: int = 96         # 24h / 15-min
    test_days: int = 7
    random_state: int = 42
