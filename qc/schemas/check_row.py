from dataclasses import dataclass
from typing import Literal, Optional


CheckStatus = Literal["PASS", "FAIL", "REVIEW", "SKIP"]


@dataclass(frozen=True)
class CheckRow:
    volunteer_id: str
    data_type: str
    filename: str
    check_name: str
    status: CheckStatus
    reason: str
    frame_index: Optional[int] = None

    def as_tuple(self) -> tuple:
        return (
            self.volunteer_id,
            self.data_type,
            self.filename,
            self.check_name,
            self.status,
            self.reason,
            self.frame_index,
        )