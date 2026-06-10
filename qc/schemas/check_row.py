from dataclasses import dataclass
from typing import Literal, Optional


CheckStatus = Literal["PASS", "FAIL", "REVIEW", "SKIP"]

# Check levels — how often / from what evidence a check is judged:
#   "video"    : once per file, from metadata (container, fps, duration, ...)
#   "sequence" : judged from the whole frame timeline (turn sequence, holds)
#   "frame"    : judged per sampled frame (size, blur, brightness, eyes, ...)
CheckLevel = Literal["video", "sequence", "frame"]


@dataclass(frozen=True)
class CheckRow:
    volunteer_id: str
    data_type: str
    filename: str
    check_name: str
    status: CheckStatus
    reason: str
    frame_index: Optional[int] = None
    level: str = "frame"

    def as_tuple(self) -> tuple:
        # NOTE: `level` sits BEFORE check_name (report convention).
        return (
            self.volunteer_id,
            self.data_type,
            self.filename,
            self.level,
            self.check_name,
            self.status,
            self.reason,
            self.frame_index,
        )