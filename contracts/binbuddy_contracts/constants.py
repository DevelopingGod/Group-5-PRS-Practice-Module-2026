"""Shared constants. Values marked PROVISIONAL must be confirmed by the named workstreams."""

from typing import Final

CONTRACTS_VERSION: Final = "0.1.0"

DEFAULT_SITE_POLICY: Final = "hdb_default"

# PROVISIONAL (WS1 + WS2 confirm by 8 Oct): dimension of the L2-normalised image embedding
# produced by the vision model. Also used by the Supabase pgvector column (WS4).
EMBEDDING_DIM: Final = 256

# Largest image the API accepts (decoded bytes).
MAX_IMAGE_BYTES: Final = 5_000_000

# PROVISIONAL (WS3 tunes on validation data): confidence at or above which no questions are asked.
CONFIDENCE_THRESHOLD: Final = 0.80

# PROVISIONAL (WS4): verification rules.
VERIFY_TIME_WINDOW_MINUTES: Final = 30
VERIFY_MIN_SIMILARITY: Final = 0.85
POINTS_PER_VERIFIED_DISPOSAL: Final = 1
DAILY_POINTS_CAP: Final = 10

# Tolerance used when checking that probability scores sum to 1.
PROBABILITY_SUM_TOLERANCE: Final = 1e-3
