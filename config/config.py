import math

RAW_SUBMISSIONS_TRAIN_FILE = "data/campaign_submissions_train.csv"
RAW_SUBMISSIONS_EVAL_FILE = "data/campaign_submissions_eval.csv"
RAW_AUDIENCES_TRAIN_FILE = "data/programme_audiences_train.csv"
RAW_AUDIENCES_EVAL_FILE = "data/programme_audiences_eval.csv"
RAW_CLEARANCE_NOTES_FILE = "data/clearance_notes.csv"

CLEAN_SUBMISSIONS_TRAIN_FILE = "processed/campaign_submissions_train_clean.csv"
CLEAN_SUBMISSIONS_EVAL_FILE = "processed/campaign_submissions_eval_clean.csv"
CLEAN_AUDIENCES_TRAIN_FILE = "processed/programme_audiences_train_clean.csv"
CLEAN_AUDIENCES_EVAL_FILE = "processed/programme_audiences_eval_clean.csv"
CLEAN_CLEARANCE_NOTES_FILE = "processed/clearance_notes_clean.csv"

DATA_QUALITY_REPORT_FILE = "output/data_quality_report.json"
TARGET_SHARE_DECIMALS = 4


NULL_TOKENS = ["", "-", "n/a", "null", "?", "not recorded"]


BOOLEAN_YES = "Yes"
BOOLEAN_NO = "No"

BOOLEAN_MAPPING = {
    "yes": BOOLEAN_YES,
    "y": BOOLEAN_YES,
    "true": BOOLEAN_YES,
    "1": BOOLEAN_YES,
    "no": BOOLEAN_NO,
    "n": BOOLEAN_NO,
    "false": BOOLEAN_NO,
    "0": BOOLEAN_NO,
}


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PREVIOUS_WEEKDAY = {weekday: WEEKDAYS[position - 1] for position, weekday in enumerate(WEEKDAYS)}

WEEKDAY_MAPPING = {
    "monday": "Monday",
    "mon": "Monday",
    "mo.": "Monday",
    "tuesday": "Tuesday",
    "tue": "Tuesday",
    "tu.": "Tuesday",
    "tues": "Tuesday",
    "wednesday": "Wednesday",
    "wed": "Wednesday",
    "we.": "Wednesday",
    "thursday": "Thursday",
    "thu": "Thursday",
    "th.": "Thursday",
    "thurs": "Thursday",
    "friday": "Friday",
    "fri": "Friday",
    "fr.": "Friday",
    "saturday": "Saturday",
    "sat": "Saturday",
    "sa.": "Saturday",
    "sunday": "Sunday",
    "sun": "Sunday",
    "su.": "Sunday",
}


SUBMISSION_CHANNEL_MAPPING = {
    "clearance portal": "Clearance portal",
    "portal": "Clearance portal",
    "email": "Email",
    "e-mail": "Email",
    "agency api": "Agency API",
    "api": "Agency API",
}

PRODUCT_CATEGORY_MAPPING = {
    "alcohol": "Alcohol",
    "alc": "Alcohol",
    "alcoholic drinks": "Alcohol",
    "betting and gaming": "Betting and gaming",
    "b&g": "Betting and gaming",
    "gambling": "Betting and gaming",
    "betting & gaming": "Betting and gaming",
    "less healthy food": "Less healthy food",
    "hfss": "Less healthy food",
    "lhf": "Less healthy food",
    "consumer credit": "Consumer credit",
    "credit": "Consumer credit",
    "medicines and health": "Medicines and health",
    "otc medicines": "Medicines and health",
    "medicines & health": "Medicines and health",
    "health": "Medicines and health",
}

SUB_CATEGORY_MAPPING = {
    "beer and lager": "Beer and lager",
    "cider": "Cider",
    "wine": "Wine",
    "spirits": "Spirits",
    "ready-to-drink cocktails": "Ready-to-drink cocktails",
    "sports betting": "Sports betting",
    "online casino": "Online casino",
    "bingo": "Bingo",
    "lottery": "Lottery",
    "poker": "Poker",
    "confectionery": "Confectionery",
    "soft drinks": "Soft drinks",
    "fast food": "Fast food",
    "savoury snacks": "Savoury snacks",
    "breakfast cereals": "Breakfast cereals",
    "ice cream": "Ice cream",
    "credit cards": "Credit cards",
    "personal loans": "Personal loans",
    "buy now pay later": "Buy now pay later",
    "car finance": "Car finance",
    "short-term loans": "Short-term loans",
    "pain relief": "Pain relief",
    "cold and flu": "Cold and flu",
    "allergy relief": "Allergy relief",
    "vitamins and supplements": "Vitamins and supplements",
    "digestive health": "Digestive health",
    "sleep aids": "Sleep aids",
}

PERFORMER_AGE_BAND_MAPPING = {
    "no performer": "No performer",
    "no performer shown": "No performer",
    "under 18": "Under 18",
    "u18": "Under 18",
    "under-18": "Under 18",
    "18 to 24": "18 to 24",
    "18-24": "18 to 24",
    "18 - 24": "18 to 24",
    "18 to 24 years": "18 to 24",
    "25 to 34": "25 to 34",
    "25-34": "25 to 34",
    "25 - 34": "25 to 34",
    "25 to 34 years": "25 to 34",
    "35 to 49": "35 to 49",
    "35-49": "35 to 49",
    "35 - 49": "35 to 49",
    "35 to 49 years": "35 to 49",
    "50 and over": "50 and over",
    "50+": "50 and over",
    "50 plus": "50 and over",
    "over 50": "50 and over",
}

CHANNEL_MAPPING = {
    "harbourlight one": "Harbourlight One",
    "hl1": "Harbourlight One",
    "h-one": "Harbourlight One",
    "harbourlight 1": "Harbourlight One",
    "harbourlight sport": "Harbourlight Sport",
    "hls": "Harbourlight Sport",
    "h-sport": "Harbourlight Sport",
    "harbourlight drama": "Harbourlight Drama",
    "hld": "Harbourlight Drama",
    "h-drama": "Harbourlight Drama",
    "harbourlight life": "Harbourlight Life",
    "hll": "Harbourlight Life",
    "h-life": "Harbourlight Life",
}

SCHEDULING_SYSTEM_MAPPING = {
    "airtime planner": "Airtime Planner",
    "txline": "TXLine",
    "slotmaster": "SlotMaster",
    "planit": "Planit",
}

GENRE_MAPPING = {
    "children's": "Children's",
    "childrens": "Children's",
    "kids": "Children's",
    "animation": "Animation",
    "animated": "Animation",
    "family entertainment": "Family entertainment",
    "entertainment": "Family entertainment",
    "reality": "Reality",
    "reality tv": "Reality",
    "soap": "Soap",
    "soap opera": "Soap",
    "soaps": "Soap",
    "drama": "Drama",
    "drama series": "Drama",
    "crime drama": "Crime drama",
    "crime": "Crime drama",
    "comedy": "Comedy",
    "sitcom": "Comedy",
    "film": "Film",
    "feature film": "Film",
    "movie": "Film",
    "documentary": "Documentary",
    "doc": "Documentary",
    "factual": "Documentary",
    "lifestyle": "Lifestyle",
    "life style": "Lifestyle",
    "quiz and game show": "Quiz and game show",
    "quiz": "Quiz and game show",
    "game show": "Quiz and game show",
    "quiz & game show": "Quiz and game show",
    "news": "News",
    "news bulletin": "News",
    "current affairs": "Current affairs",
    "sport": "Sport",
    "sports": "Sport",
    "football": "Football",
    "soccer": "Football",
    "talk and magazine": "Talk and magazine",
    "talk show": "Talk and magazine",
    "magazine": "Talk and magazine",
    "talk & magazine": "Talk and magazine",
}

DAY_TYPE_MAPPING = {
    "weekday": "Weekday",
    "weekend": "Weekend",
    "bank holiday": "Bank holiday",
}

AUTHOR_ROLE_MAPPING = {
    "clearance executive": "Clearance Executive",
    "clearance exec": "Clearance Executive",
    "senior clearance executive": "Senior Clearance Executive",
    "senior clearance exec": "Senior Clearance Executive",
    "legal adviser": "Legal Adviser",
    "legal advisor": "Legal Adviser",
    "agency contact": "Agency Contact",
    "agency": "Agency Contact",
    "clearance controller": "Clearance Controller",
    "controller": "Clearance Controller",
}

NOTE_SOURCE_MAPPING = {
    "executive first read": "Executive first read",
    "exec first read": "Executive first read",
    "legal review": "Legal review",
    "legal": "Legal review",
    "agency reply": "Agency reply",
    "agency response": "Agency reply",
    "executive update": "Executive update",
    "exec update": "Executive update",
    "controller comment": "Controller comment",
    "controller note": "Controller comment",
}

ISSUE_CATEGORY_MAPPING = {
    "health claim": "Health claim",
    "health claims": "Health claim",
    "price or offer claim": "Price or offer claim",
    "price/offer claim": "Price or offer claim",
    "pricing claim": "Price or offer claim",
    "environmental claim": "Environmental claim",
    "green claim": "Environmental claim",
    "environmental": "Environmental claim",
    "comparative claim": "Comparative claim",
    "comparative": "Comparative claim",
    "comparison claim": "Comparative claim",
    "appeal to under-18s": "Appeal to under-18s",
    "under-18 appeal": "Appeal to under-18s",
    "appeal to under 18s": "Appeal to under-18s",
    "gambling inducement": "Gambling inducement",
    "gambling inducements": "Gambling inducement",
    "inducement to gamble": "Gambling inducement",
    "alcohol and social success": "Alcohol and social success",
    "alcohol social success": "Alcohol and social success",
    "social success": "Alcohol and social success",
    "financial promotion": "Financial promotion",
    "financial promotions": "Financial promotion",
    "credit promotion": "Financial promotion",
}


BROADCAST_CLOCK_SYSTEMS = ["TXLine"]
CALENDAR_CLOCK_SYSTEMS = ["Airtime Planner", "SlotMaster", "Planit"]
BROADCAST_DAY_START_MINUTE = 360
BROADCAST_DAY_END_MINUTE = 1800
MINUTES_PER_DAY = 1440
MINUTES_PER_HOUR = 60
CLOCK_TIME_PATTERN = r"^(\d{2}):([0-5]\d)$"
CLOCK_TIME_DIGITS = 2


STANDARD_SPOT_LENGTHS = [10, 20, 30, 40, 60, 90]
SECONDS_NOTATION_PATTERN = r'^(\d+)(?:"|s| sec)?$'
MINUTES_AND_SECONDS_NOTATION_PATTERN = r"^(\d+):([0-5]\d)$"
HOURS_MINUTES_AND_SECONDS_NOTATION_PATTERN = r"^(\d{2}):([0-5]\d):([0-5]\d)$"
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
CLOCK_NUMBER_PATTERN = r"^[A-Z]{3}/[A-Z]{4}\d{3}/(\d{3})$"


WHOLE_NUMBER_PATTERN = r"^(-?\d+)$"
PLAIN_NUMBER_PATTERN = r"^-?\d+(?:\.\d+)?$"
WHOLE_PERCENTAGE_PATTERN = r"^(-?\d+)(?: ?%)?$"
DECIMAL_PERCENTAGE_PATTERN = r"^(-?\d+(?:\.\d+)?)(?: ?%)?$"
SPOTS_COUNT_PATTERN = r"^(-?\d+)(?: spots?)?$"
CHANNELS_COUNT_PATTERN = r"^(-?\d+)(?: channels?)?$"

MILLIONS_NOTATION_PATTERN = r"^(-?\d+(?:\.\d+)?)m$"
THOUSANDS_NOTATION_PATTERN = r"^(-?\d+(?:\.\d+)?)(?:k| thousand)?$"
VIEWERS_NOTATION_PATTERN = r"^(-?\d{1,3}(?:,\d{3})+)$"
DIGIT_GROUPING_SEPARATOR = ","
THOUSANDS_PER_MILLION = 1000
VIEWERS_PER_THOUSAND = 1000
AUDIENCE_DECIMALS = 1


MARKUP_PATTERN = r"<[^>]+>"
ENCODING_ARTIFACT_PATTERN = r"&[#a-z0-9]+;|\xa0"
NO_BREAK_SPACE = "\xa0"
WHITESPACE_RUN_PATTERN = r"\s+"
MINIMUM_NOTE_WORDS = 6
NOTE_TIMESTAMP_PATTERN = (r"^(?P<note_weekday>\S+) "
                          r"(?P<note_time>(?:[01]\d|2[0-3]):[0-5]\d)$")
NOTE_WEEKDAY_PART = "note_weekday"
NOTE_TIME_PART = "note_time"


SUBMISSION_TARGET_COLUMN = "amendment_required"
CLOCK_NUMBER_COLUMN = "clock_number"
COPY_LENGTH_COLUMN = "copy_length_seconds"
LEGAL_TEXT_SECONDS_COLUMN = "legal_text_seconds"
ADVERTISER_CAMPAIGNS_COLUMN = "advertiser_campaigns_past_year"
ADVERTISER_AMENDMENTS_COLUMN = "advertiser_amendments_past_year"

SUBMISSION_VALUE_MAPPINGS = {
    "submitted_via": SUBMISSION_CHANNEL_MAPPING,
    "product_category": PRODUCT_CATEGORY_MAPPING,
    "sub_category": SUB_CATEGORY_MAPPING,
    "lead_performer_age_band": PERFORMER_AGE_BAND_MAPPING,
    "weekday_submitted": WEEKDAY_MAPPING,
    "voiceover_flag": BOOLEAN_MAPPING,
    "celebrity_flag": BOOLEAN_MAPPING,
    "sports_personality_flag": BOOLEAN_MAPPING,
    "humour_flag": BOOLEAN_MAPPING,
    "price_claim_flag": BOOLEAN_MAPPING,
    "health_claim_flag": BOOLEAN_MAPPING,
    "environmental_claim_flag": BOOLEAN_MAPPING,
    "comparative_claim_flag": BOOLEAN_MAPPING,
    "offer_terms_shown_flag": BOOLEAN_MAPPING,
    "bonus_or_free_bet_flag": BOOLEAN_MAPPING,
    "representative_apr_shown_flag": BOOLEAN_MAPPING,
    "new_advertiser_flag": BOOLEAN_MAPPING,
    "amendment_required": BOOLEAN_MAPPING,
}

SUBMISSION_WHOLE_NUMBER_PATTERNS = {
    "supers_count": WHOLE_NUMBER_PATTERN,
    "legal_text_seconds": WHOLE_NUMBER_PATTERN,
    "child_appeal_elements_count": WHOLE_NUMBER_PATTERN,
    "claims_count": WHOLE_NUMBER_PATTERN,
    "spots_booked": SPOTS_COUNT_PATTERN,
    "channels_booked": CHANNELS_COUNT_PATTERN,
    "share_before_21_pct": WHOLE_PERCENTAGE_PATTERN,
    "share_in_live_sport_pct": WHOLE_PERCENTAGE_PATTERN,
    "days_to_first_transmission": WHOLE_NUMBER_PATTERN,
    "advertiser_campaigns_past_year": WHOLE_NUMBER_PATTERN,
    "advertiser_amendments_past_year": WHOLE_NUMBER_PATTERN,
}

SUBMISSION_DECIMAL_NUMBER_PATTERNS = {
    "agency_amendment_rate_pct": DECIMAL_PERCENTAGE_PATTERN,
}

SUBMISSION_RANGE_LIMITS = {
    "supers_count": (0, math.inf),
    "legal_text_seconds": (0, math.inf),
    "child_appeal_elements_count": (0, math.inf),
    "claims_count": (0, math.inf),
    "spots_booked": (1, math.inf),
    "channels_booked": (1, 4),
    "share_before_21_pct": (0, 100),
    "share_in_live_sport_pct": (0, 100),
    "days_to_first_transmission": (0, math.inf),
    "advertiser_campaigns_past_year": (0, math.inf),
    "advertiser_amendments_past_year": (0, math.inf),
    "agency_amendment_rate_pct": (0, 100),
}


AUDIENCE_TARGET_COLUMN = "under18_viewers"
TRANSMISSION_DAY_COLUMN = "transmission_day"
START_TIME_COLUMN = "start_time"
END_TIME_COLUMN = "end_time"
SCHEDULING_SYSTEM_COLUMN = "scheduling_system"

AUDIENCE_VALUE_MAPPINGS = {
    "channel": CHANNEL_MAPPING,
    "scheduling_system": SCHEDULING_SYSTEM_MAPPING,
    "transmission_day": WEEKDAY_MAPPING,
    "genre": GENRE_MAPPING,
    "lead_in_genre": GENRE_MAPPING,
    "day_type": DAY_TYPE_MAPPING,
    "live_flag": BOOLEAN_MAPPING,
    "live_sport_flag": BOOLEAN_MAPPING,
    "made_for_children_flag": BOOLEAN_MAPPING,
    "premiere_flag": BOOLEAN_MAPPING,
    "school_holiday_flag": BOOLEAN_MAPPING,
}

AUDIENCE_COLUMNS_IN_THOUSANDS = ["audience_all", "under18_viewers"]

AUDIENCE_WHOLE_NUMBER_PATTERNS = {
    "episode_number": WHOLE_NUMBER_PATTERN,
    "forecast_under18_index": WHOLE_NUMBER_PATTERN,
}

AUDIENCE_RANGE_LIMITS = {
    "episode_number": (1, math.inf),
    "forecast_under18_index": (1, 999),
    "audience_all": (0, math.inf),
    "under18_viewers": (0, math.inf),
    "end_time": (BROADCAST_DAY_START_MINUTE, BROADCAST_DAY_END_MINUTE),
}


NOTE_TARGET_COLUMN = "issue_category"
NOTE_TEXT_COLUMN = "note_text"
NOTE_LENGTH_COLUMN = "note_length_chars"
NOTE_TIMESTAMP_COLUMN = "note_written_at"

NOTE_VALUE_MAPPINGS = {
    "author_role": AUTHOR_ROLE_MAPPING,
    "note_source": NOTE_SOURCE_MAPPING,
    "product_category": PRODUCT_CATEGORY_MAPPING,
    "issue_category": ISSUE_CATEGORY_MAPPING,
}

NOTE_WHOLE_NUMBER_PATTERNS = {
    "note_sequence": WHOLE_NUMBER_PATTERN,
}


RANDOM_SEED = 42
NOTE_HELD_OUT_SHARE = 0.2
METRIC_DECIMALS = 4

AMENDMENT_MODEL_FILE = "output/models/amendment_model.pkl"
EXPOSURE_MODEL_FILE = "output/models/exposure_model.pkl"
ISSUE_MODEL_FILE = "output/models/issue_model.pkl"
AMENDMENT_SCHEMA_FILE = "output/models/amendment_features.json"
EXPOSURE_SCHEMA_FILE = "output/models/exposure_features.json"
MODEL_METRICS_FILE = "output/model_metrics.json"

SUBMISSION_TARGET_ENCODING = {BOOLEAN_YES: 1, BOOLEAN_NO: 0}

START_MINUTE_COLUMN = "start_minute"
DURATION_MINUTES_COLUMN = "duration_minutes"

AMENDMENT_NUMERIC_FEATURES = [
    COPY_LENGTH_COLUMN, "supers_count", LEGAL_TEXT_SECONDS_COLUMN, "child_appeal_elements_count",
    "claims_count", "spots_booked", "channels_booked", "share_before_21_pct",
    "share_in_live_sport_pct", "days_to_first_transmission", ADVERTISER_CAMPAIGNS_COLUMN,
    ADVERTISER_AMENDMENTS_COLUMN, "agency_amendment_rate_pct",
]

AMENDMENT_CATEGORICAL_FEATURES = [
    "product_category", "sub_category", "lead_performer_age_band", "voiceover_flag",
    "celebrity_flag", "sports_personality_flag", "humour_flag", "price_claim_flag",
    "health_claim_flag", "environmental_claim_flag", "comparative_claim_flag",
    "offer_terms_shown_flag", "bonus_or_free_bet_flag", "representative_apr_shown_flag",
    "new_advertiser_flag",
]

EXPOSURE_NUMERIC_FEATURES = [
    START_MINUTE_COLUMN, DURATION_MINUTES_COLUMN, "episode_number", "forecast_under18_index",
]

EXPOSURE_CATEGORICAL_FEATURES = [
    "channel", TRANSMISSION_DAY_COLUMN, "genre", "live_flag", "live_sport_flag",
    "made_for_children_flag", "premiere_flag", "school_holiday_flag", "day_type", "lead_in_genre",
]

MINIMUM_AMENDMENT_ROC_AUC = 0.70
MINIMUM_EXPOSURE_R2 = 0.60
MINIMUM_ISSUE_MACRO_F1 = 0.75


DOCS_DIRECTORY = "data/docs"
DATABASE_DIRECTORY = "database"
VECTOR_STORE_DIRECTORY = f"{DATABASE_DIRECTORY}/chroma_store"
VECTOR_COLLECTION_NAME = "clearance_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

CHUNK_SIZE_CHARACTERS = 700
CHUNK_OVERLAP_CHARACTERS = 120
DOCUMENT_HEADER_PATTERN = r"Document\s+[A-Za-z0-9 ]+\|\s*Issue\s*\d+.*?Review cycle:\s*[A-Za-z ]+"
SECTION_HEADING_PATTERN = r"(?m)^(\d{1,2})\. [A-Z]"
CONTROL_CHARACTER_PATTERN = r"[\x00-\x08\x0e-\x1f\x7f]"
SENTENCE_BOUNDARY_PATTERN = r"(?<=[.:;])(?<![0-9]\.)\s+"

CATEGORY_CODE_DOCUMENTS = {
    "Alcohol": "Code-Alcohol.pdf",
    "Betting and gaming": "Code-Betting-And-Gaming.pdf",
    "Less healthy food": "Code-Less-Healthy-Food.pdf",
    "Consumer credit": "Code-Consumer-Credit.pdf",
    "Medicines and health": "Code-Medicines-And-Health.pdf",
}

RESTRICTED_HOURS_PATTERN = (r"Spots for this category may not be transmitted between "
                            r"((?:[01]\d|2[0-3]):[0-5]\d) and ((?:[01]\d|2[0-3]):[0-5]\d)\.")
ANY_TIME_PATTERN = r"Spots for this category may be transmitted at any time of the broadcast day\."
LIVE_SPORT_EXCEPTION_PATTERN = (r"The restricted hours do not apply to a centre break within live "
                                r"sport coverage that begins more than (\d+) minutes after the "
                                r"coverage starts and more than (\d+) minutes before it ends\.")

CHILDREN_IN_OR_ADJACENT = "in or adjacent"
CHILDREN_IN_ONLY = "in"

CHILDREN_RULE_PATTERNS = {
    CHILDREN_IN_OR_ADJACENT: (r"Spots may not be transmitted in or adjacent to a programme made "
                              r"for children\."),
    CHILDREN_IN_ONLY: r"Spots may not be transmitted in a programme made for children\.",
}

CODE_FIGURE_PATTERNS = {
    "index_threshold": (r"Spots may not be transmitted in or adjacent to a programme whose "
                        r"under-18 audience index is (\d+) or more\."),
    "daily_cap": (r"No more than (\d+) spots of one campaign may be transmitted on one channel in "
                  r"one broadcast day\."),
    "separation_minutes": (r"Two spots of one campaign on one channel must be at least (\d+) "
                           r"minutes apart\."),
}


CAMPAIGN_REFERENCE_COLUMN = "campaign_reference"
SPOT_REFERENCE_COLUMN = "spot_reference"
CHANNEL_COLUMN = "channel"
BREAK_TIME_COLUMN = "break_time"
SPOT_LENGTH_COLUMN = "spot_length_seconds"
SCHEDULE_ENTRY_COLUMN = "schedule_entry_id"
PROGRAMME_TITLE_COLUMN = "programme_title"
FORECAST_INDEX_COLUMN = "forecast_under18_index"
LIVE_SPORT_FLAG_COLUMN = "live_sport_flag"
MADE_FOR_CHILDREN_FLAG_COLUMN = "made_for_children_flag"
NOTE_SOURCE_COLUMN = "note_source"
NOTE_SEQUENCE_COLUMN = "note_sequence"

BOOKED_SPOT_REQUIRED_COLUMNS = [
    SPOT_REFERENCE_COLUMN, CLOCK_NUMBER_COLUMN, CHANNEL_COLUMN, SCHEDULING_SYSTEM_COLUMN,
    TRANSMISSION_DAY_COLUMN, BREAK_TIME_COLUMN, SPOT_LENGTH_COLUMN,
]

BOOKED_SPOT_VALUE_MAPPINGS = {
    TRANSMISSION_DAY_COLUMN: WEEKDAY_MAPPING,
    SCHEDULING_SYSTEM_COLUMN: SCHEDULING_SYSTEM_MAPPING,
    CHANNEL_COLUMN: CHANNEL_MAPPING,
}

CAMPAIGN_NOT_IN_QUEUE_MESSAGE = "Campaign {campaign_reference} is not in the clearance queue"

BREAK_IN = "in"
BREAK_ADJACENT = "adjacent"

BREACH_WINDOW = "WINDOW"
BREACH_INDEX = "INDEX"
BREACH_CHILDREN = "CHILDREN"
BREACH_FREQUENCY = "FREQUENCY"
BREACH_SEPARATION = "SEPARATION"
BREACH_CODES = [BREACH_WINDOW, BREACH_INDEX, BREACH_CHILDREN, BREACH_FREQUENCY, BREACH_SEPARATION]

DECISION_CLEAR = "CLEAR"
DECISION_CLEAR_WITH_RESTRICTIONS = "CLEAR WITH RESTRICTIONS"
DECISION_REJECT = "REJECT"
DECISIONS = [DECISION_CLEAR, DECISION_CLEAR_WITH_RESTRICTIONS, DECISION_REJECT]

AMENDMENT_SCORE_DECIMALS = 4
AMENDMENT_BAND_HIGH = "HIGH"
AMENDMENT_BAND_MEDIUM = "MEDIUM"
AMENDMENT_BAND_LOW = "LOW"
AMENDMENT_BAND_HIGH_THRESHOLD = 0.60
AMENDMENT_BAND_MEDIUM_THRESHOLD = 0.30

NOTE_SOURCE_WEIGHTS = {
    "Executive first read": 1,
    "Legal review": 2,
    "Agency reply": 1,
    "Executive update": 1,
    "Controller comment": 1,
}
NOTE_AGREEMENT_DECIMALS = 4

EXPOSURE_DECIMALS = 1

GUIDE_DOCUMENT_BY_ISSUE = {
    "Health claim": "Guide-Health-Claims.pdf",
    "Price or offer claim": "Guide-Price-And-Offer-Claims.pdf",
    "Environmental claim": "Guide-Environmental-Claims.pdf",
    "Comparative claim": "Guide-Comparative-Claims.pdf",
    "Appeal to under-18s": "Guide-Appeal-To-Under-18s.pdf",
    "Gambling inducement": "Guide-Gambling-Inducements.pdf",
    "Alcohol and social success": "Guide-Alcohol-And-Social-Success.pdf",
    "Financial promotion": "Guide-Financial-Promotions.pdf",
}

EXECUTIVE_CHECKS_QUESTION = "What does the executive check for this issue?"
REBOOKING_QUESTION = "How is a withdrawn spot re-booked, and what should the agency do next?"
STANDING_RESTRICTIONS_QUESTION = "What standing restrictions apply to every cleared spot?"


ORGANISATION_NAME = "Harbourlight Media"

OPEN_CAMPAIGNS_FILE = "data/input/open_campaigns.csv"
BOOKED_SPOTS_FILE = "data/input/booked_spots.csv"
PROGRAMME_SCHEDULE_FILE = "data/input/programme_schedule.csv"
OPEN_CAMPAIGN_NOTES_FILE = "data/input/open_campaign_notes.csv"

PRODUCT_CATEGORY_COLUMN = "product_category"
AGENCY_CODE_COLUMN = "agency_code"
AGENCY_NAME_COLUMN = "agency_name"
ADVERTISER_NAME_COLUMN = "advertiser_name"
COPY_TITLE_COLUMN = "copy_title"

APPLICATION_DATABASE_FILE = f"{DATABASE_DIRECTORY}/breakcheck.db"
APPLICATION_DATABASE_URL = f"sqlite:///./{APPLICATION_DATABASE_FILE}"

RUN_ID_PREFIX = "RUN-"
ASSESSMENT_ID_PREFIX = "ASM-"
IDENTIFIER_HEX_LENGTH = 12

GEMINI_API_KEY_VARIABLE = "GEMINI_API_KEY"
GEMINI_MODEL_VARIABLE = "GEMINI_MODEL"
GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_TEMPERATURE = 0.2

AGENT_NAME_PLANNER = "clearance_intake_planner"
AGENT_NAME_COPY_RISK = "copy_risk_analyst"
AGENT_NAME_ISSUE = "issue_analyst"
AGENT_NAME_SCHEDULE = "schedule_compliance_advisor"
AGENT_NAME_WRITER = "clearance_writer"
AGENT_NAME_RESTRICTION_NOTICE = "restriction_notice_agent"
AGENT_NAME_CLEARANCE_CERTIFICATE = "clearance_certificate_agent"

RESPONSE_DECORATION_CHARACTERS = "*#`_~ \t-–—•>:[]"
SPOT_REFERENCE_MENTION_PATTERN = r"(?<![\w-]){spot_reference}(?![\w-])"
BOOLEAN_LABELS = {True: BOOLEAN_YES, False: BOOLEAN_NO}

AGENT_RUN_STATUS_RUNNING = "RUNNING"
AGENT_RUN_STATUS_AWAITING_DECISION = "AWAITING_DECISION"
AGENT_RUN_STATUS_COMPLETE = "COMPLETE"
AGENT_RUN_STATUS_FAILED = "FAILED"
AGENT_RUN_FINISHED_STATUSES = [AGENT_RUN_STATUS_COMPLETE, AGENT_RUN_STATUS_FAILED]

CAMPAIGN_STATUS_PENDING = "PENDING_DECISION"
CAMPAIGN_STATUS_CLEARED = "CLEARED"
CAMPAIGN_STATUS_CLEARED_WITH_RESTRICTIONS = "CLEARED_WITH_RESTRICTIONS"
CAMPAIGN_STATUS_REJECTED = "REJECTED"
CAMPAIGN_STATUS_BY_DECISION = {
    DECISION_CLEAR: CAMPAIGN_STATUS_CLEARED,
    DECISION_CLEAR_WITH_RESTRICTIONS: CAMPAIGN_STATUS_CLEARED_WITH_RESTRICTIONS,
    DECISION_REJECT: CAMPAIGN_STATUS_REJECTED,
}
DECISION_NOT_OPEN_MESSAGE = ("{decision} is not open for this campaign: {spots_breaching} of its "
                             "{spots_booked} spots break the code and {spots_compliant} comply")


CHECKPOINT_DATABASE_FILE = f"{DATABASE_DIRECTORY}/checkpoint.db"

LLM_CALLS_BEFORE_DECISION = 5
LLM_CALLS_BY_DECISION = {
    DECISION_CLEAR: 6,
    DECISION_CLEAR_WITH_RESTRICTIONS: 7,
    DECISION_REJECT: 5,
}

CAMPAIGN_STATUS_OPTIONS = [CAMPAIGN_STATUS_PENDING, CAMPAIGN_STATUS_CLEARED,
                           CAMPAIGN_STATUS_CLEARED_WITH_RESTRICTIONS, CAMPAIGN_STATUS_REJECTED]
DECIDED_CAMPAIGN_STATUSES = [CAMPAIGN_STATUS_CLEARED, CAMPAIGN_STATUS_CLEARED_WITH_RESTRICTIONS,
                             CAMPAIGN_STATUS_REJECTED]
QUEUE_CATEGORY_ALL = "All"
QUEUE_CATEGORY_OPTIONS = [QUEUE_CATEGORY_ALL] + list(CATEGORY_CODE_DOCUMENTS)

EXECUTIVE_REQUEST_MINIMUM_CHARACTERS = 5
DECISION_REASON_MINIMUM_CHARACTERS = 10

ROLE_EXECUTIVE = "Clearance Executive"
ROLE_CONTROLLER = "Clearance Controller"
ROLE_LIAISON = "Agency Liaison"

PAGE_QUEUE = "Clearance queue"
PAGE_CAMPAIGN = "Campaign detail"
PAGE_DECISION = "Decision queue"
PAGE_AUDIT = "Audit trail"
PAGE_CHAT = "Clearance chat"

ROLE_PAGE_ACCESS = {
    ROLE_CONTROLLER: [PAGE_QUEUE, PAGE_CAMPAIGN, PAGE_DECISION, PAGE_AUDIT, PAGE_CHAT],
    ROLE_EXECUTIVE: [PAGE_QUEUE, PAGE_CAMPAIGN, PAGE_AUDIT, PAGE_CHAT],
    ROLE_LIAISON: [PAGE_CHAT],
}

AREA_INTAKE = "intake"
AREA_ASSESSMENT = "assessment"
AREA_DECISION = "decision"
AREA_AUDIT = "audit"
AREA_CHAT = "chat"

ROLE_AREA_ACCESS = {
    ROLE_CONTROLLER: [AREA_INTAKE, AREA_ASSESSMENT, AREA_DECISION, AREA_AUDIT, AREA_CHAT],
    ROLE_EXECUTIVE: [AREA_INTAKE, AREA_ASSESSMENT, AREA_AUDIT, AREA_CHAT],
    ROLE_LIAISON: [AREA_CHAT],
}

AREA_BY_PATH_START = {
    "/queue": AREA_INTAKE,
    "/assessments": AREA_ASSESSMENT,
    "/clearances": AREA_AUDIT,
    "/audit": AREA_AUDIT,
    "/chat": AREA_CHAT,
}
DECISION_PATH_ENDING = "/decision"

SEED_USER_ACCOUNTS = [
    {"username": "priya", "display_name": "Priya Raman", "role": ROLE_CONTROLLER,
     "assigned_agencies": [], "password": "11111111"},
    {"username": "owen", "display_name": "Owen Hughes", "role": ROLE_CONTROLLER,
     "assigned_agencies": [], "password": "11111111"},
    {"username": "grace", "display_name": "Grace Adeyemi", "role": ROLE_CONTROLLER,
     "assigned_agencies": [], "password": "11111111"},
    {"username": "tom", "display_name": "Tom Fletcher", "role": ROLE_EXECUTIVE,
     "assigned_agencies": [], "password": "22222222"},
    {"username": "aisha", "display_name": "Aisha Rahman", "role": ROLE_EXECUTIVE,
     "assigned_agencies": [], "password": "22222222"},
    {"username": "daniel", "display_name": "Daniel Novak", "role": ROLE_EXECUTIVE,
     "assigned_agencies": [], "password": "22222222"},
    {"username": "hannah", "display_name": "Hannah Price", "role": ROLE_EXECUTIVE,
     "assigned_agencies": [], "password": "22222222"},
    {"username": "marcus", "display_name": "Marcus Bell", "role": ROLE_EXECUTIVE,
     "assigned_agencies": [], "password": "22222222"},
    {"username": "zara", "display_name": "Zara Sheikh", "role": ROLE_EXECUTIVE,
     "assigned_agencies": [], "password": "22222222"},
    {"username": "lucy", "display_name": "Lucy Carter", "role": ROLE_LIAISON,
     "assigned_agencies": ["QSC", "HWK", "PVT", "FWL"], "password": "33333333"},
    {"username": "ben", "display_name": "Ben Okoro", "role": ROLE_LIAISON,
     "assigned_agencies": ["MSL", "BKM", "SWR", "CMR"], "password": "33333333"},
    {"username": "sofia", "display_name": "Sofia Marino", "role": ROLE_LIAISON,
     "assigned_agencies": ["CRM", "RDW", "WDL", "THB"], "password": "33333333"},
    {"username": "jack", "display_name": "Jack Turner", "role": ROLE_LIAISON,
     "assigned_agencies": ["KPM", "OKB", "BLC", "SLV"], "password": "33333333"},
    {"username": "nadia", "display_name": "Nadia Haddad", "role": ROLE_LIAISON,
     "assigned_agencies": ["LNT", "GRF", "ASH", "NLT"], "password": "33333333"},
]

PASSWORD_HASH_ALGORITHM = "sha256"
PASSWORD_HASH_ROUNDS = 200000
PASSWORD_SALT_BYTES = 16
SESSION_TOKEN_BYTES = 24

SESSION_TOKEN_HEADER = "X-Session-Token"
SIGN_IN_FAILED_MESSAGE = "Username or password is not correct"
SIGN_IN_REQUIRED_MESSAGE = "Sign in first"
SESSION_ENDED_MESSAGE = ("Your session ended: this account has signed in somewhere else or "
                         "signed out. Sign in again to carry on.")
AREA_REFUSED_MESSAGE = "A {role} may not use the {area} area"
ASSESSMENT_NOT_FOUND_MESSAGE = "No clearance assessment has that identifier"
CAMPAIGN_ALREADY_ASSESSED_MESSAGE = ("Campaign {campaign_reference} has already been assessed; a "
                                     "campaign is assessed once, and a rejected copy comes back as "
                                     "a new campaign")
ALREADY_DECIDED_MESSAGE = "This clearance assessment has already been decided"
DECISION_IN_PROGRESS_MESSAGE = ("{decided_by} has sent {decision} for this assessment and it is "
                                "still being recorded; only {decided_by} can finish it, by sending "
                                "the same decision and reason again")
OWN_ASSESSMENT_REFUSED_MESSAGE = ("The person who started an assessment has reviewed the campaign "
                                  "and may not decide it; another Clearance Controller must, as "
                                  "the Advertising Clearance Policy requires")

AGENT_NAME_CLEARANCE_ASSISTANT = "clearance_assistant"

CHAT_TYPE_CODE_RULES = "CODE RULES"
CHAT_TYPE_SCHEDULING_RULES = "SCHEDULING RULES"
CHAT_TYPE_AGENCY_ADVICE = "AGENCY ADVICE"
CHAT_TYPE_CAMPAIGN_STATUS = "CAMPAIGN STATUS"
CHAT_TYPE_OUT_OF_SCOPE = "OUT OF SCOPE"
CHAT_TYPE_OPTIONS = [CHAT_TYPE_CODE_RULES, CHAT_TYPE_SCHEDULING_RULES, CHAT_TYPE_AGENCY_ADVICE,
                     CHAT_TYPE_CAMPAIGN_STATUS, CHAT_TYPE_OUT_OF_SCOPE]
CHAT_ANSWERED_TYPES = [CHAT_TYPE_CODE_RULES, CHAT_TYPE_SCHEDULING_RULES, CHAT_TYPE_AGENCY_ADVICE]

CHAT_QUESTION_MINIMUM_CHARACTERS = 3
CHAT_EXCLUDED_DOCUMENTS = ["Past-Clearances-Casebook.pdf"]
CHAT_RETRIEVAL_RESULT_COUNT = 4
CHAT_MAX_RELEVANCE_DISTANCE = 0.65
CHAT_HISTORY_TURN_LIMIT = 3
CHAT_PAGE_TURN_LIMIT = 20
CHAT_AUDIT_MAX_ROWS = 200
CAMPAIGN_REFERENCE_PATTERN = r"\bCLR-[A-Z]{3}-\d{7}\b"
CAMPAIGN_REFERENCE_FORM = "CLR-ABC-1234567"
CHAT_NO_VALUE_REPLY = "none"
CHAT_STATUS_ANSWER_IN_HISTORY = "(answered from the campaign's record, which is not repeated here)"

CHAT_RECORD_ROLES = [ROLE_EXECUTIVE, ROLE_CONTROLLER]
CHAT_REFUSAL_MESSAGE = (f"I can only answer from {ORGANISATION_NAME}'s clearance codes, policies "
                        "and guides, and I could not find anything in them that answers that.")
CHAT_CAMPAIGN_NOT_FOUND_MESSAGE = ("That campaign cannot be found. Name it by its reference, in "
                                   f"the form {CAMPAIGN_REFERENCE_FORM}.")
CHAT_NOT_ASSESSED_TEMPLATE = ("{campaign_reference} is in the clearance queue and has not been "
                              "assessed yet.")
CHAT_STATUS_TEMPLATE = "{campaign_reference} {status_phrase}. Its status last changed {changed_at}."
CHAT_STATUS_PHRASES = {
    CAMPAIGN_STATUS_PENDING: "is under review",
    CAMPAIGN_STATUS_CLEARED: "has been cleared",
    CAMPAIGN_STATUS_CLEARED_WITH_RESTRICTIONS: "has been cleared with restrictions",
    CAMPAIGN_STATUS_REJECTED: "has been rejected",
}
CHAT_DECIDED_TEMPLATE = ("The decision was {decision}: {spots_cleared} spots cleared and "
                         "{spots_withdrawn} withdrawn.")
CHAT_DOCUMENTS_SENT = {
    CAMPAIGN_STATUS_CLEARED: "The agency has been sent the clearance certificate.",
    CAMPAIGN_STATUS_CLEARED_WITH_RESTRICTIONS: ("The agency has been sent a restriction notice and "
                                                "the clearance certificate."),
    CAMPAIGN_STATUS_REJECTED: ("The agency has been sent neither a certificate nor a notice, and "
                               "may submit the copy again, amended where it needs to be, as a new "
                               "campaign."),
}
STATUS_CHANGE_TIME_FORMAT = "on %A at %H:%M UTC"

API_HOST = "0.0.0.0"
API_PORT = 8080
API_TITLE = "BreakCheck AI API"
API_BASE_URL = f"http://localhost:{API_PORT}"
API_REQUEST_TIMEOUT_SECONDS = 300

DASHBOARD_TITLE = "BreakCheck AI"
DASHBOARD_SUBTITLE = f"{ORGANISATION_NAME} | advertising clearance desk"
DASHBOARD_TIME_FORMAT = "%a %H:%M UTC"
AMENDMENT_BAND_BADGE_COLOURS = {AMENDMENT_BAND_HIGH: "red", AMENDMENT_BAND_MEDIUM: "orange",
                                AMENDMENT_BAND_LOW: "green"}
DECISION_BADGE_COLOURS = {DECISION_CLEAR: "green", DECISION_CLEAR_WITH_RESTRICTIONS: "orange",
                          DECISION_REJECT: "red"}
STATUS_BADGE_COLOURS = {CAMPAIGN_STATUS_PENDING: "orange", CAMPAIGN_STATUS_CLEARED: "green",
                        CAMPAIGN_STATUS_CLEARED_WITH_RESTRICTIONS: "blue",
                        CAMPAIGN_STATUS_REJECTED: "red"}
CHAT_TYPE_BADGE_COLOURS = {CHAT_TYPE_CODE_RULES: "blue", CHAT_TYPE_SCHEDULING_RULES: "violet",
                           CHAT_TYPE_AGENCY_ADVICE: "green", CHAT_TYPE_CAMPAIGN_STATUS: "orange",
                           CHAT_TYPE_OUT_OF_SCOPE: "gray"}

PROGRAMME_KIND_OTHER = "Programme"
PROGRAMME_KIND_LIVE_SPORT = "Live sport"
PROGRAMME_KIND_CHILDREN = "Made for children"
PROGRAMME_KIND_COLOURS = {PROGRAMME_KIND_OTHER: "#d6d3cc", PROGRAMME_KIND_LIVE_SPORT: "#2a78d6",
                          PROGRAMME_KIND_CHILDREN: "#eb6834"}
SPOT_RESULT_COMPLIANT = "Compliant spot"
SPOT_RESULT_BREACHING = "Breaching spot"
SPOT_RESULT_COLOURS = {SPOT_RESULT_COMPLIANT: "#0ca30c", SPOT_RESULT_BREACHING: "#d03b3b"}
SPOT_RESULT_SHAPES = {SPOT_RESULT_COMPLIANT: "circle", SPOT_RESULT_BREACHING: "cross"}
CHART_SURFACE_COLOUR = "#fafaf9"
CHART_LABEL_COLOUR = "#52514e"
CHART_AXIS_COLOUR = "#c3c2b7"
CHART_GRID_COLOUR = "#e1e0d9"
BROADCAST_CLOCK_TICK_MINUTES = 120
STRIP_ROW_HEIGHT = 48
STRIP_LABEL_WIDTH = 130
