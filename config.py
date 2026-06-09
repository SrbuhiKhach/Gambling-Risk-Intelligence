# ============================================================
# config.py — Ծրագրի կենտրոնական կոնֆիգուրացիա
# ============================================================
# Բոլոր path-երը, հաստատունները և պարամետրերը մեկ տեղում։
# Մյուս ֆայլերը import անում են այստեղից, ուղղակի ֆայլերում
# hard-code անելու փոխարեն։

from pathlib import Path

# ── Project root ──────────────────────────────────────────
# Path(__file__).parent — config.py-ի թղթապանակը (project root)
ROOT_DIR = Path(__file__).parent

# ── Տվյալների թղթապանակներ ──────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # բնօրինակ CSV-ներ
PROCESSED_DIR = DATA_DIR / "processed"  # մաքրված/merged տվյալներ
FIGURES_DIR = DATA_DIR / "figures"  # պահված գրաֆիկներ

# ── Ֆայլեր ───────────────────────────────────────────────
PLAYERS_CSV = RAW_DIR / "players.csv"
FEATURES_CSV = RAW_DIR / "features.csv"
TRANSACTIONS_CSV = RAW_DIR / "transactions_sample_50k.csv"
SUMMARY_CSV = RAW_DIR / "summary_by_risk.csv"

# ── Արդյունքների թղթապանակներ ───────────────────────────
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# ── ML հաստատուններ ─────────────────────────────────────
RANDOM_SEED = 42  # reproducibility-ի համար ամենուր
TEST_SIZE = 0.20  # train/test split — 80% / 20%
CV_FOLDS = 5  # cross-validation fold-երի քանակ

# ── Թիրախային փոփոխական ─────────────────────────────────
TARGET_COL = "risk_label"  # binary: 1 = at_risk կամ problem

# ── 9 վարքագծային ցուցիչներ (features) ──────────────────
# Սրանք են ML մոდելի input-ները։
# total_bet_amd-ը ԲԱՑԱՌՎԱԾ է (corr=0.95 stake_variance-ի հետ → multicollinearity)
FEATURE_COLS = [
    "bet_frequency",  # Indicator 1: խաղադրույքների հաճախականություն (bets/week)
    "stake_variance",  # Indicator 2: խաղադրույքի չափի SD (AMD) — loss aversion proxy
    "chasing_ratio",  # Indicator 3: հաջ/նախ stake ratio կորստից հետո — DSM-5
    "session_duration_avg",  # Indicator 4: միջին session-ի տևողություն (րոպե)
    "loss_streak_response",  # Indicator 5: stake փոփոխություն 3+ կորստից հետո
    "deposit_frequency",  # Indicator 6: ավանդների հաճախականություն (deposits/month)
    "time_of_day_skew",  # Indicator 7: գիշերային (00:00–06:00) խաղերի բաժին
    "withdrawal_ratio",  # Indicator 8: withdrawals / deposits ratio
    "self_excl_attempts",  # Indicator 9: ինքնաբացառման փորձերի թիվ
]

# ── Risk level կատեգորիաներ ──────────────────────────────
RISK_ORDER = ["recreational", "regular", "at_risk", "problem"]
RISK_COLORS = {
    "recreational": "#2ecc71",  # կանաչ
    "regular": "#3498db",  # կապույտ
    "at_risk": "#f39c12",  # նարնջագույն
    "problem": "#e74c3c",  # կարմիր
}

# ── Գնահատման շեմեր (proposal-ից) ───────────────────────
THRESHOLDS = {
    "auc_roc": 0.85,
    "f1": 0.78,
    "precision_k": 0.75,
    "recall": 0.80,
    "brier": 0.15,
}

# ── Թղթապանակների ստեղծում (եթե գոյություն չունեն) ──────
for _dir in [PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)