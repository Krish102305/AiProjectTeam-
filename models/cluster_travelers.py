"""Load traveler profiles, compare elbows, and score a k=5 seating model.

KMeans is fit on the training rows only. The test curve and the v1 pool
labels use those same centers and are never refit.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "profiles_clean.csv"
CHART_DIR = Path(__file__).resolve().parent / "charts"
CHART_PATH = CHART_DIR / "elbow_train_vs_test.png"
CLASH_CHART_PATH = CHART_DIR / "clash_rate_baseline_vs_v1.png"

# sex is kept on the prepared table but is not a clustering feature.
FEATURE_COLUMNS = ["age", "drinks", "smokes", "education_level", "diet_strictness"]
IMPUTE_COLUMNS = ["drinks", "smokes", "education_level"]

# "unknown" is treated the same as flexible.
DIET_STRICTNESS = {
    "flexible": 0,
    "mostly": 1,
    "strict": 2,
    "unknown": 0,
}

RANDOM_STATE = 42
K_VALUES = range(1, 11)
V1_K = 5
TABLE_SIZE = 5
N_SEATINGS = 20

TRAVELERS_PER_MONTH = 1000
PRICE_PER_NIGHT = 45
REBOOK_AFTER_GOOD = 0.40
REBOOK_AFTER_CLASH = 0.15

SEATING_RULE = (
    "Seat the traveler only with people in the same pool, "
    "filling tables of 5 and balancing gender within the pool."
)

# Filled in main once the train scaler and k=5 model exist.
TRAINED_SCALER = None
V1_MODEL = None
POOL_AVERAGES = None


def load_and_prepare(path=DATA_PATH):
    """Return the prepared table and how many missing values were filled."""
    profiles = pd.read_csv(path)

    prepared = profiles.copy()
    prepared["diet_strictness"] = prepared["diet_strictness"].map(DIET_STRICTNESS)

    filled_counts = {}
    for column in IMPUTE_COLUMNS:
        missing = int(prepared[column].isna().sum())
        filled_counts[column] = missing
        prepared[column] = prepared[column].fillna(prepared[column].median())

    prepared = prepared[["sex", *FEATURE_COLUMNS]]
    features = prepared[FEATURE_COLUMNS]
    return prepared, features, filled_counts


def average_squared_distance(model, X):
    """Mean squared distance from each row to its nearest fitted center.

    Uses the centers already learned by ``model``. Does not refit.
    """
    return float(-model.score(X) / len(X))


def find_elbow(ks, distances):
    """Return the k farthest from the line joining the first and last points.

    Both axes are min-max scaled to 0-1 before the distance is measured, so
    neither k nor the distance dominates the comparison.
    """
    ks = np.asarray(ks, dtype=float)
    distances = np.asarray(distances, dtype=float)

    x = (ks - ks.min()) / (ks.max() - ks.min())
    y_span = distances.max() - distances.min()
    y = (distances - distances.min()) / y_span

    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line = end - start
    line_length = np.hypot(line[0], line[1])

    offsets = np.empty(len(x))
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        diff = np.array([x_i, y_i]) - start
        cross = line[0] * diff[1] - line[1] * diff[0]
        offsets[i] = abs(cross) / line_length

    return int(ks[int(np.argmax(offsets))])


def fit_distance_curves(X_train, X_test):
    """Fit KMeans for each k on train and score train and test."""
    rows = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        model.fit(X_train)
        rows.append(
            {
                "k": k,
                "train_distance": average_squared_distance(model, X_train),
                "test_distance": average_squared_distance(model, X_test),
            }
        )
    return pd.DataFrame(rows)


def save_elbow_chart(curve, train_elbow, test_elbow, path=CHART_PATH):
    """Save both distance curves with a dashed line at each elbow."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        curve["k"],
        curve["train_distance"],
        marker="o",
        label="Train",
    )
    ax.plot(
        curve["k"],
        curve["test_distance"],
        marker="o",
        label="Test",
    )

    if train_elbow == test_elbow:
        ax.axvline(
            train_elbow,
            linestyle="--",
            color="gray",
            label=f"Elbow (k={train_elbow})",
        )
    else:
        ax.axvline(
            train_elbow,
            linestyle="--",
            color="C0",
            label=f"Train elbow (k={train_elbow})",
        )
        ax.axvline(
            test_elbow,
            linestyle="--",
            color="C1",
            label=f"Test elbow (k={test_elbow})",
        )

    ax.set_xlabel("k")
    ax.set_ylabel("Average squared distance")
    ax.set_xticks(list(K_VALUES))
    ax.set_title("Elbow: train vs test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pool_profile(train_rows):
    """Average of each real-unit feature by pool, plus the pool's share of train."""
    profile = train_rows.groupby("pool")[FEATURE_COLUMNS].mean()
    profile["share"] = train_rows.groupby("pool").size() / len(train_rows)
    return profile


def table_has_clash(table):
    """True when a table of people trips any clash rule."""
    age = table["age"].to_numpy()
    drinks = table["drinks"].to_numpy()
    smokes = table["smokes"].to_numpy()
    diet = table["diet_strictness"].to_numpy()

    age_spread = float(age.max() - age.min()) > 15
    drinks_clash = bool(np.any(drinks == 0) and np.any(drinks >= 3))
    smokes_clash = bool(np.any(smokes == 0) and np.any(smokes == 2))
    strict_with_flexible_majority = bool(
        np.any(diet == 2) and np.mean(diet == 0) > 0.5
    )
    return age_spread or drinks_clash or smokes_clash or strict_with_flexible_majority


def iter_tables(people, rng, by_pool):
    """Shuffle people and yield full tables of TABLE_SIZE.

    Baseline draws from everyone. v1 draws inside each predicted pool.
    People left over after filling tables of 5 are not seated.
    """
    if by_pool:
        groups = [group for _, group in people.groupby("pool", sort=True)]
    else:
        groups = [people]

    for group in groups:
        order = rng.permutation(len(group))
        seated = group.iloc[order]
        n_full = len(seated) // TABLE_SIZE
        for start in range(0, n_full * TABLE_SIZE, TABLE_SIZE):
            yield seated.iloc[start : start + TABLE_SIZE]


def clash_share(people, by_pool):
    """Share of tables with a clash across N_SEATINGS random seatings."""
    rng = np.random.default_rng(RANDOM_STATE)
    clashes = 0
    tables = 0
    for _ in range(N_SEATINGS):
        for table in iter_tables(people, rng, by_pool):
            tables += 1
            if table_has_clash(table):
                clashes += 1
    return clashes / tables


def save_clash_chart(baseline_share, v1_share, path=CLASH_CHART_PATH):
    """Save baseline and v1 clash rates with the percentage on each bar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["Baseline", "v1"]
    percents = [baseline_share * 100, v1_share * 100]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, percents)
    for bar, percent in zip(bars, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percent:.1f}%",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Tables with a clash (%)")
    ax.set_title("Clash rate: baseline vs v1")
    ax.set_ylim(0, max(percents) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def assign_pool(traveler):
    """Scale one traveler with the train scaler and return their k=5 pool.

    Gender is not a feature. After the pool is chosen, seat this traveler
    only with people in that pool, filling tables of 5 and balancing gender
    inside the pool.
    """
    row = pd.DataFrame(
        [{column: traveler[column] for column in FEATURE_COLUMNS}]
    )
    scaled = TRAINED_SCALER.transform(row)
    pool = int(V1_MODEL.predict(scaled)[0])
    averages = POOL_AVERAGES.loc[pool, FEATURE_COLUMNS]
    return {
        "pool": pool,
        "averages": {
            column: float(averages[column]) for column in FEATURE_COLUMNS
        },
        "rule": SEATING_RULE,
    }


def repeat_revenue(clash_share):
    """Rebook rate and monthly repeat revenue from a clash share."""
    rebook_rate = (
        (1 - clash_share) * REBOOK_AFTER_GOOD + clash_share * REBOOK_AFTER_CLASH
    )
    monthly = TRAVELERS_PER_MONTH * rebook_rate * PRICE_PER_NIGHT
    return rebook_rate, monthly


def main():
    global TRAINED_SCALER, V1_MODEL, POOL_AVERAGES
    prepared, features, filled_counts = load_and_prepare()

    print(f"Rows: {len(prepared)}")
    print("Missing values filled:")
    for column, count in filled_counts.items():
        print(f"  {column}: {count}")
    print("First 5 rows of features:")
    print(features.head())
    print()

    train_rows, test_rows = train_test_split(
        prepared,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_rows[FEATURE_COLUMNS])
    X_test = scaler.transform(test_rows[FEATURE_COLUMNS])

    curve = fit_distance_curves(X_train, X_test)
    train_elbow = find_elbow(curve["k"], curve["train_distance"])
    test_elbow = find_elbow(curve["k"], curve["test_distance"])

    print(f"{'k':>3}  {'train distance':>16}  {'test distance':>16}")
    for row in curve.itertuples(index=False):
        print(f"{row.k:>3}  {row.train_distance:>16.4f}  {row.test_distance:>16.4f}")
    print(f"Train elbow: k={train_elbow}")
    print(f"Test elbow: k={test_elbow}")
    print(f"Elbows match: {'yes' if train_elbow == test_elbow else 'no'}")

    save_elbow_chart(curve, train_elbow, test_elbow)
    print(f"Saved chart to {CHART_PATH}")
    print()

    train_rows = train_rows.copy()
    test_rows = test_rows.copy()
    v1 = KMeans(n_clusters=V1_K, n_init=10, random_state=RANDOM_STATE)
    v1.fit(X_train)
    train_rows["pool"] = v1.labels_
    test_rows["pool"] = v1.predict(X_test)

    profile = pool_profile(train_rows)
    print("Pool averages (train, real units):")
    print(profile.to_string(float_format=lambda value: f"{value:.4f}"))
    print()

    baseline_share = clash_share(test_rows, by_pool=False)
    v1_share = clash_share(test_rows, by_pool=True)
    print(f"Baseline clash share: {baseline_share:.1%}")
    print(f"v1 clash share: {v1_share:.1%}")

    save_clash_chart(baseline_share, v1_share)
    print(f"Saved chart to {CLASH_CHART_PATH}")
    print()

    TRAINED_SCALER = scaler
    V1_MODEL = v1
    POOL_AVERAGES = profile

    new_traveler = {
        "age": 26,
        "drinks": 2,
        "smokes": 0,
        "education_level": 3,
        "diet_strictness": 2,
    }
    assignment = assign_pool(new_traveler)
    print("New traveler:")
    print(new_traveler)
    print(f"Pool: {assignment['pool']}")
    print("Pool averages:")
    for column, value in assignment["averages"].items():
        print(f"  {column}: {value:.4f}")
    print(f"Rule: {assignment['rule']}")
    print()

    print("ASSUMPTIONS")
    print("1. 1,000 travelers per month")
    print("2. $45 per night")
    print("3. 40% rebook after a good table")
    print("4. 15% rebook after a clash table")
    print("rebook rate = (1 - clash share) * 0.40 + clash share * 0.15")
    print("monthly repeat revenue = 1000 * rebook rate * 45")
    print()

    baseline_rebook, baseline_monthly = repeat_revenue(baseline_share)
    v1_rebook, v1_monthly = repeat_revenue(v1_share)
    monthly_difference = v1_monthly - baseline_monthly
    yearly_difference = monthly_difference * 12

    print(f"Baseline rebook rate: {baseline_rebook:.2%}")
    print(f"Baseline monthly repeat revenue: ${baseline_monthly:,.2f}")
    print(f"v1 rebook rate: {v1_rebook:.2%}")
    print(f"v1 monthly repeat revenue: ${v1_monthly:,.2f}")
    print(f"Difference per month (v1 - baseline): ${monthly_difference:,.2f}")
    print(f"Difference per year (v1 - baseline): ${yearly_difference:,.2f}")


if __name__ == "__main__":
    main()
