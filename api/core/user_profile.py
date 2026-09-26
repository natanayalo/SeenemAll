from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Sequence, Dict, List, Tuple, Any
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from api.db.models import User, UserHistory, ItemEmbedding, Item
from api.config import USER_PROFILE_DECAY_HALF_LIFE

EVENT_TYPE_WEIGHTS: dict[str, float] = {
    "watched": 1.0,
    "liked": 2.0,
    "rated": 3.0,
}

COLLAB_BLEND = 0.3
COLLAB_TOP_K = 20
NEGATIVE_EVENT_TYPES = {"not_interested", "disliked"}


@dataclass
class NeighborInfo:
    user_id: str
    weight: float


def _event_weight(event_type: str, base_weight: float | int | None) -> float:
    multiplier = EVENT_TYPE_WEIGHTS.get(event_type, 1.0)
    base = float(base_weight if base_weight is not None else 1.0)
    if base < 0:
        base = 0.0
    weight = multiplier * base
    return weight


def _collect_collaborative_vector(
    db: Session, user_id: str, item_ids: Sequence[int]
) -> tuple[Optional[np.ndarray], List[NeighborInfo]]:
    if not item_ids:
        return None, []

    collab_weights: Dict[str, float] = defaultdict(float)
    try:
        rows = (
            db.query(UserHistory.user_id, UserHistory.weight)
            .filter(
                UserHistory.item_id.in_(item_ids),
                UserHistory.user_id != user_id,
            )
            .all()
        )
    except SQLAlchemyError:
        # Some tests use lightweight session stubs that cannot answer this query.
        return None, []

    for other_user_id, base_weight in rows:
        if other_user_id is None:
            continue
        collab_weights[str(other_user_id)] += float(
            base_weight if base_weight is not None else 1.0
        )

    if not collab_weights:
        return None, []

    ranked = sorted(collab_weights.items(), key=lambda kv: kv[1], reverse=True)[
        :COLLAB_TOP_K
    ]
    neighbor_ids = [uid for uid, _ in ranked]
    if not neighbor_ids:
        return None, []

    neighbors = db.query(User).filter(User.user_id.in_(neighbor_ids)).all()

    vecs = []
    weights = []
    weight_lookup = dict(ranked)
    for neighbor in neighbors:
        uid = str(neighbor.user_id)
        if uid not in weight_lookup or neighbor.long_vec is None:
            continue
        vecs.append(np.array(neighbor.long_vec, dtype="float32"))
        weights.append(weight_lookup[uid])

    if not vecs:
        return None, []

    collab_matrix = np.stack(vecs)
    collab_weights_arr = np.array(weights, dtype="float32")
    averaged = np.average(collab_matrix, axis=0, weights=collab_weights_arr)
    diagnostics = [
        NeighborInfo(user_id=uid, weight=float(weight_lookup[uid]))
        for uid in neighbor_ids
        if uid in weight_lookup
    ]
    return averaged, diagnostics


DIM = 384


def _time_decay_weights(n: int, half_life: float | None = None) -> np.ndarray:
    # recent -> 1.0, older half-life ~10 items
    if n == 0:
        return np.zeros((0,), dtype="float32")
    half_life = half_life or USER_PROFILE_DECAY_HALF_LIFE
    half_life = max(float(half_life), 1e-3)
    idx = np.arange(n, dtype="float32")  # most recent first
    w = 0.5 ** (idx / half_life)
    w /= w.sum() + 1e-8
    return w


@dataclass
class TasteCluster:
    cluster_id: int
    centroid: np.ndarray
    weight: float
    size: int
    top_genres: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "centroid": (
                self.centroid.tolist()
                if isinstance(self.centroid, np.ndarray)
                else list(self.centroid)
            ),
            "weight": round(float(self.weight), 4),
            "size": int(self.size),
            "top_genres": self.top_genres,
        }


class UserVectorResult(Tuple[Any, ...]):
    """
    Backward-compatible 5-element tuple: (long_vec, short_vec, genre_prefs, neighbors, negatives)
    with a .taste_clusters attribute for multi-interest taste modeling.
    """

    def __new__(
        cls,
        long_vec: Optional[np.ndarray],
        short_vec: Optional[np.ndarray],
        genre_prefs: Dict[str, float] | None,
        neighbors: List[NeighborInfo],
        negatives: List[int],
        taste_clusters: Optional[List[TasteCluster]] = None,
    ):
        return tuple.__new__(
            cls, (long_vec, short_vec, genre_prefs, neighbors, negatives)
        )

    def __init__(
        self,
        long_vec: Optional[np.ndarray],
        short_vec: Optional[np.ndarray],
        genre_prefs: Dict[str, float] | None,
        neighbors: List[NeighborInfo],
        negatives: List[int],
        taste_clusters: Optional[List[TasteCluster]] = None,
    ):
        self.long_vec = long_vec
        self.short_vec = short_vec
        self.genre_prefs = genre_prefs
        self.neighbors = neighbors
        self.negatives = negatives
        self.taste_clusters = taste_clusters or []


def _spherical_kmeans(
    vectors: np.ndarray,
    weights: np.ndarray,
    k: int,
    max_iter: int = 25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Weighted spherical K-Means on unit-normalized vectors.
    Returns:
        centroids: (k, D) unit-normalized centroids
        labels: (N,) cluster assignments
        dispersion: weighted average cosine distance (1 - cos(v_i, centroid))
    """
    n, d = vectors.shape
    if k <= 1 or n <= 1:
        w_sum = float(weights.sum())
        if w_sum > 0:
            c = (vectors * weights[:, None]).sum(axis=0)
        else:
            c = vectors.sum(axis=0)
        norm_c = float(np.linalg.norm(c))
        centroid = (c / norm_c if norm_c > 0 else c).reshape(1, d)
        labels = np.zeros(n, dtype=int)
        cos_sims = np.clip(np.sum(vectors * centroid, axis=1), -1.0, 1.0)
        dispersion = (
            float(np.average(1.0 - cos_sims, weights=weights))
            if w_sum > 0
            else float(np.mean(1.0 - cos_sims))
        )
        return centroid, labels, dispersion

    rng = np.random.RandomState(random_state)
    first_idx = int(np.argmax(weights))
    centroids = [vectors[first_idx]]

    for _ in range(1, k):
        cur_centroids = np.stack(centroids)
        sims = np.dot(vectors, cur_centroids.T)
        max_sim = np.max(sims, axis=1)
        dists = np.maximum(0.0, 1.0 - max_sim) * weights
        d_sum = float(dists.sum())
        if d_sum > 0:
            probs = dists / d_sum
            next_idx = int(rng.choice(n, p=probs))
        else:
            next_idx = int(rng.choice(n))
        centroids.append(vectors[next_idx])

    centroid_matrix = np.stack(centroids).astype("float32")
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        sims = np.dot(vectors, centroid_matrix.T)
        new_labels = np.argmax(sims, axis=1)

        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels

        new_centroids = np.zeros_like(centroid_matrix)
        for c in range(k):
            mask = labels == c
            if not np.any(mask):
                all_sims = np.max(np.dot(vectors, centroid_matrix.T), axis=1)
                far_idx = int(np.argmin(all_sims))
                new_centroids[c] = vectors[far_idx]
                continue
            c_weights = weights[mask]
            w_sum = float(c_weights.sum())
            if w_sum > 0:
                mean_v = (vectors[mask] * c_weights[:, None]).sum(axis=0)
            else:
                mean_v = vectors[mask].sum(axis=0)
            norm_v = float(np.linalg.norm(mean_v))
            new_centroids[c] = mean_v / norm_v if norm_v > 0 else mean_v
        centroid_matrix = new_centroids

    assigned_centroids = centroid_matrix[labels]
    cos_sims = np.clip(np.sum(vectors * assigned_centroids, axis=1), -1.0, 1.0)
    w_sum = float(weights.sum())
    dispersion = (
        float(np.average(1.0 - cos_sims, weights=weights))
        if w_sum > 0
        else float(np.mean(1.0 - cos_sims))
    )
    return centroid_matrix, labels, dispersion


def cluster_user_tastes(
    vectors: np.ndarray,
    weights: np.ndarray,
    item_genres: List[List[Dict[str, Any]] | None],
    max_k: int = 3,
    min_cluster_size: int = 2,
    dispersion_reduction_threshold: float = 0.20,
) -> List[TasteCluster]:
    """
    Dynamically identify 1 to 3 distinct user taste clusters using Spherical K-Means.
    """
    n = len(vectors)
    if n == 0:
        return []

    def _extract_top_genres(indices: Sequence[int]) -> List[str]:
        g_counts: Dict[str, float] = defaultdict(float)
        for idx in indices:
            if idx < len(item_genres):
                genres_for_item = item_genres[idx]
                if genres_for_item is not None:
                    w = float(weights[idx]) if idx < len(weights) else 1.0
                    for g in genres_for_item:
                        if isinstance(g, dict) and g.get("name"):
                            g_counts[g["name"]] += w
        sorted_genres = sorted(g_counts.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in sorted_genres[:3]]

    if n == 1:
        norm_v = float(np.linalg.norm(vectors[0]))
        c0 = vectors[0] / norm_v if norm_v > 0 else vectors[0]
        return [
            TasteCluster(
                cluster_id=0,
                centroid=c0.astype("float32"),
                weight=1.0,
                size=1,
                top_genres=_extract_top_genres([0]),
            )
        ]

    c1, l1, disp1 = _spherical_kmeans(vectors, weights, k=1)
    chosen_k = 1
    best_centroids = c1
    best_labels = l1

    total_weight = float(weights.sum()) if weights.sum() > 0 else float(n)

    if n >= 2 * min_cluster_size and disp1 > 0.05:
        c2, l2, disp2 = _spherical_kmeans(vectors, weights, k=2)
        sizes2 = [int(np.sum(l2 == c)) for c in range(2)]
        w2 = [float(weights[l2 == c].sum()) / total_weight for c in range(2)]
        if min(sizes2) >= min_cluster_size and min(w2) >= 0.15:
            rel_reduction_2 = (disp1 - disp2) / (disp1 + 1e-8)
            if rel_reduction_2 >= dispersion_reduction_threshold:
                chosen_k = 2
                best_centroids = c2
                best_labels = l2

                if max_k >= 3 and n >= 3 * min_cluster_size:
                    c3, l3, disp3 = _spherical_kmeans(vectors, weights, k=3)
                    sizes3 = [int(np.sum(l3 == c)) for c in range(3)]
                    w3 = [
                        float(weights[l3 == c].sum()) / total_weight for c in range(3)
                    ]
                    if min(sizes3) >= min_cluster_size and min(w3) >= 0.10:
                        rel_reduction_3 = (disp2 - disp3) / (disp2 + 1e-8)
                        if rel_reduction_3 >= dispersion_reduction_threshold:
                            chosen_k = 3
                            best_centroids = c3
                            best_labels = l3

    clusters: List[TasteCluster] = []
    for c in range(chosen_k):
        mask = best_labels == c
        c_size = int(np.sum(mask))
        c_weight = float(weights[mask].sum()) / total_weight if c_size > 0 else 0.0
        c_indices = np.where(mask)[0]
        clusters.append(
            TasteCluster(
                cluster_id=c,
                centroid=best_centroids[c].astype("float32"),
                weight=c_weight,
                size=c_size,
                top_genres=_extract_top_genres(c_indices),
            )
        )

    clusters.sort(key=lambda cl: cl.weight, reverse=True)
    for idx, cl in enumerate(clusters):
        cl.cluster_id = idx

    return clusters


def compute_user_vector(db: Session, user_id: str) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, float] | None,
    List[NeighborInfo],
    List[int],
]:
    # pull recent history item_ids (watched/liked/rated), newest first
    rows: Sequence[tuple[int, float, str]] = (
        db.query(UserHistory.item_id, UserHistory.weight, UserHistory.event_type)
        .filter(UserHistory.user_id == user_id)
        .order_by(UserHistory.ts.desc())
        .limit(200)
        .all()
    )
    if not rows:
        return UserVectorResult(None, None, None, [], [], taste_clusters=[])
    item_ids = [r[0] for r in rows]

    embs = (
        db.query(ItemEmbedding.item_id, ItemEmbedding.vector)
        .filter(ItemEmbedding.item_id.in_(item_ids))
        .all()
    )
    vec_map = {item_id: np.array(vector, dtype="float32") for item_id, vector in embs}

    ordered_vecs: list[np.ndarray] = []
    feedback_weights: list[float] = []
    used_rows: list[tuple[int, float, str]] = []
    negative_items: set[int] = set()
    for item_id, base_weight, event_type in rows:
        normalized_event = (event_type or "").lower()
        if normalized_event in NEGATIVE_EVENT_TYPES:
            negative_items.add(item_id)
            continue
        vec = vec_map.get(item_id)
        if vec is None:
            continue
        ordered_vecs.append(vec)
        feedback_weights.append(_event_weight(event_type, base_weight))
        used_rows.append((item_id, base_weight, event_type))

    if not ordered_vecs:
        return UserVectorResult(
            None, None, None, [], sorted(negative_items), taste_clusters=[]
        )

    vecs = np.stack(ordered_vecs).astype("float32")
    weight_arr = np.array(feedback_weights, dtype="float32")
    if not np.any(weight_arr):
        weight_arr = np.ones_like(weight_arr)

    # short_vec: time-decayed mean; long_vec: uniform mean
    td = _time_decay_weights(len(vecs))
    short_weights = td * weight_arr
    if not np.any(short_weights):
        short_weights = td
    short_weights /= short_weights.sum() + 1e-8
    short_vec = (vecs * short_weights[:, None]).sum(axis=0)

    try:
        long_vec = np.average(vecs, axis=0, weights=weight_arr)
    except ZeroDivisionError:
        long_vec = vecs.mean(axis=0)

    # L2 normalize
    def norm(x):
        n = np.linalg.norm(x)
        return (x / n).astype("float32") if n > 0 else x.astype("float32")

    # Genre preferences ----------------------------------------------------
    positive_item_ids = [item_id for item_id, _, _ in used_rows]

    item_genres = {
        item_id: genres or []
        for item_id, genres in db.query(Item.id, Item.genres)
        .filter(Item.id.in_(positive_item_ids))
        .all()
    }
    genre_totals: Dict[str, float] = defaultdict(float)
    genres_per_row = [item_genres.get(item_id) for item_id, _, _ in used_rows]

    for (item_id, base_weight, event_type), weight_value in zip(
        used_rows, short_weights
    ):
        genres = item_genres.get(item_id) or []
        if not genres:
            continue
        combined = float(weight_value)
        if combined <= 0:
            continue
        for genre in genres:
            name = genre.get("name")
            if not name:
                continue
            genre_totals[name] += combined

    total = sum(genre_totals.values())
    genre_prefs = None
    if total > 0:
        genre_prefs = {
            genre: weight / total
            for genre, weight in sorted(
                genre_totals.items(), key=lambda kv: kv[1], reverse=True
            )
        }

    # Dynamic multi-interest clustering
    taste_clusters = cluster_user_tastes(
        vectors=vecs,
        weights=short_weights,
        item_genres=genres_per_row,
        max_k=3,
    )

    collab_vec, neighbor_info = _collect_collaborative_vector(
        db, user_id, positive_item_ids
    )
    if collab_vec is not None:
        collab_norm = norm(np.array(collab_vec, dtype="float32"))
        if long_vec is None:
            long_vec = collab_norm
        else:
            long_vec = (1.0 - COLLAB_BLEND) * long_vec + COLLAB_BLEND * collab_norm

    return UserVectorResult(
        norm(long_vec),
        norm(short_vec),
        genre_prefs,
        neighbor_info,
        sorted(negative_items),
        taste_clusters=taste_clusters,
    )


def upsert_user_vectors(db: Session, user_id: str) -> None:
    res = compute_user_vector(db, user_id)
    long_vec, short_vec, genre_prefs, neighbor_info, _ = res
    raw_clusters = getattr(res, "taste_clusters", []) or []
    taste_clusters = [c.to_dict() if hasattr(c, "to_dict") else c for c in raw_clusters]
    neighbor_payload = [
        {"user_id": info.user_id, "weight": info.weight} for info in neighbor_info
    ]
    user = db.query(User).filter(User.user_id == user_id).one_or_none()
    if user is None:
        user = User(
            user_id=user_id,
            long_vec=long_vec,
            short_vec=short_vec,
            genre_prefs=genre_prefs,
            neighbors=neighbor_payload,
            taste_clusters=taste_clusters,
        )
        db.add(user)
    else:
        user.long_vec = long_vec
        user.short_vec = short_vec
        user.genre_prefs = genre_prefs
        user.neighbors = neighbor_payload
        user.taste_clusters = taste_clusters
