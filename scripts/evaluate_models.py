"""Script for comparing different embedding models performance."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from api.db.session import get_sessionmaker
from api.db.models import Item
from etl.embedding_templates import format_with_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    model_id: str
    dimension: int
    description: str


MODELS = [
    ModelConfig("baseline", "all-MiniLM-L6-v2", 384, "Current production model"),
    ModelConfig("e5", "intfloat/e5-small-v2", 384, "Latest improved Microsoft model"),
    # Add more models as they become available
]

SUPPORTED_MODEL_IDS = {cfg.model_id for cfg in MODELS}


def load_test_pairs() -> List[Tuple[int, int, float]]:
    """Load manually curated test pairs with similarity scores."""
    test_pairs_file = Path("tests/data/movie_pairs.json")
    if not test_pairs_file.exists():
        # Create initial test set
        pairs = [
            # Very similar movies
            (1025, 614, 0.9),  # The Incredibles, Guardians of the Galaxy (superhero)
            # Somewhat similar
            (598, 659, 0.7),  # Fantastic 4, Maze Runner (sci-fi action)
            # Different genres
            (558, 1103, 0.2),  # Scary Movie, Insidious (comedy vs horror)
        ]
        test_pairs_file.parent.mkdir(exist_ok=True)
        with open(test_pairs_file, "w") as f:
            json.dump(pairs, f)
    else:
        with open(test_pairs_file) as f:
            pairs = json.load(f)
    return pairs


def get_test_items(db: Session) -> List[Dict[str, Any]]:
    """Get items used in test pairs."""
    pairs = load_test_pairs()
    item_ids = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    items = []
    for item in db.query(Item).filter(Item.id.in_(item_ids)):
        items.append(
            {
                "id": item.id,
                "title": item.title,
                "overview": item.overview,
                "genres": item.genres,
                "release_year": item.release_year,
                "media_type": item.media_type,
                "runtime": item.runtime,
            }
        )
    return items


def evaluate_model(
    model_config: ModelConfig,
    items: List[Dict[str, Any]],
    test_pairs: List[Tuple[int, int, float]],
) -> Dict[str, float | str]:
    """Evaluate a model's performance on test pairs."""
    if not items or not test_pairs:
        raise Exception("Cannot evaluate model without test items and pairs")

    # Basic validation to avoid attempting to download arbitrary models during tests.
    model_id = model_config.model_id.strip()
    if not model_id:
        raise ValueError("Model id cannot be empty")
    if model_id not in SUPPORTED_MODEL_IDS and not Path(model_id).exists():
        raise ValueError(f"Unsupported model id '{model_id}'")

    # Load model
    try:
        model = SentenceTransformer(model_id)
    except Exception as e:
        raise Exception(f"Failed to load model {model_id}: {e}")

    # Format items using template
    texts = []
    id_to_idx = {}
    for idx, item in enumerate(items):
        texts.append(format_with_template("structured", item))
        id_to_idx[item["id"]] = idx

    # Generate embeddings
    start = time.time()
    embeddings = model.encode(texts, normalize_embeddings=True)
    encode_time = time.time() - start

    # Calculate similarities for test pairs
    errors = []
    for id1, id2, expected in test_pairs:
        idx1, idx2 = id_to_idx[id1], id_to_idx[id2]
        sim = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0][0]
        error = abs(sim - expected)
        errors.append(error)

    mse = np.mean(np.array(errors) ** 2)
    mae = np.mean(np.abs(errors))

    return {
        "model": model_config.name,
        "mse": float(mse),
        "mae": float(mae),
        "encode_time": encode_time,
        "num_items": len(items),
    }


def main():
    """Run evaluation of all models."""
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        items = get_test_items(db)
        test_pairs = load_test_pairs()

        results = []
        for model_config in MODELS:
            logger.info(f"Evaluating {model_config.name}...")
            try:
                metrics = evaluate_model(model_config, items, test_pairs)
                results.append(metrics)
                logger.info(f"Results for {model_config.name}:")
                logger.info(f"  MSE: {metrics['mse']:.4f}")
                logger.info(f"  MAE: {metrics['mae']:.4f}")
                logger.info(f"  Encode time: {metrics['encode_time']:.2f}s")
            except Exception as e:
                logger.error(f"Error evaluating {model_config.name}: {e}")

        # Save results
        output_file = Path("tests/data/model_evaluation_results.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
