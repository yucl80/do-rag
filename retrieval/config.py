"""Configuration settings for retrievers."""

import torch

# Relationship weights for different types of relations in the knowledge graph
RELATION_WEIGHTS = {
    "CONTAINS": 2.5,    # Strong structural relation
    "DESCRIBES": 1.8,   # Attribute relation
    "DEPENDS_ON": 2.0,  # Functional dependency
    "HAS_METRIC": 2.0,  # Performance metric
    "RELATED_TO": 0.8   # General relation
}

# Level weights for hierarchical importance in the knowledge graph
LEVEL_WEIGHTS = {
    "High": 1.5,        # High importance level
    "Mid": 1.2,         # Medium importance level
    "Low": 1.0,         # Low importance level
    "Covariate": 0.8,   # Covariate level
    "unknown": 0.5      # Unknown level
}

# Content type weights for retrieval scoring
CONTENT_WEIGHTS = {
    "text": 1.0,        # Base weight for text content
    "image": 0.8,       # Weight for image content
    "table": 1.2,       # Weight for table content
    "graph": 1.5        # Weight for graph context
}

# Default model configurations
DEFAULT_MODEL_CONFIG = {
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "intent_model": "gpt2",
    "spacy_model": "en_core_web_sm",
    "model_params": {
        "text": {
            "max_length": 512,
            "batch_size": 32,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "image": {
            "image_size": 224,
            "batch_size": 16,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    }
}

# Cache configurations
CACHE_CONFIG = {
    "path_cache_size": 1000,
    "max_workers": 4,
    "cache_ttl": 3600,  # Cache time-to-live in seconds
    "cache_dir": "./cache",
    "cache_types": {
        "embeddings": True,
        "graph_paths": True,
        "similarity_scores": True
    },
    "cache_cleanup_interval": 86400  # Cleanup interval in seconds (24 hours)
}

# Retrieval configurations
RETRIEVAL_CONFIG = {
    "default_top_k": 5,
    "max_hops": 2,
    "similarity_threshold": 0.3,
    "scoring": {
        "semantic_weight": 0.6,
        "structural_weight": 0.3,
        "temporal_weight": 0.1
    },
    "ranking": {
        "diversity_penalty": 0.1,
        "relevance_threshold": 0.5,
        "max_results_per_type": 10
    },
    "error_handling": {
        "max_retries": 3,
        "retry_delay": 1.0,
        "fallback_threshold": 0.2
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
        "file": {
            "enabled": True,
            "filename": "retrieval.log",
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5
        },
        "console": {
            "enabled": True
        }
    }
} 