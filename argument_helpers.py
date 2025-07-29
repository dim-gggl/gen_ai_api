 """Helpers pour les arguments communs utilisés dans différents modules."""

import os
from typing import Dict, Any, List

# Arguments communs par catégorie
COMMON_API_ARGS = {
    "api_key": {
        "flags": ("--api-key",),
        "help": "API key for authentication",
        "default": None
    },
    "api_host": {
        "flags": ("--api-host",),
        "help": "API host URL",
        "default": None
    },
    "timeout": {
        "flags": ("--timeout",),
        "type": float,
        "default": 60.0,
        "help": "Request timeout in seconds"
    }
}

COMMON_MODEL_ARGS = {
    "model": {
        "flags": ("--model", "-M"),
        "help": "Model to use",
        "default": None
    },
    "temperature": {
        "flags": ("--temperature", "-t"),
        "type": float,
        "default": 0.7,
        "help": "Sampling temperature (0.0 - 1.0)"
    },
    "max_tokens": {
        "flags": ("--max-tokens",),
        "type": int,
        "default": 1024,
        "help": "Maximum number of tokens to generate"
    }
}

COMMON_OUTPUT_ARGS = {
    "output": {
        "flags": ("--output", "-o"),
        "help": "Output file path",
        "default": None
    },
    "format": {
        "flags": ("--format", "-f"),
        "help": "Output format",
        "default": None
    }
}

def add_common_args(parser, arg_categories: List[str]) -> None:
    """Ajoute des arguments communs à un parser.
    
    Args:
        parser: ArgumentParser instance
        arg_categories: List of categories to add (e.g., ['api', 'model', 'output'])
    """
    all_args = {}
    
    if 'api' in arg_categories:
        all_args.update(COMMON_API_ARGS)
    if 'model' in arg_categories:
        all_args.update(COMMON_MODEL_ARGS)
    if 'output' in arg_categories:
        all_args.update(COMMON_OUTPUT_ARGS)
    
    for arg_name, arg_spec in all_args.items():
        parser.add_argument(*arg_spec["flags"], **{k: v for k, v in arg_spec.items() if k != "flags"})

def get_api_key_from_env(env_var: str, default: str = None) -> str:
    """Récupère une clé API depuis les variables d'environnement."""
    return os.getenv(env_var, default)