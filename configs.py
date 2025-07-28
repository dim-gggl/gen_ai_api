import os


###########################
# ANTHROPIC
###########################

ANTHROPIC_API_KEY = os.getenv("x_api_key")
ANTHROPIC_MODELS = [
    "claude-3-7-sonnet-latest",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514"
]

###########################
# OPENAI
###########################

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODELS = [
    "o4-mini",
    "gpt-4.1-nano",
    "o3-mini",
    "gpt-image-1"
]

###########################
# XAI
###########################

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_MODELS = [
    "grok-3-mini-fast",
    "grok-4-0709",
    "grok-3-mini",
    "grok-2-image-1212"
]

######1#####################
# GOOGLE
############################

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_MODELS = [
    "veo-3.0-generate-preview", # <--- VIDEO GENERATION
    "veo-2.0-generate-001", # <--- VIDEO GENERATION
    "gemini-2.0-flash-preview-image-generation", # <--- IMAGE GENERATION
    "imagen-4.0-generate-preview-06-06", # <--- IMAGE GENERATION
    "gemini-2.5-flash" # <--- TEXT GENERATION
]