"""
RAGAS Metrics information module - provides descriptions and display names for RAGAS metrics.
This module helps with visualization and reporting of RAGAS evaluation results.
"""

# Dictionary of metric descriptions (short, simple English only)
METRIC_DESCRIPTIONS = {
    "faithfulness": "Measures accuracy of answers based on the context.",
    "answer_relevancy": "Evaluates how well answers address the questions.",
    "context_precision": "Measures relevance of retrieved context.",
    "context_recall": "Evaluates coverage of needed information in context."
}

# Simple display names for metrics (ASCII only)
METRIC_DISPLAY_NAMES = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Relevancy", 
    "context_precision": "Precision",
    "context_recall": "Recall"
}

# Interpretation guide for scores
SCORE_INTERPRETATION = {
    "faithfulness": {
        "excellent": (0.8, 1.0),
        "good": (0.6, 0.8),
        "fair": (0.4, 0.6),
        "poor": (0.0, 0.4)
    },
    "answer_relevancy": {
        "excellent": (0.8, 1.0),
        "good": (0.6, 0.8),
        "fair": (0.4, 0.6),
        "poor": (0.0, 0.4)
    },
    "context_precision": {
        "excellent": (0.8, 1.0),
        "good": (0.6, 0.8),
        "fair": (0.4, 0.6),
        "poor": (0.0, 0.4)
    },
    "context_recall": {
        "excellent": (0.8, 1.0),
        "good": (0.6, 0.8),
        "fair": (0.4, 0.6),
        "poor": (0.0, 0.4)
    }
}

def get_metric_description(metric_name):
    """Get the description for a specific metric"""
    return METRIC_DESCRIPTIONS.get(metric_name, "No description available")

def get_metric_display_name(metric_name):
    """Get the display name for a specific metric"""
    return METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

def get_score_rating(metric_name, score):
    """Get a qualitative rating for a score"""
    if metric_name not in SCORE_INTERPRETATION:
        return "Unknown"
        
    for rating, (min_val, max_val) in SCORE_INTERPRETATION[metric_name].items():
        if min_val <= score <= max_val:
            return rating
            
    return "Unknown"
