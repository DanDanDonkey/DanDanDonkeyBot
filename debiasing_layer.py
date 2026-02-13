# debiasing_layer.py

"""
Debiasing layer that identifies and corrects systematic biases in LLM forecasts.
Runs after each forecaster generates a prediction but before ensemble aggregation.
Based on Tetlock's Superforecasting research and Actively Open-Minded Thinking (AOT).
"""

from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

# Debiasing prompts targeting specific cognitive biases
DEBIASING_PROMPTS = {
    "availability_bias": """
You may be overweighting recent/memorable events. 
List 3 relevant precedents or data points NOT covered in recent news that could affect this forecast.
After considering this, should you adjust your forecast?
""",
    
    "status_quo_bias": """
You may be assuming current trends continue by default.
Describe a realistic scenario where the OPPOSITE outcome occurs. What would trigger it?
After considering this, should you adjust your forecast?
""",
    
    "narrative_bias": """
You may be fitting a compelling story rather than following the data.
What specific data points or statistics CONTRADICT your main narrative?
After considering this, should you adjust your forecast?
""",
    
    "anchoring_bias": """
You may be anchored to the base rate or first number you saw.
If you had to forecast this WITHOUT seeing any initial numbers, what would you predict? 
Why might it differ from your original forecast?
""",
    
    "groupthink_bias": """
You may be converging on consensus views.
What would a smart contrarian expert argue? What unpopular position might be correct?
After considering this, should you adjust your forecast?
"""
}

# Actively Open-Minded Thinking (AOT) scoring criteria
AOT_CRITERIA = {
    "considered_alternatives": "Did the forecast seriously consider alternative outcomes?",
    "acknowledged_uncertainty": "Did it acknowledge key uncertainties and unknowns?",
    "sought_disconfirming": "Did it actively look for evidence against its position?",
    "updated_from_initial": "Did reasoning evolve/update during the analysis?",
    "avoided_overconfidence": "Did it avoid extreme probabilities without strong justification?"
}


async def debias_forecast(
    forecast_dict: Dict,
    question_text: str,
    llm_client
) -> Dict:
    """
    Apply debiasing layer to a single forecaster's output.
    
    Args:
        forecast_dict: Dict with 'prediction', 'reasoning', 'confidence' (or similar structure)
        question_text: The forecast question text
        llm_client: LLM client to use for debiasing prompts (should have .invoke() method)
    
    Returns:
        Dict with original forecast + debiasing analysis + adjusted prediction
    """
    
    # Extract prediction value (handle different formats)
    original_prediction = extract_prediction_value(forecast_dict)
    original_reasoning = forecast_dict.get('reasoning', str(forecast_dict))
    
    logger.info(f"Debiasing forecast with original prediction: {original_prediction}")
    
    # Run all debiasing checks
    debiasing_responses = {}
    for bias_name, prompt_template in DEBIASING_PROMPTS.items():
        full_prompt = f"""
Original forecast question: {question_text}

Your initial prediction: {original_prediction}%
Your reasoning: {original_reasoning[:1000]}

{prompt_template}

Provide your response in this format:
ADJUSTED: [yes/no]
NEW_PREDICTION: [number between 0-100]%
REASON: [brief explanation]
"""
        
        try:
            response = await llm_client.invoke(full_prompt)
            debiasing_responses[bias_name] = response
            logger.debug(f"Debiasing response for {bias_name}: {response[:200]}")
        except Exception as e:
            logger.warning(f"Debiasing check {bias_name} failed: {e}")
            debiasing_responses[bias_name] = "ADJUSTED: no\nNEW_PREDICTION: " + str(original_prediction) + "%\nREASON: Error occurred"
    
    # Score Actively Open-Minded Thinking
    aot_score = await score_aot(original_reasoning, llm_client)
    
    # Calculate adjusted prediction based on debiasing insights
    adjusted_prediction = calculate_debiased_prediction(
        original_prediction,
        debiasing_responses
    )
    
    return {
        'original_prediction': original_prediction,
        'adjusted_prediction': adjusted_prediction,
        'adjustment_magnitude': abs(adjusted_prediction - original_prediction),
        'debiasing_insights': debiasing_responses,
        'aot_score': aot_score,
        'full_reasoning': original_reasoning
    }


def extract_prediction_value(forecast: Dict) -> float:
    """Extract numeric prediction from various forecast formats."""
    # Try common keys
    for key in ['prediction', 'probability', 'forecast', 'value', 'prediction_value']:
        if key in forecast:
            val = forecast[key]
            if isinstance(val, (int, float)):
                return float(val)
            # Handle percentage strings like "45%" or "0.45"
            if isinstance(val, str):
                val = val.replace('%', '').strip()
                try:
                    num = float(val)
                    # If < 1, assume it's a probability (0.45), convert to percent
                    return num * 100 if num <= 1 else num
                except:
                    pass
    
    # If no standard key found, try to parse the whole thing
    forecast_str = str(forecast)
    import re
    match = re.search(r'(\d+\.?\d*)\s*%', forecast_str)
    if match:
        return float(match.group(1))
    
    # Default to 50% if can't parse
    logger.warning(f"Could not extract prediction from {forecast}, defaulting to 50%")
    return 50.0


async def score_aot(reasoning: str, llm_client) -> Dict[str, float]:
    """
    Score the reasoning on Actively Open-Minded Thinking criteria.
    Returns scores 0-1 for each criterion.
    """
    
    scores = {}
    for criterion, description in AOT_CRITERIA.items():
        prompt = f"""
Evaluate this forecast reasoning on the following criterion:
{description}

Reasoning to evaluate:
{reasoning[:1500]}

Score from 0.0 (not at all) to 1.0 (exemplary). 
Respond with ONLY a number between 0.0 and 1.0, nothing else.
"""
        
        try:
            response = await llm_client.invoke(prompt)
            # Parse score from response
            import re
            match = re.search(r'(\d+\.?\d*)', response.strip())
            if match:
                score = float(match.group(1))
                scores[criterion] = max(0.0, min(1.0, score))
            else:
                scores[criterion] = 0.5  # Default if parsing fails
        except Exception as e:
            logger.warning(f"AOT scoring for {criterion} failed: {e}")
            scores[criterion] = 0.5
    
    scores['overall_aot'] = sum(scores.values()) / len(scores)
    logger.info(f"AOT scores: {scores}")
    return scores


def calculate_debiased_prediction(
    original: float,
    debiasing_responses: Dict[str, str]
) -> float:
    """
    Aggregate debiasing insights into an adjusted prediction.
    Uses weighted averaging of suggested adjustments.
    """
    
    adjustments = [original]  # Start with original (gets weight in average)
    
    for bias_name, response in debiasing_responses.items():
        # Parse if there's a new prediction suggested
        if "ADJUSTED: yes" in response or "ADJUSTED: Yes" in response:
            try:
                # Extract NEW_PREDICTION: X%
                import re
                match = re.search(r'NEW_PREDICTION:\s*(\d+\.?\d*)\s*%?', response)
                if match:
                    new_pred = float(match.group(1))
                    # Sanity check: keep within 0-100
                    if 0 <= new_pred <= 100:
                        adjustments.append(new_pred)
                        logger.info(f"Bias {bias_name} suggests adjustment to {new_pred}%")
            except Exception as e:
                logger.debug(f"Could not parse adjustment from {bias_name}: {e}")
                continue
    
    # Return average of all predictions (original + debiased suggestions)
    debiased = sum(adjustments) / len(adjustments)
    logger.info(f"Debiased prediction: {original}% -> {debiased}% (based on {len(adjustments)} values)")
    return debiased


async def debias_all_forecasters(
    ensemble_forecasts: List[Dict],
    question_text: str,
    llm_client
) -> List[Dict]:
    """
    Apply debiasing to all forecasters in the ensemble.
    
    Args:
        ensemble_forecasts: List of forecast dicts from different forecasters
        question_text: The question being forecasted
        llm_client: LLM client with .invoke() method
    
    Returns:
        List of debiased forecast dicts
    """
    
    logger.info(f"Debiasing {len(ensemble_forecasts)} forecasts for question: {question_text[:100]}")
    
    debiased_forecasts = []
    
    for i, forecast in enumerate(ensemble_forecasts):
        logger.info(f"Debiasing forecaster {i+1}/{len(ensemble_forecasts)}")
        try:
            debiased = await debias_forecast(forecast, question_text, llm_client)
            debiased_forecasts.append(debiased)
        except Exception as e:
            logger.error(f"Failed to debias forecaster {i+1}: {e}")
            # Fall back to original forecast
            debiased_forecasts.append({
                'original_prediction': extract_prediction_value(forecast),
                'adjusted_prediction': extract_prediction_value(forecast),
                'adjustment_magnitude': 0,
                'debiasing_insights': {},
                'aot_score': {},
                'full_reasoning': str(forecast)
            })
    
    return debiased_forecasts


def analyze_bias_patterns(debiased_forecasts: List[Dict]) -> Dict:
    """
    Analyze which biases were most prevalent across forecasters.
    Useful for post-mortems and learning.
    """
    
    bias_adjustment_counts = {bias: 0 for bias in DEBIASING_PROMPTS.keys()}
    total_adjustments = 0
    avg_adjustment_magnitude = 0
    aot_scores_sum = {criterion: 0.0 for criterion in AOT_CRITERIA.keys()}
    aot_scores_sum['overall_aot'] = 0.0
    
    for forecast in debiased_forecasts:
        for bias_name, response in forecast.get('debiasing_insights', {}).items():
            if "ADJUSTED: yes" in response.lower():
                bias_adjustment_counts[bias_name] += 1
                total_adjustments += 1
        
        avg_adjustment_magnitude += forecast.get('adjustment_magnitude', 0)
        
        # Aggregate AOT scores
        for criterion, score in forecast.get('aot_score', {}).items():
            aot_scores_sum[criterion] += score
    
    n_forecasts = len(debiased_forecasts) if debiased_forecasts else 1
    avg_adjustment_magnitude /= n_forecasts
    
    avg_aot_scores = {k: v / n_forecasts for k, v in aot_scores_sum.items()}
    
    most_common_bias = max(bias_adjustment_counts, key=bias_adjustment_counts.get) if bias_adjustment_counts else "none"
    
    return {
        'bias_frequency': bias_adjustment_counts,
        'total_adjustments': total_adjustments,
        'avg_adjustment_magnitude': avg_adjustment_magnitude,
        'most_common_bias': most_common_bias,
        'avg_aot_scores': avg_aot_scores
    }


def format_debiasing_summary(analysis: Dict) -> str:
    """Format bias analysis for logging/reporting."""
    
    summary = "=== DEBIASING ANALYSIS ===\n"
    summary += f"Total bias adjustments triggered: {analysis['total_adjustments']}\n"
    summary += f"Average adjustment magnitude: {analysis['avg_adjustment_magnitude']:.1f}%\n"
    summary += f"Most common bias: {analysis['most_common_bias']}\n\n"
    
    summary += "Bias frequency:\n"
    for bias, count in analysis['bias_frequency'].items():
        summary += f"  - {bias}: {count}\n"
    
    summary += "\nAverage AOT scores:\n"
    for criterion, score in analysis['avg_aot_scores'].items():
        summary += f"  - {criterion}: {score:.2f}\n"
    
    return summary


# Example usage in your pipeline:
"""
# In your main forecasting pipeline:

# After generating your 5 ensemble forecasts
ensemble_forecasts = [
    {'prediction': 45.0, 'reasoning': '...'},
    {'prediction': 52.0, 'reasoning': '...'},
    # ... 3 more
]

# Apply debiasing layer
debiased_forecasts = await debias_all_forecasters(
    ensemble_forecasts, 
    question_text, 
    llm_client=your_llm_client
)

# Extract adjusted predictions for final ensemble
adjusted_predictions = [f['adjusted_prediction'] for f in debiased_forecasts]
final_prediction = sum(adjusted_predictions) / len(adjusted_predictions)

# Store bias analysis for post-mortems
bias_analysis = analyze_bias_patterns(debiased_forecasts)
print(format_debiasing_summary(bias_analysis))
"""
