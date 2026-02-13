# question_decomposer.py

"""
Question Decomposer - Breaks complex forecasting questions into simpler sub-questions.
Uses Fermi estimation methodology to decompose questions into independent components.
Based on LessWrong Fermi modeling principles and Philip Tetlock's decomposition research.
"""

import logging
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)


async def decompose_question(
    question_text: str,
    question_type: str,
    llm_client,
    max_subquestions: int = 5
) -> Dict:
    """
    Decompose a complex question into simpler sub-questions using Fermi methodology.
    
    Args:
        question_text: The main forecasting question
        question_type: Type of question (binary, numeric, multiple_choice, date)
        llm_client: LLM client with .invoke() method
        max_subquestions: Maximum number of sub-questions to generate
    
    Returns:
        Dict with 'subquestions', 'decomposition_method', 'combination_formula'
    """
    
    logger.info(f"Decomposing question: {question_text[:100]}")
    
    decomposition_prompt = f"""
You are a professional forecaster using Fermi estimation methodology.

Main question: {question_text}
Question type: {question_type}

Break this down into {max_subquestions} or fewer independent sub-questions that:
1. Are easier to research/forecast individually
2. Can be combined to answer the main question
3. Avoid double-counting or overlap
4. Cover the key causal factors

For each sub-question, specify:
- The sub-question text
- Why it matters to the main question
- Approximate base rate or typical value (if known)

Then provide a formula/method for combining sub-question answers into the main answer.

Examples of good decomposition:

Question: "Will China invade Taiwan by 2030?"
Sub-questions:
1. Does China have military capability to invade Taiwan? (P_capability)
2. Does Chinese leadership have political will to invade? (P_will)
3. Would US intervene militarily? (P_us_intervene)
4. What is annual baseline risk of conflict? (P_baseline_annual)
5. How many years until 2030? (years)

Formula: P_invasion ≈ 1 - (1 - P_baseline_annual)^years * P_capability * P_will * (1 - P_us_intervene * deterrence_factor)

Question: "What will Tesla's stock price be in 2026?"
Sub-questions:
1. What is current Tesla stock price? (current_price)
2. What is expected EV market growth rate? (market_growth)
3. What is Tesla's expected market share change? (share_change)
4. What is expected P/E ratio change? (pe_change)
5. What is the volatility/uncertainty factor? (volatility)

Formula: predicted_price = current_price * (1 + market_growth) * (1 + share_change) * (1 + pe_change) * uncertainty_range

Now decompose: {question_text}

Respond in this JSON format:
{{
  "subquestions": [
    {{"text": "...", "reasoning": "...", "base_rate": "..."}},
    ...
  ],
  "decomposition_method": "multiplicative/additive/conditional/custom",
  "combination_formula": "P_main = ...",
  "assumptions": ["list key assumptions made"]
}}
"""
    
    try:
        response = await llm_client.invoke(decomposition_prompt)
        logger.debug(f"Decomposition response: {response[:500]}")
        
        # Parse JSON from response
        decomposition = parse_decomposition_response(response)
        
        logger.info(f"Generated {len(decomposition['subquestions'])} sub-questions")
        return decomposition
        
    except Exception as e:
        logger.error(f"Question decomposition failed: {e}")
        # Return fallback with original question
        return {
            "subquestions": [{
                "text": question_text,
                "reasoning": "Decomposition failed, using original question",
                "base_rate": "unknown"
            }],
            "decomposition_method": "none",
            "combination_formula": "direct",
            "assumptions": ["No decomposition performed"]
        }


def parse_decomposition_response(response: str) -> Dict:
    """Parse LLM response into structured decomposition."""
    
    # Try to extract JSON from response
    import re
    
    # Look for JSON block
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Fallback: manual parsing
    subquestions = []
    lines = response.split('\n')
    
    current_subq = None
    for line in lines:
        # Look for numbered sub-questions
        match = re.match(r'^\d+\.\s*(.+)', line.strip())
        if match:
            if current_subq:
                subquestions.append(current_subq)
            current_subq = {
                "text": match.group(1),
                "reasoning": "",
                "base_rate": "unknown"
            }
        elif current_subq and ('reasoning:' in line.lower() or 'why:' in line.lower()):
            current_subq["reasoning"] = line.split(':', 1)[-1].strip()
        elif current_subq and ('base' in line.lower() and 'rate' in line.lower()):
            current_subq["base_rate"] = line.split(':', 1)[-1].strip()
    
    if current_subq:
        subquestions.append(current_subq)
    
    # Extract formula if present
    formula_match = re.search(r'(?:formula|combination)[:\s]*(.+)', response, re.IGNORECASE)
    combination_formula = formula_match.group(1).strip() if formula_match else "combine sub-forecasts"
    
    return {
        "subquestions": subquestions if subquestions else [{"text": "Parse failed", "reasoning": "", "base_rate": ""}],
        "decomposition_method": "inferred",
        "combination_formula": combination_formula,
        "assumptions": ["Parsed from unstructured response"]
    }


async def forecast_subquestions(
    decomposition: Dict,
    research_context: str,
    llm_client
) -> List[Dict]:
    """
    Forecast each sub-question independently.
    
    Returns:
        List of dicts with 'subquestion', 'forecast', 'reasoning'
    """
    
    subquestion_forecasts = []
    
    for i, subq_dict in enumerate(decomposition['subquestions']):
        subq_text = subq_dict['text']
        base_rate = subq_dict.get('base_rate', 'unknown')
        
        logger.info(f"Forecasting sub-question {i+1}/{len(decomposition['subquestions'])}: {subq_text[:80]}")
        
        forecast_prompt = f"""
You are forecasting a sub-component of a larger question.

Sub-question: {subq_text}
Base rate (if known): {base_rate}
Why this matters: {subq_dict.get('reasoning', 'Key factor')}

Research context:
{research_context[:2000]}

Provide:
1. Your best estimate (as probability % or numeric value)
2. Confidence interval (if applicable)
3. Key evidence used
4. Uncertainty factors

Format response as:
ESTIMATE: [value]
CONFIDENCE: [low/medium/high]
REASONING: [brief explanation]
"""
        
        try:
            response = await llm_client.invoke(forecast_prompt)
            
            # Parse estimate from response
            import re
            estimate_match = re.search(r'ESTIMATE:\s*(\d+\.?\d*)\s*%?', response)
            estimate = float(estimate_match.group(1)) if estimate_match else 50.0
            
            confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', response)
            confidence = confidence_match.group(1).lower() if confidence_match else "medium"
            
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response[:500]
            
            subquestion_forecasts.append({
                'subquestion': subq_text,
                'estimate': estimate,
                'confidence': confidence,
                'reasoning': reasoning,
                'base_rate': base_rate
            })
            
        except Exception as e:
            logger.error(f"Sub-question forecast failed for '{subq_text[:50]}': {e}")
            subquestion_forecasts.append({
                'subquestion': subq_text,
                'estimate': 50.0,
                'confidence': 'low',
                'reasoning': f'Error: {str(e)}',
                'base_rate': base_rate
            })
    
    return subquestion_forecasts


def combine_subforecasts(
    subforecasts: List[Dict],
    decomposition: Dict,
    llm_client=None
) -> Dict:
    """
    Combine sub-question forecasts into final forecast using specified method.
    
    Returns:
        Dict with 'final_forecast', 'method', 'component_contributions'
    """
    
    method = decomposition.get('decomposition_method', 'multiplicative')
    formula = decomposition.get('combination_formula', '')
    
    logger.info(f"Combining {len(subforecasts)} sub-forecasts using {method} method")
    
    # Extract numeric estimates
    estimates = [sf['estimate'] for sf in subforecasts]
    
    if method == 'multiplicative' or 'multiplicative' in formula.lower():
        # Convert to probabilities (0-1) and multiply
        probs = [e/100 if e > 1 else e for e in estimates]
        final = (sum(probs) / len(probs)) * 100  # Average of probabilities as rough combination
        
    elif method == 'additive' or 'additive' in formula.lower():
        # Simple average
        final = sum(estimates) / len(estimates)
        
    elif method == 'conditional':
        # Weighted by confidence
        weights = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        weighted_sum = sum(sf['estimate'] * weights.get(sf['confidence'], 0.7) for sf in subforecasts)
        total_weight = sum(weights.get(sf['confidence'], 0.7) for sf in subforecasts)
        final = weighted_sum / total_weight
        
    else:
        # Default: simple average
        final = sum(estimates) / len(estimates)
    
    # Calculate component contributions
    contributions = []
    for sf in subforecasts:
        contribution = {
            'subquestion': sf['subquestion'][:80],
            'estimate': sf['estimate'],
            'weight': sf['confidence'],
            'impact': (sf['estimate'] - 50.0)  # Deviation from neutral
        }
        contributions.append(contribution)
    
    return {
        'final_forecast': final,
        'method': method,
        'formula': formula,
        'component_contributions': contributions,
        'subforecasts': subforecasts
    }


async def decompose_and_forecast(
    question_text: str,
    question_type: str,
    research_context: str,
    llm_client,
    max_subquestions: int = 5
) -> Dict:
    """
    Full pipeline: decompose question, forecast sub-questions, combine results.
    
    Returns:
        Complete decomposition with final forecast
    """
    
    logger.info(f"Starting full decomposition pipeline for: {question_text[:100]}")
    
    # Step 1: Decompose
    decomposition = await decompose_question(
        question_text,
        question_type,
        llm_client,
        max_subquestions
    )
    
    # Step 2: Forecast sub-questions
    subforecasts = await forecast_subquestions(
        decomposition,
        research_context,
        llm_client
    )
    
    # Step 3: Combine
    combined = combine_subforecasts(
        subforecasts,
        decomposition,
        llm_client
    )
    
    # Add decomposition info to result
    result = {
        'original_question': question_text,
        'decomposition': decomposition,
        'final_forecast': combined['final_forecast'],
        'combination_method': combined['method'],
        'subforecasts': combined['subforecasts'],
        'component_contributions': combined['component_contributions']
    }
    
    logger.info(f"Decomposition complete. Final forecast: {combined['final_forecast']:.1f}%")
    
    return result


def format_decomposition_summary(result: Dict) -> str:
    """Format decomposition results for logging/reporting."""
    
    summary = "=== QUESTION DECOMPOSITION ===\n"
    summary += f"Original: {result['original_question']}\n"
    summary += f"Method: {result['combination_method']}\n"
    summary += f"Final Forecast: {result['final_forecast']:.1f}%\n\n"
    
    summary += "Sub-questions and forecasts:\n"
    for i, sf in enumerate(result['subforecasts'], 1):
        summary += f"{i}. {sf['subquestion']}\n"
        summary += f"   Estimate: {sf['estimate']:.1f}%\n"
        summary += f"   Confidence: {sf['confidence']}\n"
        summary += f"   Base rate: {sf['base_rate']}\n\n"
    
    summary += "Component contributions:\n"
    for contrib in result['component_contributions']:
        impact_sign = '+' if contrib['impact'] > 0 else ''
        summary += f"  • {contrib['subquestion']}: {contrib['estimate']:.1f}% ({impact_sign}{contrib['impact']:.1f})\n"
    
    return summary


# Example usage:
"""
# In your forecasting pipeline:

decomposition_result = await decompose_and_forecast(
    question_text="Will China invade Taiwan by 2030?",
    question_type="binary",
    research_context=your_research_summary,
    llm_client=your_llm_client,
    max_subquestions=5
)

print(format_decomposition_summary(decomposition_result))

# Use the final_forecast in your ensemble:
decomposed_forecast = decomposition_result['final_forecast']
"""
