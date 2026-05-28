"""
Formatting utility functions - for formatting user profiles, system signals, etc.
"""
from typing import Dict, Any, List

def format_user_profile(user_profile: Dict[str, Any]) -> str:
    """
    Format user profile

    Args:
        user_profile: User profile dictionary containing category_info and user_profile

    Returns:
        Formatted user profile string
    """
    if not user_profile:
        return ""
    
    category_info = user_profile.get('category_info', {})
    profile_data = user_profile.get('user_profile', {})
    
    formatted = []
    
    # Category info
    if category_info:
        formatted.append(f"Category: {category_info.get('category_name', '')}")
        formatted.append(f"Description: {category_info.get('description', '')}")
    
    # User summary
    if profile_data.get('summary'):
        formatted.append(f"User Summary: {profile_data['summary']}")
    
    # Behavioral patterns
    behavioral_patterns = profile_data.get('behavioralPatterns', {})
    if behavioral_patterns:
        formatted.append("Behavioral Patterns:")
        for key, value in behavioral_patterns.items():
            formatted.append(f"  - {key}: {value}")
    
    # Inferred attributes
    inferred_attributes = profile_data.get('inferredAttributes', {})
    if inferred_attributes:
        formatted.append("Inferred Attributes:")
        for key, values in inferred_attributes.items():
            if isinstance(values, list):
                formatted.append(f"  {key}:")
                for value in values:
                    formatted.append(f"    - {value}")
            else:
                formatted.append(f"  {key}: {values}")
    
    return '\n'.join(formatted)


def format_system_signals_for_user(system_signals: List[Dict[str, Any]]) -> str:
    """
    Format system signals (user perspective)

    Args:
        system_signals: System signals list

    Returns:
        Formatted system signals string
    """
    if not system_signals:
        return ""
    
    formatted = []
    for i, signal in enumerate(system_signals, 1):
        formatted.append(f"Order {i}:")
        formatted.append(f"  Merchant: {signal.get('merchantNameMap', {}).get('en', '')}")
        formatted.append(f"  Food Items: {signal.get('foodInfo', '')}")
        formatted.append(f"  Food Problem: {signal.get('faqType', '')}")
        
        # Handle instant messages
        instant_messages = signal.get('instantMessageMap', '')
        if instant_messages:
            formatted.append(f"  Recent Messages with Courier: {instant_messages}")
        
        # Region info
        metadata = signal.get('metadata', {})
        if metadata.get('region'):
            formatted.append(f"  Region: {metadata['region']}")
    
    return '\n'.join(formatted)


def format_system_signals_for_assistant(system_signals: List[Dict[str, Any]]) -> str:
    """
    Format system signals (customer service perspective)

    Args:
        system_signals: System signals list

    Returns:
        Formatted system signals string
    """
    if not system_signals:
        return ""
    
    signal = system_signals[0]
    formatted = []
    
    if signal.get('instantMessageMap'):
        formatted.append(f"- instantMessageMap: {signal['instantMessageMap']}")
    if signal.get('abnormalReports'):
        formatted.append(f"- abnormalReports: {signal['abnormalReports']}")
    if signal.get('foodInfo'):
        formatted.append(f"- foodInfo: {signal['foodInfo']}")
    if 'rcTag' in signal:
        formatted.append(f"- rcTag: {signal['rcTag']}")
    if signal.get('merchantNameMap'):
        formatted.append(f"- merchantName: {signal['merchantNameMap']['en']}")
    
    return '\n'.join(formatted)


def format_user_system_message(
    user_profile: Dict[str, Any],
    system_signals: List[Dict[str, Any]],
    core_demand: Dict[str, Any],
    prompt_template: str
) -> str:
    """
    Format user system_message

    Args:
        user_profile: User profile
        system_signals: System signals list
        core_demand: Core need
        prompt_template: Prompt template

    Returns:
        Formatted system_message
    """
    # Format user profile
    formatted_user_profile = format_user_profile(user_profile)
    
    # Format system signals
    formatted_system_signals = format_system_signals_for_user(system_signals)
    
    # Replace placeholders in prompt template
    prompt = prompt_template
    prompt = prompt.replace('{{CATEGORY}}', user_profile.get('category_info', {}).get('category_name', ''))
    prompt = prompt.replace('{{USER_PROFILE}}', formatted_user_profile)
    prompt = prompt.replace('{{SYSTEM_SIGNALS}}', formatted_system_signals)
    prompt = prompt.replace('{{CORE_DEMAND}}', core_demand.get('core_need', ''))
    
    return prompt


def format_assistant_system_message(
    system_signals: List[Dict[str, Any]],
    prompt_template: str
) -> str:
    """
    Format customer service system_message

    Args:
        system_signals: System signals list
        prompt_template: Prompt template

    Returns:
        Formatted system_message
    """
    # Format system signals (customer service perspective)
    formatted_signals = format_system_signals_for_assistant(system_signals)
    
    # Replace placeholders in prompt template
    prompt = prompt_template
    prompt = prompt.replace('{{SYSTEM_SIGNALS}}', formatted_signals)
    
    return prompt
