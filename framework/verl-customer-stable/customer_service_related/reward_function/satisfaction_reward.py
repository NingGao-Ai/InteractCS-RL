def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute customer service satisfaction reward score

    Args:
        data_source: Data source identifier
        solution_str: Model-generated response
        ground_truth: Dictionary containing user_turn_rewards
        extra_info: Extra information (optional)

    Returns:
        float: Reward score, typically in range [0, 1] or other
    """
    try:
        # Check if ground_truth is a dictionary
        if not isinstance(ground_truth, dict):
            print(f"[Warning] ground_truth is not a dict, got {type(ground_truth)}: {ground_truth}")
            return 0.0
        
        # Get user_turn_rewards list
        user_turn_rewards = ground_truth.get('user_turn_rewards', [])
        
        # Check if empty
        if not user_turn_rewards:
            return 0.0
        
        # Get the last item and convert to float
        last_reward = user_turn_rewards[-1]
        score = float(last_reward)
        
        return score
        
    except (TypeError, ValueError, AttributeError) as e:
        print(f"[Error] Failed to compute score from ground_truth: {e}")
        print(f"[Debug] ground_truth type: {type(ground_truth)}, value: {ground_truth}")
        return 0.0
    except Exception as e:
        print(f"[Error] Unexpected error in compute_score: {e}")
        return 0.0
