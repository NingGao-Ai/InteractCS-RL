"""
Create customer service RL training dataset
"""

import argparse
import os
import json
import random

import pandas as pd


def format_system_signals_for_assistant(system_signals):
    """
    Format system signals (from customer service perspective)
    """
    if not system_signals:
        return ""
    
    signal = system_signals[0] if isinstance(system_signals, list) else system_signals
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


def select_core_need_by_category(core_needs, category_id):
    """
    Select core_need based on category distribution
    """
    try:
        category_idx = int(category_id) - 1
        if category_idx < 0 or category_idx >= 5:
            return random.choice(core_needs)
    except (ValueError, TypeError):
        return random.choice(core_needs)
    
    # Extract the distribution probability for each core need under this category
    distributions = []
    for core_need in core_needs:
        dist_list = core_need.get("distribution", [])
        if category_idx < len(dist_list):
            distributions.append(dist_list[category_idx])
        else:
            distributions.append(0.0)
    
    total = sum(distributions)
    if total == 0:
        return random.choice(core_needs)
    
    # Select core need by weighted distribution
    selected = random.choices(core_needs, weights=distributions, k=1)[0]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/rl_training_data_paper/8000_sample_ordered_no_transfer")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data files
    user_profiles_file = "data/user_roleplay_data/user_roleplay_data_stage_6_user_profiles.json"
    core_need_file = "data/user_roleplay_data/user_roleplay_data_stage_8_core_need.json"
    system_signals_file = "data/user_roleplay_data/user_roleplay_data_stage_7_system_signals_pool.jsonl"
    prompt_file = "code/prompt/paper/customer-service-prompt-general2.txt"

    print("Loading data files...")
    
    # Load user profiles
    with open(user_profiles_file, 'r') as f:
        user_profiles_data = json.load(f)
        user_profiles = user_profiles_data.get('user_profiles', {})
    
    # Load core needs
    with open(core_need_file, 'r') as f:
        core_needs = json.load(f)
    
    # Load system signals
    system_signals = []
    with open(system_signals_file, 'r') as f:
        for line in f:
            if line.strip():
                system_signals.append(json.loads(line))
    
    # Load prompt template
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()

    print(f"Loaded {len(user_profiles)} user profile categories")
    print(f"Loaded {len(core_needs)} core need types")
    print(f"Loaded {len(system_signals)} system signals")

    # Create training samples
    # Category distribution (from default.yaml, must match RLSession config)
    category_distribution = {
        "1": 800,
        "2": 2400,
        "3": 2000,
        "4": 2000,
        "5": 800
    }
    
    # Calculate num_samples from category_distribution
    num_samples = sum(category_distribution.values())
    
    # Compute each category's proportion in a batch
    batch_size = args.train_batch_size
    category_ratio = {cat_id: count / num_samples for cat_id, count in category_distribution.items()}
    samples_per_category_per_batch = {cat_id: int(ratio * batch_size) for cat_id, ratio in category_ratio.items()}

    # Verify that the number of samples in a batch equals batch_size
    total_per_batch = sum(samples_per_category_per_batch.values())
    if total_per_batch != batch_size:
        # Adjust the largest category to ensure total equals batch_size
        max_cat = max(samples_per_category_per_batch, key=samples_per_category_per_batch.get)
        samples_per_category_per_batch[max_cat] += batch_size - total_per_batch
    
    print(f"\nBatch composition (batch_size={batch_size}):")
    for cat_id in sorted(samples_per_category_per_batch.keys()):
        print(f"  Category {cat_id}: {samples_per_category_per_batch[cat_id]} samples per batch")
    
    # Generate samples for each category separately
    category_samples = {cat_id: [] for cat_id in category_distribution.keys()}

    for category_id, count in category_distribution.items():
        if category_id not in user_profiles:
            continue

        category_data = user_profiles[category_id]
        profiles = category_data.get('user_profiles', [])

        # Generate all samples for the current category
        for _ in range(count):
            if not profiles:
                continue
            
            user_profile = random.choice(profiles)
            core_need = select_core_need_by_category(core_needs, category_id)
            system_signal = random.choice(system_signals)
            
            # Format system signals for assistant (customer service)
            formatted_signals = format_system_signals_for_assistant([system_signal])
            
            # Replace placeholder in prompt template
            assistant_prompt = prompt_template.replace('{{SYSTEM_SIGNALS}}', formatted_signals)
            
            sample = {
                "data_source": "customer_service_simulation",
                "prompt": [
                    {
                        "role": "system",
                        "content": assistant_prompt
                    }
                ],
                "reward_model":{
                    "style": "rule", "ground_truth": ""
                },
                "extra_info": {
                    "interaction_kwargs": {
                        "name": "user_simulation",
                        "user_profile": {
                            "category_info": category_data.get('category_info', {}),
                            "user_profile": user_profile
                        },
                        "core_demand": core_need,
                        "system_signal": system_signal,
                    }
                }
            }
            
            category_samples[category_id].append(sample)
        
        # Shuffle samples for the current category
        random.shuffle(category_samples[category_id])

    # Build training set by batch_size, ensuring uniform category distribution within each batch
    train_data = []
    train_idx = 0
    train_batch_id = 0

    # Build samples for each batch, only keeping complete batches
    batch_num = 0
    while True:
        batch_samples_list = []
        can_form_complete_batch = True
        
        # Check if a complete batch can be formed
        for category_id in sorted(category_distribution.keys()):
            samples_needed = samples_per_category_per_batch[category_id]

            # Take the required number of samples from this category
            batch_start_idx = batch_num * samples_needed
            batch_end_idx = batch_start_idx + samples_needed

            # If this category does not have enough samples, cannot form a complete batch
            if batch_end_idx > len(category_samples[category_id]):
                can_form_complete_batch = False
                break
            
            batch_samples = category_samples[category_id][batch_start_idx:batch_end_idx]
            batch_samples_list.append(batch_samples)
        
        # If a complete batch cannot be formed, stop
        if not can_form_complete_batch:
            break
        
        # Add all samples for this batch
        for batch_samples in batch_samples_list:
            for sample in batch_samples:
                sample["extra_info"]["interaction_kwargs"]["split"] = "train"
                sample["extra_info"]["interaction_kwargs"]["batch_id"] = train_batch_id
                sample["extra_info"]["interaction_kwargs"]["index"] = train_idx
                train_idx += 1
                train_data.append(sample)
        
        train_batch_id += 1
        batch_num += 1
    
    # Build test set using the same batch strategy (built independently, not split from training set)
    test_batch_size = args.test_batch_size
    test_category_ratio = {cat_id: count / num_samples for cat_id, count in category_distribution.items()}
    test_samples_per_category_per_batch = {cat_id: int(ratio * test_batch_size) for cat_id, ratio in test_category_ratio.items()}
    
    # Verify that the number of samples in a test batch equals test_batch_size
    test_total_per_batch = sum(test_samples_per_category_per_batch.values())
    if test_total_per_batch != test_batch_size:
        # Adjust the largest category to ensure total equals test_batch_size
        max_cat = max(test_samples_per_category_per_batch, key=test_samples_per_category_per_batch.get)
        test_samples_per_category_per_batch[max_cat] += test_batch_size - test_total_per_batch
    
    print(f"\nTest batch composition (batch_size={test_batch_size}):")
    for cat_id in sorted(test_samples_per_category_per_batch.keys()):
        print(f"  Category {cat_id}: {test_samples_per_category_per_batch[cat_id]} samples per batch")
    
    # Generate test set samples (only generate enough for one batch)
    test_data = []
    test_idx = 0
    
    for category_id in sorted(category_distribution.keys()):
        if category_id not in user_profiles:
            continue
        
        category_data = user_profiles[category_id]
        profiles = category_data.get('user_profiles', [])
        samples_needed = test_samples_per_category_per_batch[category_id]
        
        # Generate the required number of samples for the current category
        for _ in range(samples_needed):
            if not profiles:
                continue
            
            user_profile = random.choice(profiles)
            core_need = select_core_need_by_category(core_needs, category_id)
            system_signal = random.choice(system_signals)
            
            # Format system signals for assistant (customer service)
            formatted_signals = format_system_signals_for_assistant([system_signal])
            
            # Replace placeholder in prompt template
            assistant_prompt = prompt_template.replace('{{SYSTEM_SIGNALS}}', formatted_signals)
            
            sample = {
                "data_source": "customer_service_simulation",
                "prompt": [
                    {
                        "role": "system",
                        "content": assistant_prompt
                    }
                ],
                "reward_model":{
                    "style": "rule", "ground_truth": ""
                },
                "extra_info": {
                    "interaction_kwargs": {
                        "name": "user_simulation",
                        "user_profile": {
                            "category_info": category_data.get('category_info', {}),
                            "user_profile": user_profile
                        },
                        "core_demand": core_need,
                        "system_signal": system_signal,
                        "split": "test",
                        "batch_id": 0,
                        "index": test_idx
                    },
                }
            }
            
            test_idx += 1
            test_data.append(sample)

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Handle HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: HDFS support not available. Skipping HDFS copy.")

    # Print statistics
    print(f"\nDataset created successfully!")
    print(f"Total samples: {num_samples}")
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {local_dir}")
    print(f"\nCategory distribution:")
    for cat_id, count in category_distribution.items():
        print(f"  Category {cat_id}: {count} samples")
    
    # Print training set batch distribution
    print(f"\nTrain set batch distribution:")
    train_batch_counts = {}
    train_batch_category_counts = {}
    for sample in train_data:
        batch_id = sample["extra_info"]["interaction_kwargs"]["batch_id"]
        cat_info = sample["extra_info"]["interaction_kwargs"]["user_profile"]["category_info"]
        cat_key = cat_info.get("category_key", "unknown")
        
        if batch_id not in train_batch_counts:
            train_batch_counts[batch_id] = 0
            train_batch_category_counts[batch_id] = {}
        
        train_batch_counts[batch_id] += 1
        train_batch_category_counts[batch_id][cat_key] = train_batch_category_counts[batch_id].get(cat_key, 0) + 1
    
    print(f"  Total batches: {len(train_batch_counts)}")
    print(f"  Samples per batch: {train_batch_counts[0] if train_batch_counts else 0}")
    print(f"  Sample batch 0 category distribution: {train_batch_category_counts.get(0, {})}")
    
    # Print test set batch distribution
    print(f"\nTest set batch distribution:")
    test_batch_counts = {}
    test_batch_category_counts = {}
    for sample in test_data:
        batch_id = sample["extra_info"]["interaction_kwargs"]["batch_id"]
        cat_info = sample["extra_info"]["interaction_kwargs"]["user_profile"]["category_info"]
        cat_key = cat_info.get("category_key", "unknown")
        
        if batch_id not in test_batch_counts:
            test_batch_counts[batch_id] = 0
            test_batch_category_counts[batch_id] = {}
        
        test_batch_counts[batch_id] += 1
        test_batch_category_counts[batch_id][cat_key] = test_batch_category_counts[batch_id].get(cat_key, 0) + 1
    
    print(f"  Total batches: {len(test_batch_counts)}")
    print(f"  Samples per batch: {test_batch_counts[0] if test_batch_counts else 0}")
    print(f"  Sample batch 0 category distribution: {test_batch_category_counts.get(0, {})}")
    
    # Print the first two training samples
    print(f"\n{'='*80}")
    print("Train Sample 1:")
    print(f"{'='*80}")
    print(json.dumps(train_data[0], indent=2, ensure_ascii=False))

    print(f"\n{'='*80}")
    print("Train Sample 2:")
    print(f"{'='*80}")
    print(json.dumps(train_data[1], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
