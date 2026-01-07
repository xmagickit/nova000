import os
from traceback import print_exc
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pandas as pd
from pathlib import Path
import nova_ph2
from itertools import combinations

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening
from molecules import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
    compute_tanimoto_similarity_to_pool,
    sample_random_valid_molecules,
    compute_maccs_entropy,
    SynthonLibrary,
    generate_molecules_from_synthon_library,
    validate_molecules,
)

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")


target_models = []
antitarget_models = []

def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


def initialize_models(config: dict):
    """Initialize separate model instances for each target and antitarget sequence."""
    global target_models, antitarget_models
    target_models = []
    antitarget_models = []
    
    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)
    
    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)


# ---------- scoring helpers (reuse pre-initialized models) ----------
def target_score_from_data(data: pd.Series):
    """Score molecules against all target models."""
    global target_models, antitarget_models
    try:
        target_scores = []
        smiles_list = data.tolist()
        for target_model in target_models:
            scores = target_model.score_molecules(smiles_list)
            for antitarget_model in antitarget_models:
                antitarget_model.smiles_list = smiles_list
                antitarget_model.smiles_dict = target_model.smiles_dict

            scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
            target_scores.append(scores["target"])
        # Average across all targets
        target_series = pd.DataFrame(target_scores).mean(axis=0)
        return target_series
    except Exception as e:
        bt.logging.error(f"Target scoring error: {e}")
        return pd.Series(dtype=float)


def antitarget_scores():
    """Score molecules against all antitarget models."""
    
    global antitarget_models
    try:
        antitarget_scores = []
        for i, antitarget_model in enumerate(antitarget_models):
            antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
            antitarget_model.screen_df = virtual_screening(antitarget_model.screen_df, 
                                            antitarget_model.model, 
                                            antitarget_model.screen_loader,
                                            os.getcwd(),
                                            save_interpret=False,
                                            ligand_dict=antitarget_model.smiles_dict, 
                                            device=antitarget_model.device,
                                            save_cluster=False,
                                            )
            scores = antitarget_model.screen_df[['predicted_binding_affinity']]
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])
        
        if not antitarget_scores:
            return pd.Series(dtype=float)
        
        # average across antitargets
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        bt.logging.error(f"Antitarget scoring error: {e}")
        return pd.Series(dtype=float)


def _cpu_random_candidates_with_similarity(
    iteration: int,
    n_samples: int,
    subnet_config: dict,
    top_pool_df: pd.DataFrame,
    avoid_inchikeys: set[str] | None = None,
    thresh: float = 0.8
) -> pd.DataFrame:
    """
    CPU-side helper:
    - draws a random batch of valid molecules (independent of the GPU batch),
    - computes Tanimoto similarity vs. current top_pool,
    - returns a DataFrame with name, smiles, InChIKey, tanimoto_similarity.
    """
    try:
        random_df = sample_random_valid_molecules(
            n_samples=n_samples,
            subnet_config=subnet_config,
            avoid_inchikeys=avoid_inchikeys,
            focus_neighborhood_of=top_pool_df
        )
        if random_df.empty or top_pool_df.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

        sims = compute_tanimoto_similarity_to_pool(
            candidate_smiles=random_df["smiles"],
            pool_smiles=top_pool_df["smiles"],
        )
        random_df = random_df.copy()
        random_df["tanimoto_similarity"] = sims.reindex(random_df.index).fillna(0.0)
        random_df = random_df.sort_values(by="tanimoto_similarity", ascending=False)
        random_df_filtered = random_df[random_df["tanimoto_similarity"] >= thresh]
            
        if random_df_filtered.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
            
        random_df_filtered = random_df_filtered.reset_index(drop=True)
        return random_df_filtered[["name", "smiles", "InChIKey"]]
    except Exception as e:
        bt.logging.warning(f"[Miner] _cpu_random_candidates_with_similarity failed: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

def select_diverse_subset(pool, top_95_smiles, subset_size=5, entropy_threshold=0.1):
    smiles_list = pool["smiles"].tolist()
    for combination in combinations(smiles_list, subset_size):
        test_subset = top_95_smiles + list(combination)
        entropy = compute_maccs_entropy(test_subset)
        if entropy >= entropy_threshold:
            bt.logging.info(f"Entropy Threshold Met: {entropy:.4f}")
            return pool[pool["smiles"].isin(combination)]

    bt.logging.warning("No combination exceeded the given entropy threshold.")
    return pd.DataFrame()


def main(config: dict):
    # V8 COMBINED: Best features from v1-v7 + vs_opponent
    # - base_n_samples=1800 (from v1/v4/v6/v7) for higher exploration
    # - max_workers=3 (from vs_opponent) for better parallel processing
    # - Early synthon search: iteration > 1 (from v5/vs_opponent)
    # - Dynamic sample size boosting (from v5) - adapts to improvement rate
    # - Improved score calculation (from v7/vs_opponent) - avg + max weighted
    # - Periodic entropy checks (from v7) - every 5 iterations
    # - TOP-1 focused strategy (from v7/vs_opponent) - when scores are very high
    # - Early CPU search: iteration > 1 (from v5) - 3 strategies
    # - Better multi-range strategy (from v7/vs_opponent) - TOP-1, TOP-5, medium, broad
    base_n_samples = 3000  # Higher exploration from v1/v4/v6/v7
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    
    mutation_prob = 0.3
    elite_frac = 0.6
    
    seen_inchikeys = set()
    seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
    start = time.time()
    prev_avg_score = None
    current_avg_score = None
    score_improvement_rate = 0.0
    no_improvement_counter = 0
    
    synthon_lib = None
    use_synthon_search = False
    
    # Track best molecules and score history for adaptive strategies
    best_molecules_history = []
    max_score_history = []
    
    # Enhanced first iteration - good exploration
    n_samples_first_iteration = base_n_samples * 6 if config["allowed_reaction"] != "rxn:5" else base_n_samples * 3

    # Use 3 CPU workers for parallel exploration (from vs_opponent)
    with ProcessPoolExecutor(max_workers=2) as cpu_executor:
        while time.time() - start < 1800:
            iteration += 1
            iter_start_time = time.time()
            
            # Adaptive n_samples: maintain good throughput + dynamic boosting (from v5)
            remaining_time = 1800 - (time.time() - start)
            if remaining_time > 1500:
                base_adaptive = base_n_samples
            elif remaining_time > 900:
                base_adaptive = int(base_n_samples * 0.95)
            elif remaining_time > 600:
                base_adaptive = int(base_n_samples * 0.90)
            elif remaining_time > 300:
                base_adaptive = int(base_n_samples * 0.85)
            else:
                base_adaptive = int(base_n_samples * 0.80)
            
            # Dynamic sample size boosting based on improvement rate (from v5)
            if score_improvement_rate > 0.05:
                n_samples = int(base_adaptive * 1.5)  # Boost when improving fast
                bt.logging.info(f"[V8] High improvement, BOOSTED samples: {n_samples}")
            elif score_improvement_rate > 0.02:
                n_samples = int(base_adaptive * 1.2)
                bt.logging.info(f"[V8] Good improvement, increased samples: {n_samples}")
            else:
                n_samples = base_adaptive
            
            # Build synthon library early: iteration == 2 (from v5/vs_opponent)
            if iteration == 2 and not top_pool.empty and synthon_lib is None:
                try:
                    bt.logging.info("[V8] Building synthon library from top molecules...")
                    synthon_lib_start = time.time()
                    synthon_lib = SynthonLibrary(DB_PATH, rxn_id)
                    use_synthon_search = True
                    bt.logging.info(f"[V8] Synthon library ready! Built in {time.time() - synthon_lib_start:.2f}s")
                except Exception as e:
                    bt.logging.warning(f"[Miner] Could not build synthon library: {e}")
                    use_synthon_search = False

            component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
            # Enhanced elite pool
            elite_df = select_diverse_elites(top_pool, min(150, len(top_pool))) if not top_pool.empty else pd.DataFrame()
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None
            
            # WINNING STRATEGY: Intelligent exploration/exploitation balance
            if iteration == 1:
                bt.logging.info(f"[V8] Iteration {iteration}: Initial broad random sampling")
                data = generate_valid_random_molecules_batch(
                    rxn_id,
                    n_samples=n_samples_first_iteration,
                    db_path=DB_PATH,
                    subnet_config=config,
                    batch_size=400,
                    elite_names=None,
                    elite_frac=0.0,
                    mutation_prob=1.0,
                    avoid_inchikeys=seen_inchikeys,
                    component_weights=None,
                )
            
            elif use_synthon_search and iteration > 1 and not top_pool.empty:  # Early start from v5
                bt.logging.info(f"[V8] Iteration {iteration}: Smart synthon similarity search")
                
                # Get current max score for adaptive strategy
                current_max_score = top_pool['score'].max() if not top_pool.empty else None
                current_avg_score = top_pool['score'].mean() if not top_pool.empty else None
                max_score_history.append(current_max_score)
                if len(max_score_history) > 5:
                    max_score_history.pop(0)
                
                # IMPROVED SCORE CALCULATION (from v7/vs_opponent)
                if current_avg_score is not None and prev_avg_score is not None:
                    avg_improvement = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
                else:
                    avg_improvement = 0.0
                
                if len(max_score_history) >= 2:
                    max_improvement = (max_score_history[-1] - max_score_history[-2]) / max(abs(max_score_history[-2]), 1e-6)
                else:
                    max_improvement = 0.0
                
                # Combined improvement rate (from v7)
                score_improvement_rate = max(avg_improvement, max_improvement * 0.5)
                bt.logging.info(f"[V8] Improved score calc: avg={avg_improvement:.4f}, max={max_improvement:.4f}, final={score_improvement_rate:.4f}")
                
                # SMART: Adaptive strategy based on improvement rate AND absolute score
                has_high_score = current_max_score is not None and current_max_score > 0.01
                has_very_high_score = current_max_score is not None and current_max_score > 0.015
                
                # Time-based strategy
                time_elapsed = time.time() - start
                is_late_stage = time_elapsed > 1200
                is_very_late_stage = time_elapsed > 1500
                
                if score_improvement_rate > 0.05:
                    # High improvement: tight exploration
                    sim_threshold = 0.75
                    n_per_base = 15
                    n_seeds = 20
                    synthon_ratio = 0.75
                    bt.logging.info(f"[V8] High improvement ({score_improvement_rate:.4f}), tight similarity (0.75)")
                
                elif score_improvement_rate > 0.02:
                    # Good improvement: medium-tight exploration
                    sim_threshold = 0.70
                    n_per_base = 18
                    n_seeds = 25
                    synthon_ratio = 0.75
                    bt.logging.info(f"[V8] Good improvement ({score_improvement_rate:.4f}), medium-tight similarity (0.70)")
                
                elif score_improvement_rate > 0.005:
                    # Moderate improvement: balanced exploration
                    sim_threshold = 0.65
                    n_per_base = 20
                    n_seeds = 30
                    synthon_ratio = 0.70
                    bt.logging.info(f"[V8] Moderate improvement ({score_improvement_rate:.4f}), medium similarity (0.65)")
                
                else:
                    # Low/no improvement - PROVEN MULTI-RANGE STRATEGY (from vs_opponent/v7)
                    bt.logging.info(f"[V8] Low improvement ({score_improvement_rate:.4f}), using PROVEN MULTI-RANGE strategy")
                    
                    # SMART: Adjust strategy based on absolute score and time
                    if has_very_high_score or is_very_late_stage:
                        # When we have very high scores, add focused exploitation on TOP 1
                        # Part 1: Ultra-tight on TOP 1 molecule (30% of synthon budget)
                        n_synthon_top1 = int(n_samples * 0.21)  # 30% of 70%
                        synthon_top1_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(1),  # TOP 1 ONLY
                            n_synthon_top1,
                            min_similarity=0.85,  # Very tight
                            n_per_base=50
                        )
                        bt.logging.info(f"[V8] Generated {len(synthon_top1_df)} TOP-1 synthon candidates (sim=0.85)")
                        
                        # Part 2: Ultra-tight on top 5 molecules (10% of synthon budget)
                        n_synthon_tight = int(n_samples * 0.07)  # 10% of 70%
                        synthon_tight_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(5),
                            n_synthon_tight,
                            min_similarity=0.80,  # Tight
                            n_per_base=30
                        )
                        bt.logging.info(f"[V8] Generated {len(synthon_tight_df)} TIGHT synthon candidates (sim=0.80)")
                        
                        # Part 3: Medium on molecules 10-40 (30% of synthon budget)
                        n_synthon_medium = int(n_samples * 0.21)  # 30% of 70%
                        seed_medium = top_pool.iloc[10:40] if len(top_pool) > 40 else top_pool.iloc[5:]
                        synthon_medium_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            seed_medium,
                            n_synthon_medium,
                            min_similarity=0.55,  # Medium - like richard1220v3
                            n_per_base=15
                        )
                        bt.logging.info(f"[V8] Generated {len(synthon_medium_df)} MEDIUM synthon candidates (sim=0.55)")
                        
                        # Part 4: Broad on top 50 (30% of synthon budget)
                        n_synthon_broad = int(n_samples * 0.21)  # 30% of 70%
                        synthon_broad_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(50),
                            n_synthon_broad,
                            min_similarity=0.40,  # Broad - like richard1220v3
                            n_per_base=20
                        )
                        bt.logging.info(f"[V8] Generated {len(synthon_broad_df)} BROAD synthon candidates (sim=0.40)")
                        
                        # Combine all synthon approaches
                        synthon_df = pd.concat([synthon_top1_df, synthon_tight_df, synthon_medium_df, synthon_broad_df], ignore_index=True)
                    else:
                        # Standard: PROVEN multi-range strategy from richard1220v3
                        # Part 1: Ultra-tight on top 5 molecules (40% of synthon budget)
                        n_synthon_tight = int(n_samples * 0.28)  # 40% of 70%
                        synthon_tight_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(5),
                            n_synthon_tight,
                            min_similarity=0.80,  # Very tight!
                            n_per_base=30
                        )
                        bt.logging.info(f"[V8] Generated {len(synthon_tight_df)} TIGHT synthon candidates (sim=0.80)")
                        
                        # Part 2: Medium on molecules 10-40 (30% of synthon budget)
                        n_synthon_medium = int(n_samples * 0.21)  # 30% of 70%
                        seed_medium = top_pool.iloc[10:40] if len(top_pool) > 40 else top_pool.iloc[5:]
                        synthon_medium_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            seed_medium,
                            n_synthon_medium,
                            min_similarity=0.55,  # Medium - like richard1220v3
                            n_per_base=15
                        )
                        bt.logging.info(f"[V8] Generated {len(synthon_medium_df)} MEDIUM synthon candidates (sim=0.55)")
                        
                        # Part 3: Broad on top 50 (30% of synthon budget)
                        n_synthon_broad = int(n_samples * 0.21)  # 30% of 70%
                        synthon_broad_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(50),
                            n_synthon_broad,
                            min_similarity=0.40,  # Broad - like richard1220v3
                            n_per_base=20
                        )
                        bt.logging.info(f"[V8] Generated {len(synthon_broad_df)} BROAD synthon candidates (sim=0.40)")
                        
                        # Combine all synthon approaches
                        synthon_df = pd.concat([synthon_tight_df, synthon_medium_df, synthon_broad_df], ignore_index=True)
                    
                    synthon_df = synthon_df.drop_duplicates(subset=["name"], keep="first")
                    
                    if not synthon_df.empty:
                        synthon_df = validate_molecules(synthon_df, config)
                        bt.logging.info(f"[V8] {len(synthon_df)} multi-range synthon candidates passed validation")
                    
                    # Generate remaining from GA with component weighting
                    n_traditional = n_samples - len(synthon_df)
                    if n_traditional > 0:
                        traditional_df = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_traditional,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=400,
                            elite_names=elite_names,
                            elite_frac=elite_frac,
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )
                    else:
                        traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
                    
                    data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                    data = data.drop_duplicates(subset=["name"], keep="first")
                    bt.logging.info(f"[V8] Combined: {len(data)} total ({len(synthon_df)} multi-range synthon + {len(traditional_df)} GA)")
                    
                    # Skip the standard synthon generation below
                    synthon_df = None
                
                # Standard single-range synthon generation (for high/medium improvement)
                if score_improvement_rate > 0.005:  # Only if not using multi-range
                    n_synthon = int(n_samples * synthon_ratio)
                    synthon_gen_start = time.time()
                    synthon_df = generate_molecules_from_synthon_library(
                        synthon_lib,
                        top_pool.head(n_seeds),
                        n_synthon,
                        min_similarity=sim_threshold,
                        n_per_base=n_per_base
                    )
                    bt.logging.info(f"[V8] Generated {len(synthon_df)} synthon candidates in {time.time() - synthon_gen_start:.2f}s")
                    
                    # Generate remaining from traditional method
                    n_traditional = n_samples - len(synthon_df)
                    if n_traditional > 0:
                        traditional_df = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_traditional,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=300,
                            elite_names=elite_names,
                            elite_frac=elite_frac,
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )
                    else:
                        traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
                    
                    # Validate and combine
                    if not synthon_df.empty:
                        synthon_df = validate_molecules(synthon_df, config)
                        bt.logging.info(f"[V8] {len(synthon_df)} synthon candidates passed validation")
                    
                    data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                    data = data.drop_duplicates(subset=["name"], keep="first")
                    bt.logging.info(f"[V8] Combined: {len(data)} total ({len(synthon_df)} synthon + {len(traditional_df)} GA)")
            
            elif no_improvement_counter < 3:
                bt.logging.info(f"[V8] Iteration {iteration}: Standard genetic algorithm")
                data = generate_valid_random_molecules_batch(
                    rxn_id,
                    n_samples=n_samples,
                    db_path=DB_PATH,
                    subnet_config=config,
                    batch_size=400,
                    elite_names=elite_names,
                    elite_frac=elite_frac,
                    mutation_prob=mutation_prob,
                    avoid_inchikeys=seen_inchikeys,
                    component_weights=component_weights,
                )
            
            elif no_improvement_counter < 6:
                bt.logging.info(f"[V8] Iteration {iteration}: Exploring similar space (no_improvement={no_improvement_counter})")
                data = _cpu_random_candidates_with_similarity(
                    iteration,
                    30,
                    config,
                    top_pool.head(50)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys,
                    0.65
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
            
            else:
                bt.logging.info(f"[V8] Iteration {iteration}: Broad exploration reset (no_improvement={no_improvement_counter})")
                data = _cpu_random_candidates_with_similarity(
                    iteration,
                    40,
                    config,
                    top_pool.head(100)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys,
                    0.0
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
                no_improvement_counter = 0
            
            gen_time = time.time() - iter_start_time
            bt.logging.info(f"[V8] Iteration {iteration}: {len(data)} Samples Generated in ~{gen_time:.2f}s (pre-score)")

            if data.empty:
                bt.logging.warning(f"[V8] Iteration {iteration}: No valid molecules produced; continuing")
                continue
            
            if not seed_df.empty:
                data = pd.concat([data, seed_df])
                data = data.drop_duplicates(subset=["InChIKey"], keep="first")
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])

            try:
                filterd_data = data[~data["InChIKey"].isin(seen_inchikeys)]
                if len(filterd_data) < len(data):
                    bt.logging.warning(
                        f"[V8] Iteration {iteration}: {len(data) - len(filterd_data)} molecules were previously seen"
                    )

                dup_ratio = (len(data) - len(filterd_data)) / max(1, len(data))
                
                if dup_ratio > 0.7:
                    mutation_prob = min(0.9, mutation_prob * 1.5)
                    elite_frac = max(0.15, elite_frac * 0.7)
                    bt.logging.warning(f"[V8] SEVERE duplication ({dup_ratio:.2%})! mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                elif dup_ratio > 0.5:
                    mutation_prob = min(0.7, mutation_prob * 1.3)
                    elite_frac = max(0.2, elite_frac * 0.8)
                    bt.logging.warning(f"[V8] High duplication ({dup_ratio:.2%}), mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                    mutation_prob = max(0.05, mutation_prob * 0.95)
                    elite_frac = min(0.85, elite_frac * 1.05)

                data = filterd_data

            except Exception as e:
                bt.logging.warning(f"[V8] Pre-score deduplication failed: {e}")

            if data.empty:
                bt.logging.error(f"[V8] Iteration {iteration}: ALL molecules were duplicates! Skipping scoring and continuing...")
                # Force more diversity for next iteration
                mutation_prob = min(0.95, mutation_prob * 2.0)
                elite_frac = max(0.1, elite_frac * 0.5)
                bt.logging.warning(f"[V8] Emergency diversity boost: mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                continue  # Skip to next iteration

            data = data.reset_index(drop=True)

            # Enhanced CPU similarity search - multiple parallel searches (from v5/vs_opponent)
            # Early start: iteration > 1 (from v5)
            cpu_futures = []
            if not top_pool.empty and iteration > 1:  # Early start from v5
                # Multiple parallel CPU searches with different strategies
                if score_improvement_rate < 0.01:
                    # Strategy 1: Tight on top 5
                    cpu_futures.append((
                        cpu_executor.submit(
                            _cpu_random_candidates_with_similarity,
                            iteration,
                            50,  # Increased from v5
                            config,
                            top_pool.head(5)[["name", "smiles", "InChIKey"]],
                            seen_inchikeys,
                            0.82  # Slightly higher threshold
                        ),
                        "tight-top5"
                    ))
                    
                    # Strategy 2: Medium on top 15 (from v5)
                    cpu_futures.append((
                        cpu_executor.submit(
                            _cpu_random_candidates_with_similarity,
                            iteration,
                            40,
                            config,
                            top_pool.head(15)[["name", "smiles", "InChIKey"]],
                            seen_inchikeys,
                            0.70
                        ),
                        "medium-top15"
                    ))
                    
                    # Strategy 3: Broad on top 30 (from v5)
                    cpu_futures.append((
                        cpu_executor.submit(
                            _cpu_random_candidates_with_similarity,
                            iteration,
                            30,
                            config,
                            top_pool.head(30)[["name", "smiles", "InChIKey"]],
                            seen_inchikeys,
                            0.55
                        ),
                        "broad-top30"
                    ))
            
            gpu_start_time = time.time()

            if len(data) == 0:
                bt.logging.error(f"[V8] Iteration {iteration}: No molecules to score! Continuing...")
                continue

            data["Target"] = target_score_from_data(data["smiles"])
            data["Anti"] = antitarget_scores()
            data["score"] = data["Target"] - (config["antitarget_weight"] * data["Anti"])

            if data["score"].isna().all():
                bt.logging.error(f"[V8] Iteration {iteration}: Scoring failed (all NaN)! Continuing...")
                continue
            
            gpu_time = time.time() - gpu_start_time
            bt.logging.info(f"[V8] Iteration {iteration}: GPU scoring time ~{gpu_time:.2f}s")
            
            # Collect all CPU results
            if cpu_futures:
                for cpu_future, strategy_name in cpu_futures:
                    try:
                        cpu_df = cpu_future.result(timeout=0)
                        if not cpu_df.empty:
                            if seed_df.empty:
                                seed_df = cpu_df.copy()
                            else:
                                seed_df = pd.concat([seed_df, cpu_df], ignore_index=True)
                            bt.logging.info(f"[V8] CPU similarity ({strategy_name}) found {len(cpu_df)} candidates")
                    except TimeoutError:
                        pass
                    except Exception as e:
                        bt.logging.warning(f"[V8] CPU similarity ({strategy_name}) failed: {e}")
                
                if not seed_df.empty:
                    seed_df = seed_df.drop_duplicates(subset=["InChIKey"], keep="first")
            
            seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
            total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
            prev_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            # Safe concatenation
            if not total_data.empty:
                top_pool = pd.concat([top_pool, total_data], ignore_index=True)
                top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                top_pool = top_pool.sort_values(by="score", ascending=False)
            else:
                bt.logging.warning(f"[V8] Iteration {iteration}: No valid scored data to add to pool")

            # PERIODIC ENTROPY CHECK (from v7) - every 5 iterations
            if iteration % 5 == 0 and not top_pool.empty:
                try:
                    current_entropy = compute_maccs_entropy(
                        top_pool.head(config["num_molecules"])['smiles'].to_list()
                    )
                    bt.logging.info(f"[V8] Periodic entropy check: {current_entropy:.4f}")
                    
                    if current_entropy < config['entropy_min_threshold'] * 1.2:
                        mutation_prob = min(0.8, mutation_prob * 1.3)
                        elite_frac = max(0.3, elite_frac * 0.8)
                        bt.logging.warning(f"[V8] Low entropy detected, boosting diversity: mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                except Exception as e:
                    bt.logging.warning(f"[V8] Entropy check failed: {e}")
                
                # Track best molecules for later intensive exploration
                best_molecules_history.append({
                    'iteration': iteration,
                    'molecules': top_pool.head(10)[["name", "smiles", "InChIKey", "score"]].copy()
                })
                if len(best_molecules_history) > 6:
                    best_molecules_history.pop(0)

            remaining_time = 1800 - (time.time() - start)
            if remaining_time <= 60:
                entropy = compute_maccs_entropy(top_pool.iloc[:config["num_molecules"]]['smiles'].to_list())
                if entropy > config['entropy_min_threshold']:
                    top_pool = top_pool.head(config["num_molecules"])
                    bt.logging.info(f"[V8] Iteration {iteration}: Sufficient Entropy = {entropy:.4f}")
                else:
                    try:
                        top_95 = top_pool.iloc[:95]
                        remaining_pool = top_pool.iloc[95:]
                        additional_5 = select_diverse_subset(remaining_pool, top_95["smiles"].tolist(), 
                                                            subset_size=5, entropy_threshold=config['entropy_min_threshold'])
                        if not additional_5.empty:
                            top_pool = pd.concat([top_95, additional_5]).reset_index(drop=True)
                            entropy = compute_maccs_entropy(top_pool['smiles'].to_list())
                            bt.logging.info(f"[V8] Iteration {iteration}: Adjusted Entropy = {entropy:.4f}")
                        else:
                            top_pool = top_pool.head(config["num_molecules"])
                    except Exception as e:
                        bt.logging.warning(f"[V8] Entropy handling failed: {e}")
            else:
                top_pool = top_pool.head(config["num_molecules"])
            
            current_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            if current_avg_score is not None:
                if prev_avg_score is not None:
                    score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
                prev_avg_score = current_avg_score

            if score_improvement_rate == 0.0:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0
            
            iter_total_time = time.time() - iter_start_time
            top_entries = {"molecules": top_pool["name"].tolist()}
            total_time = time.time() - start
            
            bt.logging.info(
                f"[V8] Iteration {iteration} || Time: {iter_total_time:.2f}s | Total: {total_time:.2f}s | "
                f"Avg: {top_pool['score'].mean():.4f} | Max: {top_pool['score'].max():.4f} | "
                f"Min: {top_pool['score'].min():.4f} | Elite: {elite_frac:.2f} | "
                f"Mut: {mutation_prob:.2f} | Improve: {score_improvement_rate:.4f}"
            )

            with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    config = get_config()
    start_time_1 = time.time()
    initialize_models(config)
    bt.logging.info(f"{time.time() - start_time_1} seconds for model initialization")
    main(config)

