# BindCraft: Peptide Screening and MPNN Refinement Pipeline
#
# This script takes a target protein and a list of candidate peptide sequences.
# 1. Each input peptide is first evaluated for binding to the target.
# 2. If an input peptide passes initial filters, its predicted structure is used
#    as a template for ProteinMPNN to generate sequence variants.
# 3. These MPNN variants are then evaluated and filtered.
# 4. Accepted structures (both initial and MPNN variants) are saved.

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from copy import deepcopy # Changed from copy to deepcopy for settings dicts
import gc

# Ensure the functions directory is in the path
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, 'functions'))

try:
    from functions import * # generic_utils, colabdesign_utils, pyrosetta_utils, biopython_utils
except ImportError as e:
    print(f"Error importing from functions module: {e}")
    print("Please ensure 'functions' directory is in the same path as the script and contains necessary __init__.py")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='BindCraft: Peptide Screening and MPNN Refinement Pipeline')
    parser.add_argument('--settings', '-s', type=str, required=True,
                        help='Path to the target_settings.json file (for target PDB, chains, design_path, binder_name).')
    parser.add_argument('--peptide_input_file', '-p', type=str, required=True,
                        help='Path to a text file containing initial peptide sequences (one per line).')
    parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                        help='Path to the filters.json file. Default: ./settings_filters/default_filters.json')
    parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer.json',
                        help='Path to the advanced_settings.json file. Default: ./settings_advanced/default_4stage_multimer.json')

    args = parser.parse_args()

    print("Peptide Screening and MPNN Refinement Script")
    print(f"Settings: {args.settings}")
    print(f"Peptide Input File: {args.peptide_input_file}")
    print(f"Filters: {args.filters}")
    print(f"Advanced Settings: {args.advanced}")

    # --- Initialize Environment and Load Settings ---
    script_start_time = time.time()

    # Perform checks of input setting files
    settings_path, filters_path, advanced_path = perform_input_check(args) # Assuming perform_input_check is in generic_utils

    # Load settings from JSON (use deepcopy for mutable dicts)
    raw_target_settings, raw_advanced_settings, raw_filters = load_json_settings(settings_path, filters_path, advanced_path)
    target_settings = deepcopy(raw_target_settings)
    advanced_settings = deepcopy(raw_advanced_settings)
    filters = deepcopy(raw_filters)

    settings_file_name = os.path.basename(settings_path).split('.')[0]
    filters_file_name = os.path.basename(filters_path).split('.')[0]
    advanced_file_name = os.path.basename(advanced_path).split('.')[0]

    # Load AF2 model settings
    # Assuming load_af2_models function exists and returns (design_models, prediction_models, multimer_validation)
    # For this script, we mainly need prediction_models and multimer_validation. design_models might not be used.
    _, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])

    # Perform checks on advanced_settings
    bindcraft_folder = os.path.dirname(os.path.realpath(__file__)) # Or sys.path[0]
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

    # Generate base output directories
    # Modify generate_directories if needed, or create subdirectories manually
    design_paths = generate_directories(target_settings["design_path"])

    # Create specific subdirectories for this pipeline
    sub_dirs_to_create = [
        "Input_Eval/PDB", "Input_Eval/Relaxed", "Input_Eval/Binder",
        "MPNN_Eval/PDB", "MPNN_Eval/Relaxed", "MPNN_Eval/Binder",
        "Accepted_Input_Peptides", "Rejected_Input_Peptides",
        "Accepted_MPNN_Variants", "Rejected_MPNN_Variants"
    ]
    for sub_dir_leaf in sub_dirs_to_create:
        full_sub_dir_path = os.path.join(target_settings["design_path"], sub_dir_leaf)
        if not os.path.exists(full_sub_dir_path):
            os.makedirs(full_sub_dir_path)
        design_paths[sub_dir_leaf] = full_sub_dir_path # Add to design_paths dict for easy access

    # Define CSV file paths
    input_peptide_stats_csv = os.path.join(target_settings["design_path"], 'input_peptide_screening_stats.csv')
    mpnn_variant_stats_csv = os.path.join(target_settings["design_path"], 'mpnn_variant_stats.csv')
    failure_stats_csv = os.path.join(target_settings["design_path"], 'failure_stats.csv') # Consistent with bindcraft.py

    # Generate dataframe labels ( reusing from bindcraft.py's logic if possible or define new ones)
    # For now, assume design_labels from generate_dataframe_labels() is suitable.
    # generate_dataframe_labels might need to be available or its logic replicated.
    # Let's assume a simplified set of labels for now if generate_dataframe_labels is complex to call directly

    # Using the same labels as mpnn_design_stats.csv from bindcraft.py for now.
    # The `generate_dataframe_labels` function in bindcraft.py provides `trajectory_labels`, `design_labels`, `final_labels`.
    # We'll need `design_labels`.
    try:
        _, design_labels, _ = generate_dataframe_labels()
    except NameError: # Fallback if generate_dataframe_labels is not directly available from imported functions
        print("Warning: generate_dataframe_labels() not found. Using placeholder CSV labels. CSV output might be incomplete.")
        design_labels = ["Name", "Sequence", "Length", "pLDDT_avg", "iPTM_avg", "Status"] # Minimal example

    create_dataframe(input_peptide_stats_csv, design_labels)
    create_dataframe(mpnn_variant_stats_csv, design_labels)
    generate_filter_pass_csv(failure_stats_csv, args.filters) # From bindcraft.py

    # Initialize PyRosetta
    try:
        pr_init_command = f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1'
        pr.init(pr_init_command) # Assuming pr (pyrosetta) is imported via 'from functions import *'
        print("PyRosetta initialized successfully.")
    except Exception as e:
        print(f"Error initializing PyRosetta: {e}")
        print("Please ensure PyRosetta is correctly installed and configured (e.g., dalphaball_path in advanced_settings).")
        # Decide if exit or continue without PyRosetta features
        # For now, we'll assume it's critical and would be caught by function calls later if pr is not defined.

    print(f"Running peptide screening for target: {target_settings['binder_name']}")
    print(f"Advanced settings profile: {advanced_file_name}")
    print(f"Filter profile: {filters_file_name}")

    # --- Read Input Peptides ---
    try:
        with open(args.peptide_input_file, 'r') as f:
            input_peptide_sequences = [line.strip() for line in f if line.strip()]
        if not input_peptide_sequences:
            print(f"Error: Peptide input file '{args.peptide_input_file}' is empty or contains only whitespace. Exiting.")
            sys.exit(1)
        print(f"Read {len(input_peptide_sequences)} peptide sequences from '{args.peptide_input_file}'.")
    except FileNotFoundError:
        print(f"Error: Peptide input file '{args.peptide_input_file}' not found. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading peptide input file '{args.peptide_input_file}': {e}. Exiting.")
        sys.exit(1)

    # Initialize counters or global variables for the pipeline
    total_input_peptides_processed = 0
    total_input_accepted_stage1 = 0
    total_mpnn_variants_generated = 0
    total_mpnn_variants_accepted = 0

    # Placeholder for AF2 model compilation (will be done inside loops per peptide length)
    # complex_prediction_model = None
    # binder_prediction_model = None

    # --- Compile AF2 Models (once) ---
    # These models will have prep_inputs called for each peptide length later
    clear_mem()
    complex_prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings["num_recycles_validation"], # Using validation recycles for prediction
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
        use_initial_guess=False, # Explicitly False for input peptide stage
        use_initial_atom_pos=False # Explicitly False for input peptide stage
    )
    print("AF2 complex prediction model compiled.")

    binder_prediction_model = mk_afdesign_model(
        protocol="hallucination",
        use_templates=False,
        initial_guess=False,
        use_initial_atom_pos=False,
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation
    )
    print("AF2 binder monomer prediction model compiled.")

    binder_chain_id = "B" # Consistent with BindCraft default for binder

    # --- Stage 1: Initial Evaluation of Input Peptides ---
    print("\n--- Starting Stage 1: Initial Evaluation of Input Peptides ---")
    for peptide_idx, input_sequence in enumerate(input_peptide_sequences):
        total_input_peptides_processed += 1
        print(f"\nProcessing Input Peptide {total_input_peptides_processed}/{len(input_peptide_sequences)}: {input_sequence}")

        current_peptide_sequence = re.sub("[^A-Z]", "", input_sequence.upper())
        if not current_peptide_sequence:
            print(f"Skipping invalid or empty sequence: {input_sequence}")
            continue

        peptide_length = len(current_peptide_sequence)
        eval_name = f"{target_settings['binder_name']}_InputPeptide_{total_input_peptides_processed}_L{peptide_length}"
        stage1_time_start = time.time()

        # Prepare AF2 models for the current peptide length
        try:
            complex_prediction_model.prep_inputs(
                pdb_filename=target_settings["starting_pdb"],
                chain=target_settings["chains"],
                binder_len=peptide_length,
                rm_target_seq=advanced_settings["rm_template_seq_predict"],
                rm_target_sc=advanced_settings["rm_template_sc_predict"]
            )
            binder_prediction_model.prep_inputs(length=peptide_length)
        except Exception as e:
            print(f"Error preparing AF2 models for peptide {current_peptide_sequence} (length {peptide_length}): {e}")
            # Consider how to log this failure, perhaps to a dedicated error log or notes in CSV
            continue # Skip to next peptide

        # Predict binder complex
        # predict_binder_complex expects trajectory_pdb for initial guess if enabled, but it's disabled here.
        # It's also used for naming output files if mpnn_design_name is not sufficiently unique.
        # Here, eval_name should be unique.
        # The 6th argument to predict_binder_complex is trajectory_pdb, used for initial guess if model is set up for it.
        # Since complex_prediction_model is set with use_initial_guess=False, this path won't be used for guessing.
        # It might still be used for some internal logic or default atom positions if not using big_bang.
        # Passing target_settings["starting_pdb"] as a safe default.
        initial_complex_stats, pass_initial_af2_filters = predict_binder_complex(
            model=complex_prediction_model, # Corrected argument name
            binder_sequence=current_peptide_sequence,
            mpnn_design_name=eval_name, # Using eval_name as the base for PDB file names
            target_pdb=target_settings["starting_pdb"],
            chain=target_settings["chains"],
            length=peptide_length,
            trajectory_pdb=target_settings["starting_pdb"], # Placeholder, not used for initial guess here
            prediction_models_to_run=prediction_models, # Corrected argument name
            advanced_settings=advanced_settings,
            filters_to_apply=filters, # Corrected argument name
            design_paths_dict=design_paths, # Corrected argument name
            failure_csv_path=failure_stats_csv, # Corrected argument name
            # Specify output subdirectories for this stage
            output_pdb_dir=design_paths["Input_Eval/PDB"],
            output_relaxed_pdb_dir=design_paths["Input_Eval/Relaxed"]
        )

        if not pass_initial_af2_filters:
            print(f"Peptide {eval_name} failed initial AF2 filters. Skipping detailed scoring and MPNN.")
            # Minimal log entry for failure might be useful here if not handled by predict_binder_complex's failure logging
            # For now, assume predict_binder_complex logs to failure_stats_csv if filters are specified.
            # We might want to add a row to input_peptide_stats_csv indicating this specific failure.
            log_data_input_peptide = [eval_name, "initial_eval", peptide_length, None, None, # seed, helicity placeholders
                                      target_settings['target_hotspot_residues'], current_peptide_sequence,
                                      None, None, None] # interface_res, mpnn_score, mpnn_seqid
            # Fill remaining with None up to "Time", "Notes", "Settings", "Filters", "Advanced"
            num_metric_placeholders = len(design_labels) - len(log_data_input_peptide) - 5
            log_data_input_peptide.extend([None] * num_metric_placeholders)
            log_data_input_peptide.extend([format_time(time.time() - stage1_time_start), "Failed initial AF2 predict", settings_file_name, filters_file_name, advanced_file_name])
            insert_data(input_peptide_stats_csv, log_data_input_peptide)
            # PDBs might have been saved by predict_binder_complex; could move to a specific "failed_af2_predict" folder if desired.
            continue # To the next input peptide

        # --- Detailed Scoring for Input Peptide (if basic AF2 filters passed) ---
        detailed_complex_stats = deepcopy(initial_complex_stats) # initial_complex_stats has AF2 metrics
        best_model_pdb_for_mpnn = None
        interface_residues_for_mpnn = None

        # Iterate through models (1-5) that were predicted
        for model_idx in prediction_models: # model_idx is 0,1,2,3,4
            model_num = model_idx + 1 # model_num is 1,2,3,4,5

            unrelaxed_pdb_path = os.path.join(design_paths["Input_Eval/PDB"], f"{eval_name}_model{model_num}.pdb")
            relaxed_pdb_path = os.path.join(design_paths["Input_Eval/Relaxed"], f"{eval_name}_model{model_num}.pdb")

            if not os.path.exists(relaxed_pdb_path): # Should have been created by predict_binder_complex if it passed its internal filters
                print(f"Warning: Relaxed PDB {relaxed_pdb_path} not found for {eval_name} model {model_num}. Skipping detailed scoring for this model.")
                if model_num in detailed_complex_stats: # Ensure AF2 stats are still there
                     detailed_complex_stats[model_num].update({k: None for k in ['Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'Target_RMSD', 'InterfaceAAs']}) # etc.
                continue

            clash_unrelaxed = calculate_clash_score(unrelaxed_pdb_path)
            clash_relaxed = calculate_clash_score(relaxed_pdb_path)

            rosetta_scores, rosetta_interface_AAs, rosetta_interface_residues_str = score_interface(relaxed_pdb_path, binder_chain_id)

            ss_alpha, ss_beta, ss_loops, ss_alpha_i, ss_beta_i, ss_loops_i, i_plddt_val, ss_plddt_val = calc_ss_percentage(relaxed_pdb_path, advanced_settings, binder_chain_id, is_relaxed=True) # Use relaxed for SS of final state

            target_rmsd_val = target_pdb_rmsd(relaxed_pdb_path, target_settings["starting_pdb"], target_settings["chains"]) # RMSD on relaxed

            # Update stats for this model
            if model_num not in detailed_complex_stats: detailed_complex_stats[model_num] = {}
            detailed_complex_stats[model_num].update({
                'Unrelaxed_Clashes': clash_unrelaxed, 'Relaxed_Clashes': clash_relaxed,
                'Binder_Energy_Score': rosetta_scores['binder_score'], 'Surface_Hydrophobicity': rosetta_scores['surface_hydrophobicity'],
                'ShapeComplementarity': rosetta_scores['interface_sc'], 'PackStat': rosetta_scores['interface_packstat'],
                'dG': rosetta_scores['interface_dG'], 'dSASA': rosetta_scores['interface_dSASA'],
                'dG/dSASA': rosetta_scores['interface_dG_SASA_ratio'], 'Interface_SASA_%': rosetta_scores['interface_fraction'],
                'Interface_Hydrophobicity': rosetta_scores['interface_hydrophobicity'], 'n_InterfaceResidues': rosetta_scores['interface_nres'],
                'n_InterfaceHbonds': rosetta_scores['interface_interface_hbonds'], 'InterfaceHbondsPercentage': rosetta_scores['interface_hbond_percentage'],
                'n_InterfaceUnsatHbonds': rosetta_scores['interface_delta_unsat_hbonds'], 'InterfaceUnsatHbondsPercentage': rosetta_scores['interface_delta_unsat_hbonds_percentage'],
                'InterfaceAAs': rosetta_interface_AAs, 'InterfaceRes': rosetta_interface_residues_str, # Storing this per model if needed, or take first good one
                'i_pLDDT_Rosetta': i_plddt_val, 'ss_pLDDT_Rosetta': ss_plddt_val, # Naming to distinguish if AF2 also provides these
                'Interface_Helix%': ss_alpha_i, 'Interface_BetaSheet%': ss_beta_i, 'Interface_Loop%': ss_loops_i,
                'Binder_Helix%': ss_alpha, 'Binder_BetaSheet%': ss_beta, 'Binder_Loop%': ss_loops,
                'Target_RMSD': target_rmsd_val,
                'Hotspot_RMSD': None # No trajectory for hotspot comparison in this stage
            })
            if interface_residues_for_mpnn is None: # Take from first successfully scored model
                interface_residues_for_mpnn = rosetta_interface_residues_str


        complex_averages = calculate_averages(detailed_complex_stats, handle_aa=True)
        # If no model was scored, averages might be empty or full of Nones.
        if not interface_residues_for_mpnn and complex_averages.get('InterfaceRes'): # Fallback from average if per-model failed
            interface_residues_for_mpnn = complex_averages['InterfaceRes']


        # Predict binder alone
        binder_alone_stats = predict_binder_alone(
            model=binder_prediction_model, # Corrected argument name
            binder_sequence=current_peptide_sequence,
            mpnn_design_name=eval_name, # Base name for PDBs
            length=peptide_length,
            trajectory_pdb=None, # No trajectory reference for RMSD calculation to original
            binder_chain_id_in_traj="A", # Default if no trajectory_pdb
            prediction_models_to_run=prediction_models, # Corrected argument name
            advanced_settings=advanced_settings,
            design_paths_dict=design_paths, # Corrected argument name
            # Specify output subdirectories for this stage
            output_pdb_dir=design_paths["Input_Eval/Binder"]
        )
        # Add Binder_RMSD as None for each model in binder_alone_stats as no trajectory_pdb
        for model_num_key in binder_alone_stats:
            if isinstance(binder_alone_stats[model_num_key], dict): # Ensure it's a dict
                 binder_alone_stats[model_num_key]['Binder_RMSD'] = None
        binder_alone_averages = calculate_averages(binder_alone_stats)


        # Log data for input peptide
        seq_notes = validate_design_sequence(current_peptide_sequence, complex_averages.get('Relaxed_Clashes', float('inf')), advanced_settings)
        stage1_time_taken = time.time() - stage1_time_start

        log_data_input_peptide = [
            eval_name, "initial_eval", peptide_length, None, None, # Seed, Helicity placeholders
            target_settings['target_hotspot_residues'], current_peptide_sequence,
            interface_residues_for_mpnn if interface_residues_for_mpnn else complex_averages.get('InterfaceRes'), # Use average if first model failed for interface res
            None, None # MPNN Score, MPNN SeqID placeholders
        ]
        # Append complex stats
        for label in design_labels: # Iterate through all expected labels to fill data
            if label.startswith("Average_") and not label.startswith("Average_Binder_"):
                log_data_input_peptide.append(complex_averages.get(label.replace("Average_",""), None))
            elif label.startswith(tuple(f"{i+1}_" for i in range(5))) and not label.startswith(tuple(f"{i+1}_Binder_" for i in range(5))):
                model_num_str, metric_key = label.split("_", 1)
                log_data_input_peptide.append(detailed_complex_stats.get(int(model_num_str), {}).get(metric_key, None))
        # Append binder stats
        for label in design_labels:
            if label.startswith("Average_Binder_"):
                log_data_input_peptide.append(binder_alone_averages.get(label.replace("Average_Binder_",""), None))
            elif label.startswith(tuple(f"{i+1}_Binder_" for i in range(5))):
                model_num_str, _, metric_key = label.split("_", 2) # e.g. 1_Binder_pLDDT
                log_data_input_peptide.append(binder_alone_stats.get(int(model_num_str), {}).get(metric_key, None))

        # Ensure correct number of entries before time/notes by padding if necessary
        # This is a bit fragile; depends heavily on design_labels structure.
        # Current length of log_data_input_peptide before time/notes/settings:
        # Name, Alg, Len, Seed, Heli, Hotspots, Seq, InterfaceRes, MPNNScore, MPNNSeqRec = 10
        # + Num_Avg_Complex_Metrics + (Num_Models * Num_Per_Model_Complex_Metrics)
        # + Num_Avg_Binder_Metrics + (Num_Models * Num_Per_Model_Binder_Metrics)
        # The number of metrics is fixed by design_labels.
        # The number of "header" fields before metrics is 10.
        # The number of "footer" fields is 5 (Time, Notes, Settings, Filters, Advanced).
        # So, total fields = 10 (header) + NUM_METRICS_IN_DESIGN_LABELS + 5 (footer)
        # Let's find how many metric fields are in design_labels:
        metric_labels_in_design_labels = [l for l in design_labels if l not in ['Name', 'Algorithm', 'Length', 'Seed', 'Helicity', 'Hotspots', 'Sequence', 'Interface Res', 'MPNN_score', 'MPNN_seq_recovery', 'Time', 'Notes', 'Settings', 'Filters', 'Advanced']]

        current_log_length = len(log_data_input_peptide)
        expected_length_before_footer = 10 + len(metric_labels_in_design_labels)

        if current_log_length < expected_length_before_footer:
            log_data_input_peptide.extend([None] * (expected_length_before_footer - current_log_length))
        elif current_log_length > expected_length_before_footer: # Should not happen if logic above is correct
            log_data_input_peptide = log_data_input_peptide[:expected_length_before_footer]


        log_data_input_peptide.extend([format_time(stage1_time_taken), seq_notes, settings_file_name, filters_file_name, advanced_file_name])
        insert_data(input_peptide_stats_csv, log_data_input_peptide)

        # Determine best model for filter check and for MPNN input
        # Using average iPTM or a specific model's iPTM/pLDDT
        # For simplicity, let's use average iPTM from complex_averages for now, or find best model by iPTM
        best_model_num_for_mpnn = None
        highest_iptm = -1.0
        if complex_averages.get('i_pTM') is not None and complex_averages['i_pTM'] > advanced_settings.get("min_avg_iptm_for_mpnn_seed", 0.3): # Example threshold
             # Find which model contributed most or had highest iPTM if needed, or just use average passing.
             # For now, if average is good, try to find the best single model by iPTM.
            for model_n, stats in detailed_complex_stats.items():
                if isinstance(stats, dict) and stats.get('i_pTM', -1.0) > highest_iptm:
                    highest_iptm = stats['i_pTM']
                    best_model_num_for_mpnn = model_n

        if best_model_num_for_mpnn is None: # Fallback: use model with best pLDDT if no good iPTM found
            highest_plddt = -1.0
            for model_n, stats in detailed_complex_stats.items():
                if isinstance(stats, dict) and stats.get('pLDDT', -1.0) > highest_plddt:
                    highest_plddt = stats['pLDDT']
                    best_model_num_for_mpnn = model_n

        if best_model_num_for_mpnn is None and len(detailed_complex_stats) > 0 : # Still none, take first available model
             best_model_num_for_mpnn = list(detailed_complex_stats.keys())[0]


        passes_stage1_filters = check_filters(log_data_input_peptide, design_labels, filters)

        if passes_stage1_filters == True and best_model_num_for_mpnn is not None:
            total_input_accepted_stage1 += 1
            print(f"Input peptide {eval_name} PASSED Stage 1 filters.")
            best_model_pdb_for_mpnn = os.path.join(design_paths["Input_Eval/Relaxed"], f"{eval_name}_model{best_model_num_for_mpnn}.pdb")
            if not os.path.exists(best_model_pdb_for_mpnn):
                print(f"Error: Best model PDB {best_model_pdb_for_mpnn} for MPNN input does not exist! Skipping MPNN for this peptide.")
            else:
                shutil.copy(best_model_pdb_for_mpnn, os.path.join(design_paths["Accepted_Input_Peptides"], f"{eval_name}_model{best_model_num_for_mpnn}_accepted.pdb"))
                # --- STAGE 2 & 3 (MPNN) WOULD GO HERE ---
                print(f"Proceeding to MPNN refinement for {eval_name} using model {best_model_num_for_mpnn} and interface: {interface_residues_for_mpnn}")

                # --- Stage 2: ProteinMPNN Refinement ---
                mpnn_refinement_start_time = time.time()
                # Ensure interface_residues_for_mpnn is suitable for mpnn_gen_sequence (e.g. string "B52,B53")
                # The mpnn_gen_sequence in bindcraft uses a binder_chain argument and then prep_inputs handles fixing.
                # We need to ensure the PDB for MPNN has the target as chain A and binder as chain B if mpnn_fix_interface is used.
                # The best_model_pdb_for_mpnn is a complex. mpnn_gen_sequence expects a PDB file and the chain ID of the binder.

                # The original mpnn_gen_sequence takes trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings
                # Here, trajectory_pdb is best_model_pdb_for_mpnn
                # binder_chain is binder_chain_id ("B")
                # trajectory_interface_residues is interface_residues_for_mpnn

                print(f"Running ProteinMPNN for {eval_name} based on {best_model_pdb_for_mpnn}")
                mpnn_generated_sequences = mpnn_gen_sequence(
                    trajectory_pdb=best_model_pdb_for_mpnn,
                    binder_chain=binder_chain_id, # Assuming this is "B"
                    trajectory_interface_residues=interface_residues_for_mpnn,
                    advanced_settings=advanced_settings
                )

                # Filter out duplicate sequences that might already be in mpnn_variant_stats_csv (if resuming)
                # This check is similar to bindcraft.py's MPNN loop.
                # For a new script, this might be less critical unless appending to existing CSVs.
                # For now, we'll process all unique sequences generated in this run.

                # Create set of MPNN sequences with allowed amino acid composition & not already processed (if resuming, not implemented yet)
                # Similar to bindcraft.py logic:
                restricted_AAs_mpnn = set(aa.strip().upper() for aa in advanced_settings.get("omit_AAs","").split(',')) if advanced_settings.get("force_reject_AA_mpnn", advanced_settings.get("force_reject_AA",False)) else set()

                unique_mpnn_variants_to_process = []
                seen_mpnn_sequences = set() # For this batch of MPNN designs from one seed

                # mpnn_generated_sequences is a dict: {'seq': [seq1, ...], 'score': [score1,...], 'seqid': [seqid1,...]}
                num_mpnn_candidates = len(mpnn_generated_sequences['seq'])

                for i in range(num_mpnn_candidates):
                    seq = mpnn_generated_sequences['seq'][i][-peptide_length:] # Ensure correct length if MPNN outputs full complex
                    score = mpnn_generated_sequences['score'][i]
                    seqid = mpnn_generated_sequences['seqid'][i]

                    if seq in seen_mpnn_sequences:
                        continue
                    if restricted_AAs_mpnn and any(aa in seq.upper() for aa in restricted_AAs_mpnn):
                        print(f"MPNN sequence {seq} for {eval_name} rejected due to restricted AAs.")
                        continue

                    unique_mpnn_variants_to_process.append({'seq': seq, 'score': score, 'seqid': seqid})
                    seen_mpnn_sequences.add(seq)

                print(f"Generated {len(unique_mpnn_variants_to_process)} unique MPNN variants for {eval_name} after AA restriction.")
                total_mpnn_variants_generated += len(unique_mpnn_variants_to_process)

                # --- Stage 3: Evaluation of MPNN-Generated Variants ---
                accepted_mpnn_count_for_this_seed = 0
                max_mpnn_to_accept_per_seed = advanced_settings.get("max_mpnn_sequences", 3) # Default to 3

                for mpnn_variant_idx, variant_info in enumerate(unique_mpnn_variants_to_process):
                    if accepted_mpnn_count_for_this_seed >= max_mpnn_to_accept_per_seed:
                        print(f"Reached max_mpnn_sequences ({max_mpnn_to_accept_per_seed}) for seed {eval_name}. Stopping further MPNN variant processing for this seed.")
                        break

                    mpnn_seq = variant_info['seq']
                    mpnn_score_val = round(variant_info['score'], 2)
                    mpnn_seqid_val = round(variant_info['seqid'], 2)

                    mpnn_variant_eval_name = f"{eval_name}_MPNN_{mpnn_variant_idx+1}_L{len(mpnn_seq)}"
                    print(f"\nEvaluating MPNN Variant {mpnn_variant_idx+1}/{len(unique_mpnn_variants_to_process)} for {eval_name}: {mpnn_seq}")
                    stage3_time_start = time.time()

                    # Prepare AF2 models for this MPNN variant's length (usually same as parent)
                    current_mpnn_variant_length = len(mpnn_seq)
                    # This prep_inputs might be redundant if length hasn't changed from parent peptide_length
                    # but good practice if MPNN could alter length (it shouldn't here)
                    complex_prediction_model.prep_inputs(
                        pdb_filename=target_settings["starting_pdb"],
                        chain=target_settings["chains"],
                        binder_len=current_mpnn_variant_length,
                        rm_target_seq=advanced_settings["rm_template_seq_predict"],
                        rm_target_sc=advanced_settings["rm_template_sc_predict"]
                    )
                    binder_prediction_model.prep_inputs(length=current_mpnn_variant_length)

                    mpnn_complex_stats, pass_mpnn_af2_filters = predict_binder_complex(
                        model=complex_prediction_model,
                        binder_sequence=mpnn_seq,
                        mpnn_design_name=mpnn_variant_eval_name,
                        target_pdb=target_settings["starting_pdb"],
                        chain=target_settings["chains"],
                        length=current_mpnn_variant_length,
                        trajectory_pdb=best_model_pdb_for_mpnn, # Can use the seed PDB as a reference if needed by model's initial guess (though it's off)
                        prediction_models_to_run=prediction_models,
                        advanced_settings=advanced_settings,
                        filters_to_apply=filters,
                        design_paths_dict=design_paths,
                        failure_csv_path=failure_stats_csv,
                        output_pdb_dir=design_paths["MPNN_Eval/PDB"],
                        output_relaxed_pdb_dir=design_paths["MPNN_Eval/Relaxed"]
                    )

                    if not pass_mpnn_af2_filters:
                        print(f"MPNN variant {mpnn_variant_eval_name} failed initial AF2 filters. Skipping detailed scoring.")
                        log_data_mpnn = [mpnn_variant_eval_name, "mpnn_eval", current_mpnn_variant_length, None, None,
                                         target_settings['target_hotspot_residues'], mpnn_seq,
                                         None, mpnn_score_val, mpnn_seqid_val]
                        num_metric_placeholders_mpnn = len(design_labels) - len(log_data_mpnn) - 5
                        log_data_mpnn.extend([None] * num_metric_placeholders_mpnn)
                        log_data_mpnn.extend([format_time(time.time() - stage3_time_start), "Failed MPNN initial AF2 predict", settings_file_name, filters_file_name, advanced_file_name])
                        insert_data(mpnn_variant_stats_csv, log_data_mpnn)
                        continue

                    # Detailed scoring for MPNN variant
                    mpnn_detailed_complex_stats = deepcopy(mpnn_complex_stats)
                    mpnn_interface_residues_str = None

                    for model_idx_mpnn in prediction_models:
                        model_num_mpnn = model_idx_mpnn + 1
                        unrelaxed_pdb_mpnn = os.path.join(design_paths["MPNN_Eval/PDB"], f"{mpnn_variant_eval_name}_model{model_num_mpnn}.pdb")
                        relaxed_pdb_mpnn = os.path.join(design_paths["MPNN_Eval/Relaxed"], f"{mpnn_variant_eval_name}_model{model_num_mpnn}.pdb")

                        if not os.path.exists(relaxed_pdb_mpnn):
                            if model_num_mpnn in mpnn_detailed_complex_stats:
                                mpnn_detailed_complex_stats[model_num_mpnn].update({k: None for k in ['Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Target_RMSD']})
                            continue

                        cl_u_mpnn = calculate_clash_score(unrelaxed_pdb_mpnn)
                        cl_r_mpnn = calculate_clash_score(relaxed_pdb_mpnn)
                        rs_mpnn, raa_mpnn, rir_mpnn = score_interface(relaxed_pdb_mpnn, binder_chain_id)
                        ss_a_m, ss_b_m, ss_l_m, ss_ai_m, ss_bi_m, ss_li_m, iplddt_m, ssplddt_m = calc_ss_percentage(relaxed_pdb_mpnn, advanced_settings, binder_chain_id, is_relaxed=True)
                        tr_rmsd_m = target_pdb_rmsd(relaxed_pdb_mpnn, target_settings["starting_pdb"], target_settings["chains"])

                        # Hotspot RMSD for MPNN variant: compare to the initial accepted input peptide structure
                        hs_rmsd_m = unaligned_rmsd(best_model_pdb_for_mpnn, relaxed_pdb_mpnn, binder_chain_id, binder_chain_id) if best_model_pdb_for_mpnn else None


                        if model_num_mpnn not in mpnn_detailed_complex_stats: mpnn_detailed_complex_stats[model_num_mpnn] = {}
                        mpnn_detailed_complex_stats[model_num_mpnn].update({
                            'Unrelaxed_Clashes': cl_u_mpnn, 'Relaxed_Clashes': cl_r_mpnn,
                            'Binder_Energy_Score': rs_mpnn['binder_score'], 'Surface_Hydrophobicity': rs_mpnn['surface_hydrophobicity'],
                            'ShapeComplementarity': rs_mpnn['interface_sc'], 'PackStat': rs_mpnn['interface_packstat'],
                            'dG': rs_mpnn['interface_dG'], 'dSASA': rs_mpnn['interface_dSASA'],
                            'dG/dSASA': rs_mpnn['interface_dG_SASA_ratio'], 'Interface_SASA_%': rs_mpnn['interface_fraction'],
                            'Interface_Hydrophobicity': rs_mpnn['interface_hydrophobicity'], 'n_InterfaceResidues': rs_mpnn['interface_nres'],
                            'n_InterfaceHbonds': rs_mpnn['interface_interface_hbonds'], 'InterfaceHbondsPercentage': rs_mpnn['interface_hbond_percentage'],
                            'n_InterfaceUnsatHbonds': rs_mpnn['interface_delta_unsat_hbonds'], 'InterfaceUnsatHbondsPercentage': rs_mpnn['interface_delta_unsat_hbonds_percentage'],
                            'InterfaceAAs': raa_mpnn, 'InterfaceRes': rir_mpnn,
                            'i_pLDDT_Rosetta': iplddt_m, 'ss_pLDDT_Rosetta': ssplddt_m,
                            'Interface_Helix%': ss_ai_m, 'Interface_BetaSheet%': ss_bi_m, 'Interface_Loop%': ss_li_m,
                            'Binder_Helix%': ss_a_m, 'Binder_BetaSheet%': ss_b_m, 'Binder_Loop%': ss_l_m,
                            'Target_RMSD': tr_rmsd_m, 'Hotspot_RMSD': hs_rmsd_m
                        })
                        if mpnn_interface_residues_str is None: mpnn_interface_residues_str = rir_mpnn

                    mpnn_complex_averages = calculate_averages(mpnn_detailed_complex_stats, handle_aa=True)
                    if not mpnn_interface_residues_str and mpnn_complex_averages.get('InterfaceRes'):
                         mpnn_interface_residues_str = mpnn_complex_averages['InterfaceRes']

                    mpnn_binder_alone_stats = predict_binder_alone(
                        model=binder_prediction_model, binder_sequence=mpnn_seq, mpnn_design_name=mpnn_variant_eval_name,
                        length=current_mpnn_variant_length, trajectory_pdb=best_model_pdb_for_mpnn, # For RMSD alignment to original seed
                        binder_chain_id_in_traj=binder_chain_id, # Chain of binder in best_model_pdb_for_mpnn
                        prediction_models_to_run=prediction_models, advanced_settings=advanced_settings,
                        design_paths_dict=design_paths, output_pdb_dir=design_paths["MPNN_Eval/Binder"]
                    )
                    # Calculate Binder_RMSD for MPNN variants relative to the initial seed peptide's binder structure
                    for model_num_key_mpnn in mpnn_binder_alone_stats:
                        if isinstance(mpnn_binder_alone_stats[model_num_key_mpnn], dict):
                            mpnn_binder_pdb_path = os.path.join(design_paths["MPNN_Eval/Binder"], f"{mpnn_variant_eval_name}_model{model_num_key_mpnn}.pdb")
                            if os.path.exists(mpnn_binder_pdb_path) and best_model_pdb_for_mpnn and os.path.exists(best_model_pdb_for_mpnn):
                                rmsd_val = unaligned_rmsd(best_model_pdb_for_mpnn, mpnn_binder_pdb_path, binder_chain_id, "A") # Compare binder in seed PDB (chain B) to current binder alone (chain A)
                                mpnn_binder_alone_stats[model_num_key_mpnn]['Binder_RMSD'] = rmsd_val
                            else:
                                mpnn_binder_alone_stats[model_num_key_mpnn]['Binder_RMSD'] = None

                    mpnn_binder_alone_averages = calculate_averages(mpnn_binder_alone_stats)

                    mpnn_seq_notes = validate_design_sequence(mpnn_seq, mpnn_complex_averages.get('Relaxed_Clashes', float('inf')), advanced_settings)
                    stage3_time_taken = time.time() - stage3_time_start

                    log_data_mpnn = [
                        mpnn_variant_eval_name, "mpnn_eval", current_mpnn_variant_length, None, None, # Seed from input peptide, Helicity
                        target_settings['target_hotspot_residues'], mpnn_seq,
                        mpnn_interface_residues_str, mpnn_score_val, mpnn_seqid_val
                    ]
                    # Append complex stats for MPNN variant
                    for label in design_labels:
                        if label.startswith("Average_") and not label.startswith("Average_Binder_"):
                            log_data_mpnn.append(mpnn_complex_averages.get(label.replace("Average_",""), None))
                        elif label.startswith(tuple(f"{i+1}_" for i in range(5))) and not label.startswith(tuple(f"{i+1}_Binder_" for i in range(5))):
                            model_n_str, metric_k = label.split("_", 1)
                            log_data_mpnn.append(mpnn_detailed_complex_stats.get(int(model_n_str), {}).get(metric_k, None))
                    # Append binder stats for MPNN variant
                    for label in design_labels:
                        if label.startswith("Average_Binder_"):
                            log_data_mpnn.append(mpnn_binder_alone_averages.get(label.replace("Average_Binder_",""), None))
                        elif label.startswith(tuple(f"{i+1}_Binder_" for i in range(5))):
                            model_n_str, _, metric_k = label.split("_", 2)
                            log_data_mpnn.append(mpnn_binder_alone_stats.get(int(model_n_str), {}).get(metric_k, None))

                    current_log_len_mpnn = len(log_data_mpnn)
                    expected_len_before_footer_mpnn = 10 + len(metric_labels_in_design_labels)
                    if current_log_len_mpnn < expected_len_before_footer_mpnn:
                        log_data_mpnn.extend([None] * (expected_len_before_footer_mpnn - current_log_len_mpnn))
                    elif current_log_len_mpnn > expected_len_before_footer_mpnn:
                         log_data_mpnn = log_data_mpnn[:expected_len_before_footer_mpnn]

                    log_data_mpnn.extend([format_time(stage3_time_taken), mpnn_seq_notes, settings_file_name, filters_file_name, advanced_file_name])
                    insert_data(mpnn_variant_stats_csv, log_data_mpnn)

                    passes_stage3_filters = check_filters(log_data_mpnn, design_labels, filters)

                    best_model_num_mpnn_variant = None
                    highest_iptm_mpnn = -1.0
                    for model_n, stats in mpnn_detailed_complex_stats.items():
                        if isinstance(stats, dict) and stats.get('i_pTM', -1.0) > highest_iptm_mpnn:
                            highest_iptm_mpnn = stats['i_pTM']
                            best_model_num_mpnn_variant = model_n
                    if best_model_num_mpnn_variant is None and len(mpnn_detailed_complex_stats) > 0: # Fallback
                         best_model_num_mpnn_variant = list(mpnn_detailed_complex_stats.keys())[0]


                    if passes_stage3_filters == True and best_model_num_mpnn_variant is not None:
                        total_mpnn_variants_accepted += 1
                        accepted_mpnn_count_for_this_seed +=1
                        print(f"MPNN variant {mpnn_variant_eval_name} PASSED Stage 3 filters.")
                        best_pdb_mpnn_variant = os.path.join(design_paths["MPNN_Eval/Relaxed"], f"{mpnn_variant_eval_name}_model{best_model_num_mpnn_variant}.pdb")
                        if os.path.exists(best_pdb_mpnn_variant):
                            shutil.copy(best_pdb_mpnn_variant, os.path.join(design_paths["Accepted_MPNN_Variants"], f"{mpnn_variant_eval_name}_model{best_model_num_mpnn_variant}_accepted.pdb"))
                    elif best_model_num_mpnn_variant is not None: # Failed filters but had a best model
                        print(f"MPNN variant {mpnn_variant_eval_name} FAILED Stage 3 filters.")
                        pdb_to_reject_mpnn = os.path.join(design_paths["MPNN_Eval/Relaxed"], f"{mpnn_variant_eval_name}_model{best_model_num_mpnn_variant}.pdb")
                        if os.path.exists(pdb_to_reject_mpnn):
                            shutil.copy(pdb_to_reject_mpnn, os.path.join(design_paths["Rejected_MPNN_Variants"], f"{mpnn_variant_eval_name}_model{best_model_num_mpnn_variant}_rejected.pdb"))
                    else: # No best model and failed filters (likely prediction issue)
                         print(f"MPNN variant {mpnn_variant_eval_name} FAILED Stage 3 filters (no best model identified).")
                    gc.collect() # Clean up after each MPNN variant

                print(f"Finished MPNN processing for seed {eval_name}. Accepted {accepted_mpnn_count_for_this_seed} MPNN variants.")

        else: # This else is for: if passes_stage1_filters == True and best_model_num_for_mpnn is not None:
            print(f"Input peptide {eval_name} FAILED Stage 1 filters or no suitable model found for MPNN.")
            if best_model_num_for_mpnn is not None: # If there was a best model, even if it failed filters
                pdb_to_reject = os.path.join(design_paths["Input_Eval/Relaxed"], f"{eval_name}_model{best_model_num_for_mpnn}.pdb")
                if os.path.exists(pdb_to_reject):
                     shutil.copy(pdb_to_reject, os.path.join(design_paths["Rejected_Input_Peptides"], f"{eval_name}_model{best_model_num_for_mpnn}_rejected.pdb"))
            else: # No specific best model, maybe save all if any exist
                for model_idx_rej in prediction_models:
                    model_num_rej = model_idx_rej + 1
                    pdb_to_reject = os.path.join(design_paths["Input_Eval/Relaxed"], f"{eval_name}_model{model_num_rej}.pdb")
                    if os.path.exists(pdb_to_reject):
                        shutil.copy(pdb_to_reject, os.path.join(design_paths["Rejected_Input_Peptides"], f"{eval_name}_model{model_num_rej}_rejected_no_best.pdb"))

        gc.collect() # Clean up memory after processing each input peptide

    print("\n--- Finished Stage 1: Initial Evaluation of Input Peptides ---")


    # --- End of Script ---
    elapsed_time = time.time() - script_start_time
    elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"
    print(f"\n--- Pipeline Finished ---")
    print(f"Total input peptides processed: {total_input_peptides_processed}")
    print(f"Total input peptides accepted (Stage 1): {total_input_accepted_stage1}")
    print(f"Total MPNN variants generated: {total_mpnn_variants_generated}")
    print(f"Total MPNN variants accepted: {total_mpnn_variants_accepted}")
    print(f"Total script execution time: {elapsed_text}")

if __name__ == '__main__':
    main()
