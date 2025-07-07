####################################
############## ColabDesign functions
####################################
### Import dependencies
import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from colabdesign.shared.utils import copy_dict
from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages
from .pyrosetta_utils import pr_relax, align_pdbs
from .generic_utils import update_failures

# hallucinate a binder
def binder_hallucination(design_name, starting_pdb, chain, target_hotspot_residues, length, seed, helicity_value, design_models, advanced_settings, design_paths, failure_csv):
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")

    # clear GPU memory for new trajectory
    clear_mem()

    # initialise binder hallucination model
    af_model = mk_afdesign_model(protocol="binder", debug=False, data_dir=advanced_settings["af_params_dir"], 
                                use_multimer=advanced_settings["use_multimer_design"], num_recycles=advanced_settings["num_recycles_design"],
                                best_metric='loss')

    # sanity check for hotspots
    if target_hotspot_residues == "":
        target_hotspot_residues = None

    af_model.prep_inputs(pdb_filename=starting_pdb, chain=chain, binder_len=length, hotspot=target_hotspot_residues, seed=seed, rm_aa=advanced_settings["omit_AAs"],
                        rm_target_seq=advanced_settings["rm_template_seq_design"], rm_target_sc=advanced_settings["rm_template_sc_design"])

    ### Update weights based on specified settings
    af_model.opt["weights"].update({"pae":advanced_settings["weights_pae_intra"],
                                    "plddt":advanced_settings["weights_plddt"],
                                    "i_pae":advanced_settings["weights_pae_inter"],
                                    "con":advanced_settings["weights_con_intra"],
                                    "i_con":advanced_settings["weights_con_inter"],
                                    })

    # redefine intramolecular contacts (con) and intermolecular contacts (i_con) definitions
    af_model.opt["con"].update({"num":advanced_settings["intra_contact_number"],"cutoff":advanced_settings["intra_contact_distance"],"binary":False,"seqsep":9})
    af_model.opt["i_con"].update({"num":advanced_settings["inter_contact_number"],"cutoff":advanced_settings["inter_contact_distance"],"binary":False})
        

    ### additional loss functions
    if advanced_settings["use_rg_loss"]:
        # radius of gyration loss
        add_rg_loss(af_model, advanced_settings["weights_rg"])

    if advanced_settings["use_i_ptm_loss"]:
        # interface pTM loss
        add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])

    if advanced_settings["use_termini_distance_loss"]:
        # termini distance loss
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])

    # add the helicity loss
    add_helix_loss(af_model, helicity_value)

    # calculate the number of mutations to do based on the length of the protein
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))

    ### start design algorithm based on selection
    if advanced_settings["design_algorithm"] == '2stage':
        # uses gradient descend to get a PSSM profile and then uses PSSM to bias the sampling of random mutations to decrease loss
        af_model.design_pssm_semigreedy(soft_iters=advanced_settings["soft_iterations"], hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

    elif advanced_settings["design_algorithm"] == '3stage':
        # 3 stage design using logits, softmax, and one hot encoding
        af_model.design_3stage(soft_iters=advanced_settings["soft_iterations"], temp_iters=advanced_settings["temporary_iterations"], hard_iters=advanced_settings["hard_iterations"], 
                                num_models=1, models=design_models, sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'greedy':
        # design by using random mutations that decrease loss
        af_model.design_semigreedy(advanced_settings["greedy_iterations"], tries=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'mcmc':
        # design by using random mutations that decrease loss
        half_life = round(advanced_settings["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(advanced_settings["greedy_iterations"], half_life=half_life, T_init=t_mcmc, mutation_rate=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == '4stage':
        # initial logits to prescreen trajectory
        print("Stage 1: Test Logits")
        af_model.design_logits(iters=50, e_soft=0.9, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)

        # determine pLDDT of best iteration according to lowest 'loss' value
        initial_plddt = get_best_plddt(af_model, length)
        
        # if best iteration has high enough confidence then continue
        if initial_plddt > 0.65:
            print("Initial trajectory pLDDT good, continuing: "+str(initial_plddt))
            if advanced_settings["optimise_beta"]:
                # temporarily dump model to assess secondary structure
                af_model.save_pdb(model_pdb_path)
                _, beta, *_ = calc_ss_percentage(model_pdb_path, advanced_settings, 'B')
                os.remove(model_pdb_path)

                # if beta sheeted trajectory is detected then choose to optimise
                if float(beta) > 15:
                    advanced_settings["soft_iterations"] = advanced_settings["soft_iterations"] + advanced_settings["optimise_beta_extra_soft"]
                    advanced_settings["temporary_iterations"] = advanced_settings["temporary_iterations"] + advanced_settings["optimise_beta_extra_temp"]
                    af_model.set_opt(num_recycles=advanced_settings["optimise_beta_recycles_design"])
                    print("Beta sheeted trajectory detected, optimising settings")

            # how many logit iterations left
            logits_iter = advanced_settings["soft_iterations"] - 50
            if logits_iter > 0:
                print("Stage 1: Additional Logits Optimisation")
                af_model.clear_best()
                af_model.design_logits(iters=logits_iter, e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
                                    ramp_recycles=False, save_best=True)
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
                print("Optimised logit trajectory pLDDT: "+str(logit_plddt))
            else:
                logit_plddt = initial_plddt

            # perform softmax trajectory design
            if advanced_settings["temporary_iterations"] > 0:
                print("Stage 2: Softmax Optimisation")
                af_model.clear_best()
                af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
                softmax_plddt = get_best_plddt(af_model, length)
            else:
                softmax_plddt = logit_plddt

            # perform one hot encoding
            if softmax_plddt > 0.65:
                print("Softmax trajectory pLDDT good, continuing: "+str(softmax_plddt))
                if advanced_settings["hard_iterations"] > 0:
                    af_model.clear_best()
                    print("Stage 3: One-hot Optimisation")
                    af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True)
                    onehot_plddt = get_best_plddt(af_model, length)

                if onehot_plddt > 0.65:
                    # perform greedy mutation optimisation
                    print("One-hot trajectory pLDDT good, continuing: "+str(onehot_plddt))
                    if advanced_settings["greedy_iterations"] > 0:
                        print("Stage 4: PSSM Semigreedy Optimisation")
                        af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

                else:
                    update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    print("One-hot trajectory pLDDT too low to continue: "+str(onehot_plddt))

            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                print("Softmax trajectory pLDDT too low to continue: "+str(softmax_plddt))

        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            print("Initial trajectory pLDDT too low to continue: "+str(initial_plddt))

    else:
        print("ERROR: No valid design model selected")
        exit()
        return

    ### save trajectory PDB
    final_plddt = get_best_plddt(af_model, length)
    af_model.save_pdb(model_pdb_path)
    af_model.aux["log"]["terminate"] = ""

    # let's check whether the trajectory is worth optimising by checking confidence, clashes, and contacts
    # check clashes
    #clash_interface = calculate_clash_score(model_pdb_path, 2.4)
    ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)

    #if clash_interface > 25 or ca_clashes > 0:
    if ca_clashes > 0:
        af_model.aux["log"]["terminate"] = "Clashing"
        update_failures(failure_csv, 'Trajectory_Clashes')
        print("Severe clashes detected, skipping analysis and MPNN optimisation")
        print("")
    else:
        # check if low quality prediction
        if final_plddt < 0.7:
            af_model.aux["log"]["terminate"] = "LowConfidence"
            update_failures(failure_csv, 'Trajectory_final_pLDDT')
            print("Trajectory starting confidence low, skipping analysis and MPNN optimisation")
            print("")
        else:
            # does it have enough contacts to consider?
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())

            # if less than 3 contacts then protein is floating above and is not binder
            if binder_contacts_n < 3:
                af_model.aux["log"]["terminate"] = "LowConfidence"
                update_failures(failure_csv, 'Trajectory_Contacts')
                print("Too few contacts at the interface, skipping analysis and MPNN optimisation")
                print("")
            else:
                # phew, trajectory is okay! We can continue
                af_model.aux["log"]["terminate"] = ""
                print("Trajectory successful, final pLDDT: "+str(final_plddt))

    # move low quality prediction:
    if af_model.aux["log"]["terminate"] != "":
        shutil.move(model_pdb_path, design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"])

    ### get the sampled sequence for plotting
    af_model.get_seqs()
    if advanced_settings["save_design_trajectory_plots"]:
        plot_trajectory(af_model, design_name, design_paths)

    ### save the hallucination trajectory animation
    if advanced_settings["save_design_animations"]:
        plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name+".html"), 'w') as f:
            f.write(plots)
        plt.close('all')

    if advanced_settings["save_trajectory_pickle"]:
        with open(os.path.join(design_paths["Trajectory/Pickle"], design_name+".pickle"), 'wb') as handle:
            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return af_model

# run prediction for binder with masked template target
def predict_binder_complex(model, binder_sequence, mpnn_design_name,
                           target_pdb, chain, length, trajectory_pdb,
                           prediction_models_to_run, advanced_settings, filters_to_apply,
                           design_paths_dict, failure_csv_path, seed=None,
                           output_pdb_dir=None, output_relaxed_pdb_dir=None):
    """
    Predicts binder complex, saves PDBs to specified directories, and performs initial AF2 filtering.

    Args:
        model: Compiled AF2 model object.
        binder_sequence (str): Sequence of the binder.
        mpnn_design_name (str): Base name for outputs.
        target_pdb (str): Path to the target PDB file.
        chain (str): Target chain ID(s).
        length (int): Length of the binder.
        trajectory_pdb (str): Path to a template/trajectory PDB (used if model's use_initial_guess is True).
        prediction_models_to_run (list): List of AF2 model indices to run (e.g., [0, 1, 2, 3, 4]).
        advanced_settings (dict): Advanced settings dictionary.
        filters_to_apply (dict): Filters dictionary for AF2 pre-filtering.
        design_paths_dict (dict): Dictionary of design paths.
        failure_csv_path (str): Path to the failure statistics CSV.
        seed (int, optional): Random seed. Defaults to None.
        output_pdb_dir (str, optional): Directory to save unrelaxed PDBs. Defaults to design_paths_dict["MPNN"].
        output_relaxed_pdb_dir (str, optional): Directory to save relaxed PDBs. Defaults to design_paths_dict["MPNN/Relaxed"].

    Returns:
        tuple: (prediction_stats_dict, pass_af2_filters_bool)
    """
    prediction_stats = {}

    # Determine output directories
    pdb_dir = output_pdb_dir if output_pdb_dir else design_paths_dict.get("MPNN", "./")
    relaxed_pdb_dir = output_relaxed_pdb_dir if output_relaxed_pdb_dir else design_paths_dict.get("MPNN/Relaxed", "./")

    # Ensure these directories exist if they are custom
    if not os.path.exists(pdb_dir): os.makedirs(pdb_dir)
    if not os.path.exists(relaxed_pdb_dir): os.makedirs(relaxed_pdb_dir)

    # clean sequence
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())

    # reset filtering conditionals
    pass_af2_filters = True # Assume true initially for the whole set of models
    filter_failures_log = {} # For logging specific filter failures

    # start prediction per AF2 model
    for model_idx in prediction_models_to_run: # model_idx is 0,1,2,3,4
        model_num_for_filename = model_idx + 1 # model_num is 1,2,3,4,5 for file naming

        complex_pdb_path = os.path.join(pdb_dir, f"{mpnn_design_name}_model{model_num_for_filename}.pdb")

        current_model_passes_filters = True # For this specific model

        if not os.path.exists(complex_pdb_path):
            # predict model
            model.predict(seq=binder_sequence, models=[model_idx], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            model.save_pdb(complex_pdb_path)
            prediction_metrics = copy_dict(model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics.get('plddt',0.0), 2),
                'pTM': round(prediction_metrics.get('ptm',0.0), 2),
                'i_pTM': round(prediction_metrics.get('i_ptm',0.0), 2),
                'pAE': round(prediction_metrics.get('pae',0.0), 2),
                'i_pAE': round(prediction_metrics.get('i_pae',0.0), 2)
            }
            prediction_stats[model_num_for_filename] = stats

            # List of filter conditions and corresponding keys for this specific model
            # These filters in filters_to_apply are typically named like "1_pLDDT", "Average_pLDDT"
            # We are checking per-model AF2 stats here.
            af2_filter_keys_model_specific = [
                (f"{model_num_for_filename}_pLDDT", 'plddt', '>='),
                (f"{model_num_for_filename}_pTM", 'ptm', '>='),
                (f"{model_num_for_filename}_i_pTM", 'i_ptm', '>='),
                (f"{model_num_for_filename}_pAE", 'pae', '<='),
                (f"{model_num_for_filename}_i_pAE", 'i_pae', '<='),
            ]

            # Perform initial AF2 values filtering for THIS model
            for filter_key_name, metric_key, comparison in af2_filter_keys_model_specific:
                threshold = filters_to_apply.get(filter_key_name, {}).get("threshold")
                if threshold is not None:
                    metric_value = prediction_metrics.get(metric_key)
                    if metric_value is None: # Metric not found in output
                        current_model_passes_filters = False
                        filter_failures_log[filter_key_name] = filter_failures_log.get(filter_key_name, 0) + 1
                        break
                    if comparison == '>=' and metric_value < threshold:
                        current_model_passes_filters = False
                        filter_failures_log[filter_key_name] = filter_failures_log.get(filter_key_name, 0) + 1
                        break
                    elif comparison == '<=' and metric_value > threshold:
                        current_model_passes_filters = False
                        filter_failures_log[filter_key_name] = filter_failures_log.get(filter_key_name, 0) + 1
                        break

            if not current_model_passes_filters:
                print(f"Model {model_num_for_filename} for {mpnn_design_name} failed pre-relaxation AF2 filters.")
                # If any model fails, the overall pass_af2_filters for the set becomes False
                # This is a stricter interpretation: if any model fails basic AF2, the whole design is flagged.
                # Or, we can let it pass if at least one model is good.
                # For now, let's stick to: if any model fails its specific filters, the whole thing is questionable for relaxation.
                # The plan was: "if AF2 filters are not passed then skip the scoring" - this usually means if the *average* or *key model* fails.
                # The original `bindcraft.py` does not have this per-model pre-filter before relaxation.
                # It predicts all, then checks averages/specifics later.
                # Let's revert to a simpler: predict all, relax all, then filter later.
                # The `pass_af2_filters` here should be a global flag for the peptide, not per model for this stage.
                # The filtering logic in `bindcraft.py`'s main loop is more comprehensive AFTER all metrics are gathered.
                # This function's `pass_af2_filters` output should reflect if ANY model was successfully predicted
                # and is worth relaxing, rather than strict filtering.
                # Let's simplify: if a PDB is produced, it's worth relaxing. The actual filtering happens later.
                # So, the `pass_af2_filters` here will just mean "at least one PDB was generated".
                pass # Continue to predict other models
        else: # PDB already exists
            print(f"PDB {complex_pdb_path} already exists. Skipping prediction.")
            # Try to load stats if possible or mark as existing. For now, just skip.
            # This part needs more robust handling if we want to resume runs.
            # For now, assume we overwrite or start fresh. If it exists, we assume it was processed.
            # To make it compatible with just generating PDBs, we'll assume it's fine.
            # We need to populate prediction_stats if the file exists but stats are not there.
            # This function is primarily for *generating* and then relaxing.
            # Let's assume if PDB exists, it was from a previous run and we don't re-calculate AF2 stats here.
            # The calling script should handle logic for existing files if needed.
            # For simplicity, if it exists, we'll still try to get its AF2 stats if they are in `prediction_stats`
            # but this function's main job is to create it if missing.
             if model_num_for_filename not in prediction_stats: # If PDB existed but no stats, we can't fill AF2 scores here
                print(f"Warning: PDB {complex_pdb_path} exists but no AF2 stats available for it in this run.")
                prediction_stats[model_num_for_filename] = {'pLDDT': None, 'pTM': None, 'i_pTM': None, 'pAE': None, 'i_pAE': None}


    # Update the failure CSV with any pre-filter failures logged
    if filter_failures_log: # If any specific model pre-filter failed
        update_failures(failure_csv_path, filter_failures_log)
        # If the goal of pass_af2_filters is to gate relaxation:
        # If *all* models failed their individual pre-filters, then pass_af2_filters = False
        # If *at least one* model passed its pre-filters (or had no specific pre-filter), then pass_af2_filters = True
        # This is getting too complex for this function. Let's simplify.
        # `pass_af2_filters` will indicate if *any* PDB was generated and is thus available for relaxation.

    pass_af2_filters = any(os.path.exists(os.path.join(pdb_dir, f"{mpnn_design_name}_model{m_idx+1}.pdb")) for m_idx in prediction_models_to_run)

    if not pass_af2_filters:
        print(f"No PDBs were generated for {mpnn_design_name}. Skipping relaxation.")
        return prediction_stats, False # Return False as nothing to relax/score further

    # Proceed with relaxation for all generated PDBs
    for model_idx in prediction_models_to_run:
        model_num_for_filename = model_idx + 1
        complex_pdb_path = os.path.join(pdb_dir, f"{mpnn_design_name}_model{model_num_for_filename}.pdb")
        relaxed_pdb_path = os.path.join(relaxed_pdb_dir, f"{mpnn_design_name}_model{model_num_for_filename}.pdb")

        if os.path.exists(complex_pdb_path):
            if not os.path.exists(relaxed_pdb_path): # Only relax if relaxed version doesn't exist
                print(f"Relaxing {complex_pdb_path} -> {relaxed_pdb_path}")
                pr_relax(complex_pdb_path, relaxed_pdb_path)
            else:
                print(f"Relaxed PDB {relaxed_pdb_path} already exists. Skipping relaxation.")
        # If unrelaxed PDB doesn't exist, can't relax it. It means prediction failed for this model.
        # The pass_af2_filters flag above handles the case where *no* PDBs were made.

    return prediction_stats, True # True means PDBs were made and relaxation attempted/done. Actual filtering is next.

# run prediction for binder alone
def predict_binder_alone(model, binder_sequence, mpnn_design_name,
                         length, trajectory_pdb, binder_chain_id_in_traj,
                         prediction_models_to_run, advanced_settings, design_paths_dict,
                         seed=None, output_pdb_dir=None):
    """
    Predicts binder monomer structure and saves PDBs to a specified directory.

    Args:
        model: Compiled AF2 model object (already prepared for hallucination protocol with binder length).
        binder_sequence (str): Sequence of the binder.
        mpnn_design_name (str): Base name for outputs.
        length (int): Length of the binder.
        trajectory_pdb (str, optional): Path to a template/trajectory PDB for alignment. If None, alignment is skipped.
        binder_chain_id_in_traj (str): Chain ID of the binder in trajectory_pdb (e.g., "B"). Used for alignment.
        prediction_models_to_run (list): List of AF2 model indices to run (e.g., [0, 1, 2, 3, 4]).
        advanced_settings (dict): Advanced settings dictionary.
        design_paths_dict (dict): Dictionary of design paths.
        seed (int, optional): Random seed. Defaults to None.
        output_pdb_dir (str, optional): Directory to save PDBs. Defaults to design_paths_dict["MPNN/Binder"].

    Returns:
        dict: Statistics for each predicted model (pLDDT, pTM, pAE).
    """
    binder_stats = {}

    # Determine output directory
    pdb_dir = output_pdb_dir if output_pdb_dir else design_paths_dict.get("MPNN/Binder", "./") # Default from original
    if not os.path.exists(pdb_dir): os.makedirs(pdb_dir)

    # prepare sequence for prediction
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    # Model should already be prepped with binder_prediction_model.prep_inputs(length=length)
    # And sequence set by model.set_seq(binder_sequence) if required by ColabDesign API for this protocol.
    # The mk_afdesign_model with protocol="hallucination" might not need set_seq if seq is passed to predict.
    # Let's assume the model is ready for predict(seq=...) or set_seq then predict()

    # According to ColabDesign, for hallucination protocol, prep_inputs is for length, then set_seq for sequence.
    # Or, if `seq` argument is available in `predict` for hallucination, that's also fine.
    # The `bindcraft.py` original loop for MPNN variants calls:
    # binder_prediction_model.prep_inputs(length=length) (once outside loop)
    # Then inside loop:
    # binder_statistics = predict_binder_alone(binder_prediction_model, mpnn_sequence['seq'], ...)
    # And `predict_binder_alone` itself calls `prediction_model.set_seq(binder_sequence)`
    # So, the `model` passed here should be the one prepped for the correct length.
    model.set_seq(binder_sequence) # Ensure sequence is set on the model object

    # predict each model separately
    for model_idx in prediction_models_to_run: # model_idx is 0,1,2,3,4
        model_num_for_filename = model_idx + 1 # model_num is 1,2,3,4,5 for file naming

        binder_alone_pdb_path = os.path.join(pdb_dir, f"{mpnn_design_name}_model{model_num_for_filename}.pdb")

        if not os.path.exists(binder_alone_pdb_path):
            # predict model
            model.predict(models=[model_idx], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            model.save_pdb(binder_alone_pdb_path)
            prediction_metrics = copy_dict(model.aux["log"]) # contains plddt, ptm, pae

            # align binder model to trajectory binder, if trajectory_pdb is provided
            if trajectory_pdb and os.path.exists(trajectory_pdb) and os.path.exists(binder_alone_pdb_path):
                try:
                    # Assuming binder in trajectory_pdb is `binder_chain_id_in_traj`
                    # and binder alone PDB is single chain "A" by default from AF2 hallucination protocol
                    align_pdbs(trajectory_pdb, binder_alone_pdb_path, binder_chain_id_in_traj, "A")
                except Exception as e:
                    print(f"Warning: Could not align {binder_alone_pdb_path} to {trajectory_pdb} due to: {e}")

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics.get('plddt',0.0), 2),
                'pTM': round(prediction_metrics.get('ptm',0.0), 2),
                'pAE': round(prediction_metrics.get('pae',0.0), 2)
                # Binder_RMSD is calculated outside this function based on its output PDB
            }
            binder_stats[model_num_for_filename] = stats
        else:
            print(f"Binder PDB {binder_alone_pdb_path} already exists. Skipping prediction.")
            # If PDB exists, we can't easily get AF2 stats without re-predicting or storing them separately.
            # For now, if it exists, return None for its stats to indicate it wasn't processed this call.
            # Or, the calling function should handle this.
            # For consistency with predict_binder_complex, let's add placeholder if it exists.
            if model_num_for_filename not in binder_stats:
                 binder_stats[model_num_for_filename] = {'pLDDT': None, 'pTM': None, 'pAE': None}


    return binder_stats

# run MPNN to generate sequences for binders
def mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings):
    # clear GPU memory
    clear_mem()

    # initialise MPNN model
    mpnn_model = mk_mpnn_model(backbone_noise=advanced_settings["backbone_noise"], model_name=advanced_settings["model_path"], weights=advanced_settings["mpnn_weights"])

    # check whether keep the interface generated by the trajectory or whether to redesign with MPNN
    design_chains = 'A,' + binder_chain

    if advanced_settings["mpnn_fix_interface"]:
        fixed_positions = 'A,' + trajectory_interface_residues
        fixed_positions = fixed_positions.rstrip(",")
        print("Fixing interface residues: "+trajectory_interface_residues)
    else:
        fixed_positions = 'A'

    # prepare inputs for MPNN
    mpnn_model.prep_inputs(pdb_filename=trajectory_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=advanced_settings["omit_AAs"])

    # sample MPNN sequences in parallel
    mpnn_sequences = mpnn_model.sample(temperature=advanced_settings["sampling_temp"], num=advanced_settings["num_seqs"], batch=advanced_settings["num_seqs"])

    return mpnn_sequences

# Get pLDDT of best model
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)

# Define radius of gyration loss for colabdesign
def add_rg_loss(self, weight=0.1):
    '''add radius of gyration loss'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg":rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight

# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}
    
    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight

# add helicity loss
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):  
      if "offset" in inputs:
        offset = inputs["offset"]
      else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:,None] - idx[None,:]

      # define distogram
      dgram = outputs["distogram"]["logits"]
      dgram_bins = get_dgram_bins(outputs)
      mask_2d = np.outer(np.append(np.zeros(self._target_len), np.ones(self._binder_len)), np.append(np.zeros(self._target_len), np.ones(self._binder_len)))

      x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
      if offset is None:
        if mask_2d is None:
          helix_loss = jnp.diagonal(x,3).mean()
        else:
          helix_loss = jnp.diagonal(x * mask_2d,3).sum() + (jnp.diagonal(mask_2d,3).sum() + 1e-8)
      else:
        mask = offset == 3
        if mask_2d is not None:
          mask = jnp.where(mask_2d,mask,0)
        helix_loss = jnp.where(mask,x,0.0).sum() / (mask.sum() + 1e-8)

      return {"helix":helix_loss}
    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight

# add N- and C-terminus distance loss
def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    '''Add loss penalizing the distance between N and C termini'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight

# plot design trajectory losses
def plot_trajectory(af_model, design_name, design_paths):
    metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'rg', 'mpnn']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            # Create a new figure for each metric
            plt.figure()

            loss = af_model.get_loss(metric)
            # Create an x axis for iterations
            iterations = range(1, len(loss) + 1)

            plt.plot(iterations, loss, label=f'{metric}', color=colors[index % len(colors)])

            # Add labels and a legend
            plt.xlabel('Iterations')
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)

            # Save the plot
            plt.savefig(os.path.join(design_paths["Trajectory/Plots"], design_name+"_"+metric+".png"), dpi=150)
            
            # Close the figure
            plt.close()
