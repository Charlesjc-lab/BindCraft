####################################
###################### BindCraft Run
####################################
### Import dependencies
from functions import *

# Check if JAX-capable GPU is available, otherwise exit
check_jax_gpu()

######################################
### parse input paths
parser = argparse.ArgumentParser(description='Script to run BindCraft binder design.')

parser.add_argument('--settings', '-s', type=str, required=True,
                    help='Path to the basic settings.json file. Required.')
parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                    help='Path to the filters.json file used to filter design. If not provided, default will be used.')
parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer.json',
                    help='Path to the advanced.json file with additional design settings. If not provided, default will be used.')
parser.add_argument('--peptide_list_file', '-p', type=str, default=None,
                    help='Path to a text file containing peptide sequences (one per line) for direct evaluation. If provided, binder design/hallucination is skipped.')

args = parser.parse_args()

# perform checks of input setting files
settings_path, filters_path, advanced_path = perform_input_check(args)

# Peptide evaluation mode
peptide_evaluation_mode = False
input_peptide_sequences = []
if args.peptide_list_file:
    peptide_evaluation_mode = True
    try:
        with open(args.peptide_list_file, 'r') as f:
            input_peptide_sequences = [line.strip() for line in f if line.strip()]
        if not input_peptide_sequences:
            print(f"Warning: Peptide list file {args.peptide_list_file} is empty or contains only whitespace.")
            # Potentially exit or handle as an error depending on desired behavior
        else:
            print(f"Running in peptide evaluation mode with {len(input_peptide_sequences)} sequences from {args.peptide_list_file}.")
    except FileNotFoundError:
        print(f"Error: Peptide list file {args.peptide_list_file} not found. Exiting.")
        exit()
    except Exception as e:
        print(f"Error reading peptide list file {args.peptide_list_file}: {e}. Exiting.")
        exit()

### load settings from JSON
target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)

settings_file = os.path.basename(settings_path).split('.')[0]
filters_file = os.path.basename(filters_path).split('.')[0]
advanced_file = os.path.basename(advanced_path).split('.')[0]

### load AF2 model settings
design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])

### perform checks on advanced_settings
bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

### generate directories, design path names can be found within the function
design_paths = generate_directories(target_settings["design_path"])

### generate dataframes
trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

create_dataframe(trajectory_csv, trajectory_labels)
create_dataframe(mpnn_csv, design_labels)
create_dataframe(final_csv, final_labels)
generate_filter_pass_csv(failure_csv, args.filters)

####################################
####################################
####################################
### initialise PyRosetta
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
print(f"Running binder design for target {settings_file}")
print(f"Design settings used: {advanced_file}")
print(f"Filtering designs based on {filters_file}")

####################################
# initialise counters
script_start_time = time.time()
trajectory_n = 1
accepted_designs = 0

if not peptide_evaluation_mode:
    ### start design loop (original behavior)
    while True:
        ### check if we have the target number of binders
        final_designs_reached = check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels)

        if final_designs_reached:
            # stop design loop execution
            break

        ### check if we reached maximum allowed trajectories
        max_trajectories_reached = check_n_trajectories(design_paths, advanced_settings)

        if max_trajectories_reached:
            break

        ### Initialise design
        # measure time to generate design
        trajectory_start_time = time.time()

        # generate random seed to vary designs
        seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

        # sample binder design length randomly from defined distribution
        # In peptide_evaluation_mode, length is derived from input peptide
        samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
        length = np.random.choice(samples)

        # load desired helicity value to sample different secondary structure contents
        helicity_value = load_helicity(advanced_settings)

        # generate design name and check if same trajectory was already run
        design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
        trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
        trajectory_exists = any(os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs)

        if not trajectory_exists:
            print("Starting trajectory: "+design_name)

            ### Begin binder hallucination
            trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"],
                                                target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                                design_models, advanced_settings, design_paths, failure_csv)
            trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"]) # contains plddt, ptm, i_ptm, pae, i_pae
            trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")

            # round the metrics to two decimal places
            trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}

            # time trajectory
            trajectory_time = time.time() - trajectory_start_time
            trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
            print("Starting trajectory took: "+trajectory_time_text)
            print("")

            # Proceed if there is no trajectory termination signal
            if trajectory.aux["log"]["terminate"] == "":
                # Relax binder to calculate statistics
                trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
                pr_relax(trajectory_pdb, trajectory_relaxed)

                # define binder chain, placeholder in case multi-chain parsing in ColabDesign gets changed
                binder_chain = "B"

                # Calculate clashes before and after relaxation
                num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
                num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

                # secondary structure content of starting trajectory binder and interface
                trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_i_plddt, trajectory_ss_plddt = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

                # analyze interface scores for relaxed af2 trajectory
                trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(trajectory_relaxed, binder_chain)

                # starting binder sequence
                trajectory_sequence = trajectory.get_seq(get_best=True)[0]

                # analyze sequence
                traj_seq_notes = validate_design_sequence(trajectory_sequence, num_clashes_relaxed, advanced_settings)

                # target structure RMSD compared to input PDB
                trajectory_target_rmsd = target_pdb_rmsd(trajectory_pdb, target_settings["starting_pdb"], target_settings["chains"])

                # save trajectory statistics into CSV
                trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], trajectory_sequence, trajectory_interface_residues,
                                    trajectory_metrics['plddt'], trajectory_metrics['ptm'], trajectory_metrics['i_ptm'], trajectory_metrics['pae'], trajectory_metrics['i_pae'],
                                    trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed, trajectory_interface_scores['binder_score'],
                                    trajectory_interface_scores['surface_hydrophobicity'], trajectory_interface_scores['interface_sc'], trajectory_interface_scores['interface_packstat'],
                                    trajectory_interface_scores['interface_dG'], trajectory_interface_scores['interface_dSASA'], trajectory_interface_scores['interface_dG_SASA_ratio'],
                                    trajectory_interface_scores['interface_fraction'], trajectory_interface_scores['interface_hydrophobicity'], trajectory_interface_scores['interface_nres'], trajectory_interface_scores['interface_interface_hbonds'],
                                    trajectory_interface_scores['interface_hbond_percentage'], trajectory_interface_scores['interface_delta_unsat_hbonds'], trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
                                    trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_interface_AA, trajectory_target_rmsd,
                                    trajectory_time_text, traj_seq_notes, settings_file, filters_file, advanced_file]
                insert_data(trajectory_csv, trajectory_data)

                if advanced_settings["enable_mpnn"]:
                    # initialise MPNN counters
                    mpnn_n = 1
                    accepted_mpnn = 0
                    mpnn_dict = {}
                    design_start_time = time.time()

                    ### MPNN redesign of starting binder
                    mpnn_trajectories = mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings)
                    existing_mpnn_sequences = set(pd.read_csv(mpnn_csv, usecols=['Sequence'])['Sequence'].values)

                    # create set of MPNN sequences with allowed amino acid composition
                    restricted_AAs = set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(',')) if advanced_settings["force_reject_AA"] else set()

                    mpnn_sequences_to_process = sorted({ # Renamed to avoid conflict
                        mpnn_trajectories['seq'][n][-length:]: {
                            'seq': mpnn_trajectories['seq'][n][-length:],
                            'score': mpnn_trajectories['score'][n],
                            'seqid': mpnn_trajectories['seqid'][n]
                        } for n in range(advanced_settings["num_seqs"])
                        if (not restricted_AAs or not any(aa in mpnn_trajectories['seq'][n][-length:].upper() for aa in restricted_AAs))
                        and mpnn_trajectories['seq'][n][-length:] not in existing_mpnn_sequences
                    }.values(), key=lambda x: x['score'])

                    del existing_mpnn_sequences

                    # check whether any sequences are left after amino acid rejection and duplication check, and if yes proceed with prediction
                    if mpnn_sequences_to_process: # Use new name
                        # add optimisation for increasing recycles if trajectory is beta sheeted
                        if advanced_settings["optimise_beta"] and float(trajectory_beta) > 15:
                            advanced_settings["num_recycles_validation"] = advanced_settings["optimise_beta_recycles_valid"]

                        ### Compile prediction models once for faster prediction of MPNN sequences
                        clear_mem()
                        # compile complex prediction model
                        complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=advanced_settings["num_recycles_validation"], data_dir=advanced_settings["af_params_dir"],
                                                                    use_multimer=multimer_validation, use_initial_guess=advanced_settings["predict_initial_guess"], use_initial_atom_pos=advanced_settings["predict_bigbang"])
                        if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
                            complex_prediction_model.prep_inputs(pdb_filename=trajectory_pdb, chain='A', binder_chain='B', binder_len=length, use_binder_template=True, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                                                rm_target_sc=advanced_settings["rm_template_sc_predict"], rm_template_ic=True)
                        else:
                            complex_prediction_model.prep_inputs(pdb_filename=target_settings["starting_pdb"], chain=target_settings["chains"], binder_len=length, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                                                rm_target_sc=advanced_settings["rm_template_sc_predict"])

                        # compile binder monomer prediction model
                        binder_prediction_model = mk_afdesign_model(protocol="hallucination", use_templates=False, initial_guess=False,
                                                                    use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"],
                                                                    data_dir=advanced_settings["af_params_dir"], use_multimer=multimer_validation)
                        binder_prediction_model.prep_inputs(length=length)

                        # iterate over designed sequences
                        for mpnn_sequence_item in mpnn_sequences_to_process: # Use new name and item
                            mpnn_time = time.time()
                            current_peptide_sequence = mpnn_sequence_item['seq'] # Extract sequence
                            mpnn_score_val = round(mpnn_sequence_item['score'],2) # Extract score
                            mpnn_seqid_val = round(mpnn_sequence_item['seqid'],2) # Extract seqid

                            # generate mpnn design name numbering
                            mpnn_design_name = design_name + "_mpnn" + str(mpnn_n)
                            
                            # add design to dictionary
                            mpnn_dict[mpnn_design_name] = {'seq': current_peptide_sequence, 'score': mpnn_score_val, 'seqid': mpnn_seqid_val}

                            # save fasta sequence
                            if advanced_settings["save_mpnn_fasta"] is True:
                                save_fasta(mpnn_design_name, current_peptide_sequence, design_paths)

                            ### Predict mpnn redesigned binder complex using masked templates
                            mpnn_complex_statistics, pass_af2_filters = predict_binder_complex(complex_prediction_model,
                                                                                            current_peptide_sequence, mpnn_design_name,
                                                                                            target_settings["starting_pdb"], target_settings["chains"],
                                                                                            length, trajectory_pdb, prediction_models, advanced_settings,
                                                                                            filters, design_paths, failure_csv)

                            # if AF2 filters are not passed then skip the scoring
                            if not pass_af2_filters:
                                print(f"Base AF2 filters not passed for {mpnn_design_name}, skipping interface scoring")
                                mpnn_n += 1
                                continue

                            # calculate statistics for each model individually
                            for model_num in prediction_models:
                                mpnn_design_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                                mpnn_design_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                                if os.path.exists(mpnn_design_pdb):
                                    # Calculate clashes before and after relaxation
                                    num_clashes_mpnn = calculate_clash_score(mpnn_design_pdb)
                                    num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)

                                    # analyze interface scores for relaxed af2 trajectory
                                    # In peptide evaluation mode, trajectory_interface_residues might not be well-defined initially.
                                    # This is called after predict_binder_complex, so mpnn_design_relaxed should exist.
                                    mpnn_interface_scores, mpnn_interface_AA, current_interface_residues = score_interface(mpnn_design_relaxed, binder_chain)


                                    # secondary structure content of starting trajectory binder
                                    mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface, mpnn_beta_interface, mpnn_loops_interface, mpnn_i_plddt, mpnn_ss_plddt = calc_ss_percentage(mpnn_design_pdb, advanced_settings, binder_chain)

                                    # unaligned RMSD calculate to determine if binder is in the designed binding site
                                    # In peptide mode, trajectory_pdb might not be directly comparable if we skip hallucination for that exact peptide.
                                    # This will need careful handling in step 3. For now, we might pass None or a generic reference.
                                    rmsd_site = unaligned_rmsd(trajectory_pdb, mpnn_design_pdb, binder_chain, binder_chain) if trajectory_pdb and os.path.exists(trajectory_pdb) else None


                                    # calculate RMSD of target compared to input PDB
                                    target_rmsd = target_pdb_rmsd(mpnn_design_pdb, target_settings["starting_pdb"], target_settings["chains"])

                                    # add the additional statistics to the mpnn_complex_statistics dictionary
                                    mpnn_complex_statistics[model_num+1].update({
                                        'i_pLDDT': mpnn_i_plddt,
                                        'ss_pLDDT': mpnn_ss_plddt,
                                        'Unrelaxed_Clashes': num_clashes_mpnn,
                                        'Relaxed_Clashes': num_clashes_mpnn_relaxed,
                                        'Binder_Energy_Score': mpnn_interface_scores['binder_score'],
                                        'Surface_Hydrophobicity': mpnn_interface_scores['surface_hydrophobicity'],
                                        'ShapeComplementarity': mpnn_interface_scores['interface_sc'],
                                        'PackStat': mpnn_interface_scores['interface_packstat'],
                                        'dG': mpnn_interface_scores['interface_dG'],
                                        'dSASA': mpnn_interface_scores['interface_dSASA'],
                                        'dG/dSASA': mpnn_interface_scores['interface_dG_SASA_ratio'],
                                        'Interface_SASA_%': mpnn_interface_scores['interface_fraction'],
                                        'Interface_Hydrophobicity': mpnn_interface_scores['interface_hydrophobicity'],
                                        'n_InterfaceResidues': mpnn_interface_scores['interface_nres'],
                                        'n_InterfaceHbonds': mpnn_interface_scores['interface_interface_hbonds'],
                                        'InterfaceHbondsPercentage': mpnn_interface_scores['interface_hbond_percentage'],
                                        'n_InterfaceUnsatHbonds': mpnn_interface_scores['interface_delta_unsat_hbonds'],
                                        'InterfaceUnsatHbondsPercentage': mpnn_interface_scores['interface_delta_unsat_hbonds_percentage'],
                                        'InterfaceAAs': mpnn_interface_AA,
                                        'Interface_Helix%': mpnn_alpha_interface,
                                        'Interface_BetaSheet%': mpnn_beta_interface,
                                        'Interface_Loop%': mpnn_loops_interface,
                                        'Binder_Helix%': mpnn_alpha,
                                        'Binder_BetaSheet%': mpnn_beta,
                                        'Binder_Loop%': mpnn_loops,
                                        'Hotspot_RMSD': rmsd_site,
                                        'Target_RMSD': target_rmsd
                                    })

                                    # save space by removing unrelaxed predicted mpnn complex pdb?
                                    if advanced_settings["remove_unrelaxed_complex"]:
                                        os.remove(mpnn_design_pdb)

                            # calculate complex averages
                            mpnn_complex_averages = calculate_averages(mpnn_complex_statistics, handle_aa=True)

                            ### Predict binder alone in single sequence mode
                            # trajectory_pdb might not be relevant here for peptide mode.
                            binder_statistics = predict_binder_alone(binder_prediction_model, current_peptide_sequence, mpnn_design_name, length,
                                                                    None, binder_chain, prediction_models, advanced_settings, design_paths) # Pass None for trajectory_pdb

                            # extract RMSDs of binder to the original trajectory
                            for model_num in prediction_models:
                                mpnn_binder_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                                if os.path.exists(mpnn_binder_pdb):
                                    # This RMSD might need re-evaluation if trajectory_pdb is not from direct hallucination of this peptide.
                                    rmsd_binder = unaligned_rmsd(trajectory_pdb, mpnn_binder_pdb, binder_chain, "A") if trajectory_pdb and os.path.exists(trajectory_pdb) else None
                                else:
                                    rmsd_binder = None


                                # append to statistics
                                binder_statistics[model_num+1].update({
                                        'Binder_RMSD': rmsd_binder
                                    })

                                # save space by removing binder monomer models?
                                if advanced_settings["remove_binder_monomer"]:
                                    os.remove(mpnn_binder_pdb)

                            # calculate binder averages
                            binder_averages = calculate_averages(binder_statistics)

                            # analyze sequence to make sure there are no cysteins and it contains residues that absorb UV for detection
                            seq_notes = validate_design_sequence(current_peptide_sequence, mpnn_complex_averages.get('Relaxed_Clashes', None), advanced_settings)

                            # measure time to generate design
                            mpnn_end_time = time.time() - mpnn_time
                            elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(mpnn_end_time // 3600), int((mpnn_end_time % 3600) // 60), int(mpnn_end_time % 60))}"


                            # Insert statistics about MPNN design into CSV, will return None if corresponding model does note exist
                            model_numbers = range(1, 6)
                            statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                                                'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                                                'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                                                'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

                            # Initialize mpnn_data with the non-statistical data
                            # Seed and helicity_value might be less relevant or need default values for peptide evaluation mode
                            mpnn_data = [mpnn_design_name, "peptide_evaluation", length, seed if 'seed' in locals() else 0, helicity_value if 'helicity_value' in locals() else 0.0, target_settings["target_hotspot_residues"], current_peptide_sequence, current_interface_residues if 'current_interface_residues' in locals() else None, mpnn_score_val, mpnn_seqid_val]


                            # Add the statistical data for mpnn_complex
                            for label in statistics_labels:
                                mpnn_data.append(mpnn_complex_averages.get(label, None))
                                for model in model_numbers:
                                    mpnn_data.append(mpnn_complex_statistics.get(model, {}).get(label, None))

                            # Add the statistical data for binder
                            for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:  # These are the labels for binder alone
                                mpnn_data.append(binder_averages.get(label, None))
                                for model in model_numbers:
                                    mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

                            # Add the remaining non-statistical data
                            mpnn_data.extend([elapsed_mpnn_text, seq_notes, settings_file, filters_file, advanced_file])

                            # insert data into csv
                            insert_data(mpnn_csv, mpnn_data)

                            # find best model number by pLDDT
                            # Adjusted range for plddt_values based on new mpnn_data structure if needed; assuming it's still valid
                            plddt_values_for_best_model = {k: mpnn_complex_statistics.get(k, {}).get('pLDDT', -1.0) for k in prediction_models}
                            if not plddt_values_for_best_model or all(v == -1.0 for v in plddt_values_for_best_model.values()):
                                print(f"Warning: Could not determine best model for {mpnn_design_name} due to missing pLDDT values. Skipping filter check or using model 1 as default.")
                                best_model_number = 1 # Default or handle error
                            else:
                                best_model_number = max(plddt_values_for_best_model, key=plddt_values_for_best_model.get) + 1


                            best_model_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{best_model_number}.pdb")
                            if not os.path.exists(best_model_pdb): # Check if best_model_pdb exists before filter check
                                print(f"Warning: Best model PDB {best_model_pdb} not found for {mpnn_design_name}. Skipping filter check and acceptance.")
                            else:
                                # run design data against filter thresholds
                                filter_conditions_met = check_filters(mpnn_data, design_labels, filters) # Renamed variable
                                if filter_conditions_met == True:
                                    print(mpnn_design_name+" passed all filters")
                                    accepted_mpnn += 1
                                    accepted_designs += 1

                                    # copy designs to accepted folder
                                    shutil.copy(best_model_pdb, design_paths["Accepted"])

                                    # insert data into final csv
                                    final_data = [''] + mpnn_data # Assuming first column in final_csv is an index or placeholder
                                    insert_data(final_csv, final_data)

                                    # copy animation from accepted trajectory - SKIPPING for peptide_evaluation_mode
                                    # if advanced_settings["save_design_animations"]:
                                    #     accepted_animation = os.path.join(design_paths["Accepted/Animation"], f"{design_name}.html")
                                    #     if not os.path.exists(accepted_animation) and os.path.exists(os.path.join(design_paths["Trajectory/Animation"], f"{design_name}.html")):
                                    #         shutil.copy(os.path.join(design_paths["Trajectory/Animation"], f"{design_name}.html"), accepted_animation)

                                    # copy plots of accepted trajectory - SKIPPING for peptide_evaluation_mode
                                    # plot_files = os.listdir(design_paths["Trajectory/Plots"])
                                    # plots_to_copy = [f for f in plot_files if f.startswith(design_name) and f.endswith('.png')]
                                    # for accepted_plot in plots_to_copy:
                                    #     source_plot = os.path.join(design_paths["Trajectory/Plots"], accepted_plot)
                                    #     target_plot = os.path.join(design_paths["Accepted/Plots"], accepted_plot)
                                    #     if not os.path.exists(target_plot):
                                    #         shutil.copy(source_plot, target_plot)

                                else:
                                    print(f"Unmet filter conditions for {mpnn_design_name}")
                                    failure_df = pd.read_csv(failure_csv)
                                    special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
                                    incremented_columns = set()

                                    for column in filter_conditions_met: # Use the returned list of failed conditions
                                        base_column = column
                                        for prefix in special_prefixes:
                                            if column.startswith(prefix):
                                                base_column = column.split('_', 1)[1]

                                        if base_column not in incremented_columns:
                                            if base_column in failure_df.columns:
                                                failure_df[base_column] = failure_df[base_column] + 1
                                            else:
                                                print(f"Warning: Column {base_column} not found in failure_csv.csv")
                                            incremented_columns.add(base_column)

                                    failure_df.to_csv(failure_csv, index=False)
                                    if os.path.exists(best_model_pdb): # Ensure PDB exists before copying
                                     shutil.copy(best_model_pdb, design_paths["Rejected"])

                            # increase MPNN design number
                            mpnn_n += 1

                            # if enough mpnn sequences of the same trajectory pass filters then stop
                            if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
                                break

                        if accepted_mpnn >= 1:
                            print("Found "+str(accepted_mpnn)+" MPNN designs passing filters")
                            print("")
                        else:
                            print("No accepted MPNN designs found for this trajectory.")
                            print("")

                    else:
                        print('Duplicate MPNN designs sampled with different trajectory, skipping current trajectory optimisation')
                        print("")

                    # save space by removing unrelaxed design trajectory PDB
                    if advanced_settings["remove_unrelaxed_trajectory"] and trajectory_pdb and os.path.exists(trajectory_pdb):
                        os.remove(trajectory_pdb)


                    # measure time it took to generate designs for one trajectory
                    design_time = time.time() - design_start_time
                    design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
                    print("Design and validation of trajectory "+design_name+" took: "+design_time_text)
                else: # This 'else' corresponds to 'if advanced_settings["enable_mpnn"]:'
                    # Code to handle case where MPNN is disabled, directly processing trajectory_sequence
                    # This part will be largely similar to the MPNN loop but uses trajectory_sequence
                    # and won't have mpnn_score or mpnn_seqid.
                    # This section needs to be filled in or adapted if peptide_evaluation_mode
                    # should also work when enable_mpnn is false.
                    # For now, assuming peptide_evaluation_mode implies a flow similar to MPNN processing.
                    print("MPNN is disabled. Original trajectory sequence would be processed here.")


            # analyse the rejection rate of trajectories to see if we need to readjust the design weights
            if trajectory_n >= advanced_settings["start_monitoring"] and advanced_settings["enable_rejection_check"]:
                acceptance = accepted_designs / trajectory_n
                if not acceptance >= advanced_settings["acceptance_rate"]:
                    print("The ratio of successful designs is lower than defined acceptance rate! Consider changing your design settings!")
                    print("Script execution stopping...")
                    break

        # increase trajectory number
        trajectory_n += 1
        gc.collect()

else: # This is the new block for peptide_evaluation_mode
    print("Starting peptide evaluation mode...")
    peptide_idx = 0
    # Compile models once before the loop
    # Note: 'length' will vary per peptide, so prep_inputs might need to be in the loop or handled carefully.
    # For now, we prepare models and will call prep_inputs inside the loop for each peptide.
    clear_mem()
    # Compile complex prediction model (generic setup, specific prep_inputs later)
    # We will remove predict_initial_guess and predict_bigbang for simplicity in peptide evaluation mode,
    # as trajectory_pdb (initial guess structure) is not generated.
    # Users wanting this can run normally or we can add more complex input options later.
    actual_predict_initial_guess = advanced_settings["predict_initial_guess"]
    actual_predict_bigbang = advanced_settings["predict_bigbang"]
    if peptide_evaluation_mode:
        print("Peptide evaluation mode: predict_initial_guess and predict_bigbang will be set to False for AF2 complex prediction.")
        actual_predict_initial_guess = False
        actual_predict_bigbang = False

    complex_prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
        use_initial_guess=actual_predict_initial_guess, # Effectively False if peptide_evaluation_mode
        use_initial_atom_pos=actual_predict_bigbang # Effectively False if peptide_evaluation_mode
    )

    # Compile binder monomer prediction model (generic setup)
    binder_prediction_model = mk_afdesign_model(
        protocol="hallucination", use_templates=False, initial_guess=False,
        use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"], use_multimer=multimer_validation
    )
    binder_chain = "B" # Assuming binder is chain B, consistent with original script

    for current_peptide_sequence in input_peptide_sequences:
        peptide_idx += 1
        length = len(current_peptide_sequence)
        # Sanitize sequence
        current_peptide_sequence = re.sub("[^A-Z]", "", current_peptide_sequence.upper())
        if not current_peptide_sequence:
            print(f"Skipping invalid or empty sequence at index {peptide_idx-1}.")
            continue

        # Generate a design name for the input peptide
        design_name_prefix = target_settings["binder_name"] + "_input_peptide_" + str(peptide_idx)
        # Using a fixed seed or index for "seed" part of name for consistency with original naming, though less relevant here.
        # helicity_value is also less relevant for direct evaluation.
        seed_val = peptide_idx
        helicity_val = 0.0 # Default or placeholder

        print(f"Processing input peptide {peptide_idx}/{len(input_peptide_sequences)}: {current_peptide_sequence} (Length: {length})")
        peptide_eval_time_start = time.time()

        # Prepare complex prediction model for current peptide length
        # No trajectory_pdb for initial guess here.
        complex_prediction_model.prep_inputs(
            pdb_filename=target_settings["starting_pdb"],
            chain=target_settings["chains"],
            binder_len=length,
            rm_target_seq=advanced_settings["rm_template_seq_predict"],
            rm_target_sc=advanced_settings["rm_template_sc_predict"]
        )
        # Prepare binder monomer model for current peptide length
        binder_prediction_model.prep_inputs(length=length)

        # Generate a unique name for this evaluation, similar to mpnn_design_name
        eval_design_name = f"{design_name_prefix}_l{length}_s{seed_val}"

        # For peptide evaluation, there's no prior "trajectory" PDB.
        # predict_binder_complex's trajectory_pdb argument is used if use_initial_guess is True.
        # Since we are setting it to False for peptide_evaluation_mode, we can pass None.
        # However, the function signature requires it. We can pass target_settings["starting_pdb"]
        # as a placeholder, but it won't be used for initial guess if use_initial_guess is False.
        # Let's ensure predict_binder_complex handles this.
        # The `trajectory_pdb` is also used by `predict_binder_alone` for alignment if provided.
        # We'll pass None to `predict_binder_alone` for `trajectory_pdb`.

        # No MPNN score/seqid for input peptides
        mpnn_score = None
        mpnn_seqid = None
        # No trajectory_interface_residues from a hallucination step. This will be calculated after scoring.
        # We will get `current_interface_residues` from `score_interface` later.

        ### Predict binder complex
        peptide_complex_statistics, pass_af2_filters = predict_binder_complex(
            complex_prediction_model,
            current_peptide_sequence,
            eval_design_name, # mpnn_design_name equivalent
            target_settings["starting_pdb"],
            target_settings["chains"],
            length,
            target_settings["starting_pdb"], # Placeholder for trajectory_pdb, not used if initial_guess is False
            prediction_models, advanced_settings,
            filters, design_paths, failure_csv
        )

        if not pass_af2_filters:
            print(f"Base AF2 filters not passed for {eval_design_name}, skipping further processing.")
            # Log failure or minimal info to mpnn_csv? For now, matches original skip.
            continue

        # Initialize dicts for collecting stats over models
        processed_complex_stats = {}
        processed_binder_stats = {}

        # Placeholder for interface residues, will be updated per model if structure allows
        current_interface_residues_str = None

        for model_num_idx, model_num_val in enumerate(prediction_models): # model_num_val is 0,1,2,3,4
            model_id = model_num_val + 1 # 1,2,3,4,5 for file naming and dict keys

            eval_design_pdb = os.path.join(design_paths["MPNN"], f"{eval_design_name}_model{model_id}.pdb")
            eval_design_relaxed_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{eval_design_name}_model{model_id}.pdb")

            if os.path.exists(eval_design_pdb): # This PDB is created by predict_binder_complex
                # (Relaxation is also done inside predict_binder_complex if pass_af2_filters is True)

                # Calculate clashes (already done by predict_binder_complex if it calls pr_relax and score_interface)
                # We need to ensure these are calculated if not already.
                # The original code calculates these after predict_binder_complex loop.
                # Let's assume predict_binder_complex has relaxed the PDB to eval_design_relaxed_pdb.

                num_clashes_unrelaxed = calculate_clash_score(eval_design_pdb)
                num_clashes_relaxed = calculate_clash_score(eval_design_relaxed_pdb)

                # Score interface for the relaxed PDB
                interface_scores, interface_AA_comp, model_interface_residues_str = score_interface(eval_design_relaxed_pdb, binder_chain)
                if current_interface_residues_str is None : # Take from the first valid model
                    current_interface_residues_str = model_interface_residues_str


                # Secondary structure
                ss_alpha, ss_beta, ss_loops, ss_alpha_interface, ss_beta_interface, ss_loops_interface, i_plddt_val, ss_plddt_val = calc_ss_percentage(eval_design_pdb, advanced_settings, binder_chain)

                # Target RMSD
                target_rmsd_val = target_pdb_rmsd(eval_design_pdb, target_settings["starting_pdb"], target_settings["chains"])

                # Hotspot RMSD: No direct 'trajectory_pdb' from hallucination for input peptides.
                # This metric might be 'None' or calculated against a reference if defined.
                # For now, set to None.
                hotspot_rmsd_val = None

                # Update the per-model stats in peptide_complex_statistics (which should already have AF2 scores)
                stats_for_model = peptide_complex_statistics.get(model_id, {})
                stats_for_model.update({
                    'i_pLDDT': i_plddt_val, 'ss_pLDDT': ss_plddt_val,
                    'Unrelaxed_Clashes': num_clashes_unrelaxed, 'Relaxed_Clashes': num_clashes_relaxed,
                    'Binder_Energy_Score': interface_scores['binder_score'],
                    'Surface_Hydrophobicity': interface_scores['surface_hydrophobicity'],
                    'ShapeComplementarity': interface_scores['interface_sc'],
                    'PackStat': interface_scores['interface_packstat'],
                    'dG': interface_scores['interface_dG'], 'dSASA': interface_scores['interface_dSASA'],
                    'dG/dSASA': interface_scores['interface_dG_SASA_ratio'],
                    'Interface_SASA_%': interface_scores['interface_fraction'],
                    'Interface_Hydrophobicity': interface_scores['interface_hydrophobicity'],
                    'n_InterfaceResidues': interface_scores['interface_nres'],
                    'n_InterfaceHbonds': interface_scores['interface_interface_hbonds'],
                    'InterfaceHbondsPercentage': interface_scores['interface_hbond_percentage'],
                    'n_InterfaceUnsatHbonds': interface_scores['interface_delta_unsat_hbonds'],
                    'InterfaceUnsatHbondsPercentage': interface_scores['interface_delta_unsat_hbonds_percentage'],
                    'InterfaceAAs': interface_AA_comp,
                    'Interface_Helix%': ss_alpha_interface, 'Interface_BetaSheet%': ss_beta_interface, 'Interface_Loop%': ss_loops_interface,
                    'Binder_Helix%': ss_alpha, 'Binder_BetaSheet%': ss_beta, 'Binder_Loop%': ss_loops,
                    'Hotspot_RMSD': hotspot_rmsd_val, 'Target_RMSD': target_rmsd_val
                })
                peptide_complex_statistics[model_id] = stats_for_model


                if advanced_settings["remove_unrelaxed_complex"]:
                    os.remove(eval_design_pdb)
            else:
                print(f"Warning: Unrelaxed PDB {eval_design_pdb} not found for model {model_id} of {eval_design_name}. Skipping detailed scoring for this model.")


        # Calculate complex averages from peptide_complex_statistics
        peptide_complex_averages = calculate_averages(peptide_complex_statistics, handle_aa=True)

        ### Predict binder alone
        # Passing None for trajectory_pdb as no direct hallucinated reference exists for input peptides.
        current_binder_statistics = predict_binder_alone(
            binder_prediction_model, current_peptide_sequence, eval_design_name, length,
            None, binder_chain, prediction_models, advanced_settings, design_paths
        )

        for model_num_idx, model_num_val in enumerate(prediction_models):
            model_id = model_num_val + 1
            peptide_binder_pdb = os.path.join(design_paths["MPNN/Binder"], f"{eval_design_name}_model{model_id}.pdb")

            binder_rmsd_val = None # No trajectory to compare against for input peptides directly
            if os.path.exists(peptide_binder_pdb):
                 # Binder_RMSD: Original calculates RMSD to trajectory_pdb.
                 # For input peptides, this is undefined or needs a different reference. Set to None.
                if advanced_settings["remove_binder_monomer"]:
                    os.remove(peptide_binder_pdb)

            stats_for_binder_model = current_binder_statistics.get(model_id, {})
            stats_for_binder_model.update({'Binder_RMSD': binder_rmsd_val})
            current_binder_statistics[model_id] = stats_for_binder_model


        peptide_binder_averages = calculate_averages(current_binder_statistics)

        # Sequence validation notes
        seq_notes = validate_design_sequence(current_peptide_sequence, peptide_complex_averages.get('Relaxed_Clashes', None), advanced_settings)

        eval_time_end = time.time() - peptide_eval_time_start
        elapsed_eval_text = f"{'%d hours, %d minutes, %d seconds' % (int(eval_time_end // 3600), int((eval_time_end % 3600) // 60), int(eval_time_end % 60))}"

        # Assemble data for CSV logging (similar to mpnn_data)
        # Some fields like 'seed', 'helicity_value', 'mpnn_score', 'mpnn_seqid' are placeholders or defaults
        # 'trajectory_interface_residues' is now 'current_interface_residues_str' from scoring the first valid model.
        log_data = [
            eval_design_name, "peptide_evaluation", length, seed_val, helicity_val,
            target_settings["target_hotspot_residues"], current_peptide_sequence,
            current_interface_residues_str, # This is the string of interface residues like "B52,B53..."
            mpnn_score, mpnn_seqid
        ]

        model_ids_for_csv = range(1, 6) # Max 5 models expected in CSV
        csv_stat_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                           'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                           'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                           'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

        for label in csv_stat_labels:
            log_data.append(peptide_complex_averages.get(label, None))
            for mid in model_ids_for_csv:
                log_data.append(peptide_complex_statistics.get(mid, {}).get(label, None))

        binder_alone_labels = ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']
        for label in binder_alone_labels:
            log_data.append(peptide_binder_averages.get(label, None))
            for mid in model_ids_for_csv:
                log_data.append(current_binder_statistics.get(mid, {}).get(label, None))

        log_data.extend([elapsed_eval_text, seq_notes, settings_file, filters_file, advanced_file])
        insert_data(mpnn_csv, log_data) # Log to mpnn_csv for now

        # Determine best model for filter check (e.g., based on average i_pTM or specific model's i_pTM)
        # Original uses pLDDT of complex models.
        plddt_values_for_best_model = {k: peptide_complex_statistics.get(k, {}).get('pLDDT', -1.0) for k in peptide_complex_statistics if isinstance(peptide_complex_statistics.get(k), dict)}

        if not plddt_values_for_best_model or all(v == -1.0 for v in plddt_values_for_best_model.values()):
            print(f"Warning: Could not determine best model for {eval_design_name} due to missing pLDDTs. Using model 1 as default or skipping acceptance.")
            best_model_id_for_acceptance = 1 # Default to model 1 if no pLDDTs
        else:
            # key is model_id (1-5)
            best_model_id_for_acceptance = max(plddt_values_for_best_model, key=plddt_values_for_best_model.get)

        best_model_relaxed_pdb_path = os.path.join(design_paths["MPNN/Relaxed"], f"{eval_design_name}_model{best_model_id_for_acceptance}.pdb")

        if not os.path.exists(best_model_relaxed_pdb_path):
            print(f"Warning: Best model PDB {best_model_relaxed_pdb_path} not found for {eval_design_name}. Skipping filter check and acceptance.")
        else:
            filter_pass_conditions = check_filters(log_data, design_labels, filters)
            if filter_pass_conditions == True:
                print(f"{eval_design_name} passed all filters.")
                accepted_designs += 1
                shutil.copy(best_model_relaxed_pdb_path, design_paths["Accepted"])
                final_log_data = [''] + log_data # Prep for final_csv
                insert_data(final_csv, final_log_data)
                # Skipping animation/plot copying for peptide evaluation mode
            else:
                print(f"Unmet filter conditions for {eval_design_name}: {filter_pass_conditions}")
                # Log failure reasons
                failure_df = pd.read_csv(failure_csv)
                special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
                incremented_columns = set()
                for column_name in filter_pass_conditions: # filter_pass_conditions is list of failed metric names
                    base_column_name = column_name
                    for prefix in special_prefixes:
                        if column_name.startswith(prefix):
                            base_column_name = column_name.split('_', 1)[1]

                    if base_column_name not in incremented_columns:
                        if base_column_name in failure_df.columns:
                            failure_df[base_column_name] = failure_df[base_column_name] + 1
                        else:
                             print(f"Warning: Metric {base_column_name} not found as a column in {failure_csv}. Cannot increment failure count.")
                        incremented_columns.add(base_column_name)
                failure_df.to_csv(failure_csv, index=False)
                if os.path.exists(best_model_relaxed_pdb_path): # Ensure PDB exists before copying
                    shutil.copy(best_model_relaxed_pdb_path, design_paths["Rejected"])
        gc.collect()
    print(f"Finished peptide evaluation. Accepted {accepted_designs} designs.")


### Script finished
elapsed_time = time.time() - script_start_time
elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"
-print("Finished all designs. Script execution for "+str(trajectory_n)+" trajectories took: "+elapsed_text)
+if not peptide_evaluation_mode:
+    print("Finished all designs. Script execution for "+str(trajectory_n)+" trajectories took: "+elapsed_text)
+else:
+    print(f"Finished all peptide evaluations. Total accepted designs: {accepted_designs}. Script execution took: " + elapsed_text)