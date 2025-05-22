# Multiple experts BC agent training
# python protomotions/train_agent_bc.py \
# +exp=full_body_tracker/mlp_single_motion_flat_terrain.yaml motion_file=multi_motion_bc.yaml \
# +robot=smpl +simulator=isaacgym \
# +experiment_name=bc_training_v0 +opt=wandb


python protomotions/train_agent_bc.py \
+exp=full_body_tracker/bc_mlp_single_motion_flat_terrain.yaml \
+robot=smpl +simulator=isaacgym motion_file=multi_motion_bc.yaml \
+experiment_name=bc_v0 +opt=wandb \
+expert_mapping_json=protomotions/config/expert/expert_mapping.json