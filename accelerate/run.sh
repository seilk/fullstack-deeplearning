accelerate config # create a config file 
accelerate test --config_file configs/path_to_config.yaml # test the config file
accelerate launch --config_file configs/acc-config.yaml path_to_script.py --args_for_the_script # launch the script