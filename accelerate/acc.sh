accelerate config # create a config file 
accelerate test --config_file path_to_config.yaml # test the config file
accelerate launch --config_file path_to_config.yaml path_to_script.py --args_for_the_script # launch the script