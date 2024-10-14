## Landau fitting over ToT spectrum:
Edit the board configuration file according to the currently analyzed board. Then:
```
python3 landaumvp.py <board_config_file_name>.yaml -s
```

## To produce just the calibration plots:
```
python3 calib.py <board_config_file_name>.yaml -s
```
Note: During the calibration log fitting, RuntimeWarnings may pop up, because the range of parameters is not constrained at the moment. Using the -s option suppresses these warnings.

## Extra requirements:
landaupy: https://github.com/SengerM/landaupy
pyyaml
