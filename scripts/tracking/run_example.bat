@echo off
REM Example batch script for processing one CSV clip
REM Usage: run_example.bat

python scripts/tracking/main.py "d:\sleap_output\top_mouth_180\top_mouth_180_20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1905.506-1913.422.csv" --sleap-output-dir "d:\sleap_output" --output-dir "d:\sleap_output\processed"

pause
