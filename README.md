Pipeline Summary

1. **Data Preparation**
   - Run `data_shanghai.py` to organize the ShanghaiTech dataset
   - Run `extract_shanghai.py` to extract detection and tracking metadata using YOLOv7 + BoT-SORT

2. **Model Training & Scoring**
   - Use `unified_runner.py`:
     - First in **train mode** to fit the models (e.g., GMM, CADI and IF)
     - Then in **test mode** to generate anomaly scores
   - Update input database and output paths accordingly
   - Models are stored in the models folder

3. **Evaluation**
   - `frame_eval.py`: for frame-level AUC and AP metrics
   - `clean_eval.py`: for detection-level evaluation (e.g., AUC, RBDC, TBDC, ect.)
   - Additional metric-specific scripts are available in the `eval/` folder

4. **Visualization**
   - `visualize.py` overlays original frames with:
     - Detection bounding boxes
     - Ground-truth masks
     - Anomaly score heatmaps
