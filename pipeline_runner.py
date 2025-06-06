import os
import cv2
import pandas as pd
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import PipelineConfig
from feature_extractor import FeatureExtractor
from anomaly_detector import AnomalyDetector
from utilz import derive_features, select_feature_columns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyDetectionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(config.yolo, config.bot_sort)
        self.anomaly_detector = AnomalyDetector(config.cadi)
        
    def process_frame(self, frame_path: str) -> List[Dict[str, Any]]:
        """Process a single frame and extract features."""
        try:
            frame_idx = int(os.path.splitext(os.path.basename(frame_path))[0])
            img = cv2.imread(frame_path)
            
            if img is None:
                logger.warning(f"Could not read image {frame_path}")
                return []
                
            features_list = self.feature_extractor.extract_features(img)
            
            for features in features_list:
                features['filename'] = os.path.basename(frame_path)
                features['frame_idx'] = frame_idx
                
            return features_list
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {e}")
            return []
            
    def process_folder(self, folder_path: str) -> pd.DataFrame:
        """Process all frames in a folder."""
        try:
            data = []
            frame_paths = []
            
            # Collect all valid frame paths
            for fname in sorted(os.listdir(folder_path)):
                if fname.startswith("._"):
                    continue
                frame_paths.append(os.path.join(folder_path, fname))
                
            # Process frames in parallel
            with ThreadPoolExecutor() as executor:
                future_to_path = {
                    executor.submit(self.process_frame, path): path 
                    for path in frame_paths
                }
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        features = future.result()
                        data.extend(features)
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                        
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {e}")
            return pd.DataFrame()
            
    def run(self):
        """Run the complete anomaly detection pipeline."""
        try:
            for folder_name in os.listdir(self.config.root_dir):
                if folder_name == "output":
                    continue
                    
                folder_path = os.path.join(self.config.root_dir, folder_name)
                
                if not os.path.isdir(folder_path):
                    continue
                    
                logger.info(f"Processing folder: {folder_name}")
                
                # Extract features
                df = self.process_folder(folder_path)
                if df.empty:
                    logger.warning(f"No features extracted from {folder_name}")
                    continue
                    
                # Derive additional features
                df = derive_features(df)
                
                # Process class probabilities
                class_prob_df = pd.DataFrame(df['class_probabilities'].tolist(), index=df.index)
                class_prob_df.columns = [f'class_prob_{i}' for i in range(class_prob_df.shape[1])]
                df = pd.concat([df, class_prob_df], axis=1)
                df.drop(columns=['class_probabilities'], inplace=True)
                
                # Select features for anomaly detection
                feature_cols = select_feature_columns(df)
                
                # Run anomaly detection
                df = self.anomaly_detector.run_all(df, feature_cols)
                
                # Save results
                output_path = os.path.join(self.config.output_dir, "scores", f"{folder_name}_scored.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved results to {output_path}")
                
        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            raise

def main():
    config = PipelineConfig()
    pipeline = AnomalyDetectionPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
