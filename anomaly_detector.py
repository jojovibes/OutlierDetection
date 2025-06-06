import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from cadi.Src.forest import Forest
from cadi.Src.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

from config import CADIConfig

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, config: CADIConfig):
        self.config = config
        self.scaler = StandardScaler()
        
    def _prepare_features(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Prepare and scale features for anomaly detection."""
        try:
            X = df[feature_cols].copy()
            return self.scaler.fit_transform(X)
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
            
    def run_cadi(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Run CADI anomaly detection."""
        try:
            X = self._prepare_features(df, feature_cols)
            
            # Create temporary dataset for CADI
            dataset = Dataset(X)
            
            # Initialize and build forest
            forest = Forest(dataset, 
                          nbT=self.config.nb_trees,
                          method="cadi",
                          maxHeight=self.config.max_height)
            forest.build()
            
            # Run anomaly detection
            forest.anomalyDetection(binary=True, 
                                  contamination_rate=self.config.contamination_rate)
            forest.clustering()
            forest.explain_anomalies()
            
            # Update dataframe with results
            df["cadi_anomaly"] = np.zeros(len(df), dtype=int)
            df.loc[forest.anomalies, "cadi_anomaly"] = 1
            df["score_cadi"] = forest.scores
            df["cadi_cluster"] = forest.clusters_affectations
            df["cadi_explanation"] = [forest.explanations.get(i, "") for i in range(len(df))]
            
            return df
            
        except Exception as e:
            logger.error(f"Error in CADI detection: {e}")
            raise
            
    def run_gmm(self, df: pd.DataFrame, feature_cols: List[str], 
                n_components: int = 3) -> pd.DataFrame:
        """Run GMM-based anomaly detection."""
        try:
            X = self._prepare_features(df, feature_cols)
            
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            scores = -gmm.score_samples(X)
            
            # Normalize scores to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            
            df["score_gmm"] = scores
            df["gmm_anomaly"] = (scores > np.percentile(scores, 
                                                       (1 - self.config.contamination_rate) * 100)).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in GMM detection: {e}")
            raise
            
    def run_if(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Run Isolation Forest anomaly detection."""
        try:
            X = self._prepare_features(df, feature_cols)
            
            iforest = IsolationForest(contamination=self.config.contamination_rate,
                                    random_state=42)
            scores = -iforest.score_samples(X)
            
            # Normalize scores to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            
            df["score_if"] = scores
            df["if_anomaly"] = (scores > np.percentile(scores, 
                                                     (1 - self.config.contamination_rate) * 100)).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {e}")
            raise
            
    def run_all(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Run all anomaly detection methods and combine results."""
        try:
            df = self.run_cadi(df, feature_cols)
            df = self.run_gmm(df, feature_cols)
            df = self.run_if(df, feature_cols)
            
            # Compute ensemble score
            df["score_ensemble"] = (df["score_cadi"] + df["score_gmm"] + df["score_if"]) / 3
            df["ensemble_anomaly"] = (df["score_ensemble"] > 
                                    np.percentile(df["score_ensemble"], 
                                                (1 - self.config.contamination_rate) * 100)).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in ensemble detection: {e}")
            raise