import pandas as pd
import numpy as np
from typing import List, Dict, Any

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select relevant feature columns for anomaly detection."""
    # Exclude non-feature columns
    exclude_cols = ['track_id', 'filename', 'frame_idx', 'bbox', 'class_probabilities']
    
    # Get all numeric columns that are not excluded
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and 
                   pd.api.types.is_numeric_dtype(df[col])]
    
    return feature_cols

def compute_trajectory_features(track_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute advanced trajectory features from track history."""
    if len(track_history) < 2:
        return {
            'acceleration': 0.0,
            'smoothness': 0.0,
            'direction_change': 0.0
        }
        
    velocities = []
    directions = []
    
    for i in range(1, len(track_history)):
        prev_bbox = track_history[i-1]['bbox']
        curr_bbox = track_history[i]['bbox']
        
        dx = (curr_bbox[0] + curr_bbox[2]) / 2 - (prev_bbox[0] + prev_bbox[2]) / 2
        dy = (curr_bbox[1] + curr_bbox[3]) / 2 - (prev_bbox[1] + prev_bbox[3]) / 2
        
        velocity = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)
        
        velocities.append(velocity)
        directions.append(direction)
        
    # Compute acceleration
    accelerations = np.diff(velocities)
    avg_acceleration = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0.0
    
    # Compute trajectory smoothness
    direction_changes = np.diff(directions)
    smoothness = 1.0 - np.mean(np.abs(direction_changes)) / np.pi if len(direction_changes) > 0 else 0.0
    
    # Compute direction change rate
    direction_change_rate = np.mean(np.abs(direction_changes)) if len(direction_changes) > 0 else 0.0
    
    return {
        'acceleration': avg_acceleration,
        'smoothness': smoothness,
        'direction_change': direction_change_rate
    }

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional features from the base features."""
    try:
        # Group by track_id to compute trajectory features
        track_features = []
        for track_id, track_df in df.groupby('track_id'):
            track_history = track_df.sort_values('frame_idx').to_dict('records')
            features = compute_trajectory_features(track_history)
            features['track_id'] = track_id
            track_features.append(features)
            
        # Merge trajectory features back to main dataframe
        track_features_df = pd.DataFrame(track_features)
        df = df.merge(track_features_df, on='track_id', how='left')
        
        # Compute bounding box features
        df['bbox_width'] = df['bbox'].apply(lambda x: x[2] - x[0])
        df['bbox_height'] = df['bbox'].apply(lambda x: x[3] - x[1])
        df['bbox_area'] = df['bbox_width'] * df['bbox_height']
        df['bbox_aspect_ratio'] = df['bbox_width'] / df['bbox_height']
        
        # Compute motion features
        df['speed'] = df['velocity']
        df['acceleration_magnitude'] = df['acceleration']
        df['motion_smoothness'] = df['smoothness']
        
        # Fill NaN values with appropriate defaults
        df = df.fillna({
            'acceleration': 0.0,
            'smoothness': 1.0,
            'direction_change': 0.0,
            'bbox_width': 0.0,
            'bbox_height': 0.0,
            'bbox_area': 0.0,
            'bbox_aspect_ratio': 1.0,
            'speed': 0.0,
            'acceleration_magnitude': 0.0,
            'motion_smoothness': 1.0
        })
        
        return df
        
    except Exception as e:
        print(f"Error deriving features: {e}")
        return df