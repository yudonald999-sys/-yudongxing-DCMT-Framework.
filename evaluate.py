"""
CMCE Evaluation Script

Evaluates models on the Cross-Modal Chunking Evaluation benchmark.

Usage:
    python evaluation/evaluate.py --model_path checkpoints/dcmt_best.pt --dataset CMCE

Author: Dongxing Yu
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm


class CMCEEvaluator:
    """
    Evaluator for the Cross-Modal Chunking Evaluation benchmark.
    
    Metrics:
        - Chunk Detection F1: Precision/recall for boundary detection
        - Alignment Accuracy: Correct cross-modal chunk pairings
        - Boundary IoU: Intersection over union for chunk boundaries
        - CMCE Score: Weighted combination of above metrics
    """
    
    def __init__(
        self,
        chunk_threshold: float = 0.5,
        alignment_threshold: float = 0.5
    ):
        self.chunk_threshold = chunk_threshold
        self.alignment_threshold = alignment_threshold
        
    def compute_chunk_f1(
        self,
        pred_boundaries: np.ndarray,
        gt_boundaries: np.ndarray,
        tolerance: int = 2
    ) -> Dict[str, float]:
        """
        Compute F1 score for chunk boundary detection.
        
        Args:
            pred_boundaries: Predicted boundary positions
            gt_boundaries: Ground truth boundary positions
            tolerance: Position tolerance for matching
            
        Returns:
            Dict with precision, recall, and F1
        """
        # Find boundary positions
        pred_pos = np.where(pred_boundaries > self.chunk_threshold)[0]
        gt_pos = np.where(gt_boundaries > 0.5)[0]
        
        if len(pred_pos) == 0 and len(gt_pos) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        if len(pred_pos) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        if len(gt_pos) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Match predictions to ground truth within tolerance
        matched_pred = set()
        matched_gt = set()
        
        for p in pred_pos:
            for g in gt_pos:
                if abs(p - g) <= tolerance and g not in matched_gt:
                    matched_pred.add(p)
                    matched_gt.add(g)
                    break
        
        precision = len(matched_pred) / len(pred_pos)
        recall = len(matched_gt) / len(gt_pos)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def compute_alignment_accuracy(
        self,
        pred_alignment: np.ndarray,
        gt_alignment: list
    ) -> float:
        """
        Compute accuracy of cross-modal chunk alignments.
        
        Args:
            pred_alignment: Predicted alignment matrix [N_v, N_t]
            gt_alignment: List of ground truth alignments
            
        Returns:
            Alignment accuracy
        """
        correct = 0
        total = len(gt_alignment)
        
        for align in gt_alignment:
            v_idx = align['visual_chunk']
            t_idx = align['text_chunk']
            
            # Check if predicted alignment matches
            if pred_alignment[v_idx, t_idx] > self.alignment_threshold:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def compute_boundary_iou(
        self,
        pred_chunks: list,
        gt_chunks: list
    ) -> float:
        """
        Compute IoU for chunk boundaries (visual or textual).
        
        Args:
            pred_chunks: List of predicted chunk spans
            gt_chunks: List of ground truth chunk spans
            
        Returns:
            Mean IoU across chunks
        """
        ious = []
        
        for gt in gt_chunks:
            best_iou = 0
            for pred in pred_chunks:
                # Compute IoU
                if 'bbox' in gt:  # Visual chunks
                    gt_box = gt['bbox']
                    pred_box = pred['bbox']
                    
                    x1 = max(gt_box[0], pred_box[0])
                    y1 = max(gt_box[1], pred_box[1])
                    x2 = min(gt_box[2], pred_box[2])
                    y2 = min(gt_box[3], pred_box[3])
                    
                    intersection = max(0, x2-x1) * max(0, y2-y1)
                    gt_area = (gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
                    pred_area = (pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1])
                    union = gt_area + pred_area - intersection
                    
                    iou = intersection / union if union > 0 else 0
                else:  # Text chunks
                    gt_span = set(range(gt['start'], gt['end']))
                    pred_span = set(range(pred['start'], pred['end']))
                    
                    intersection = len(gt_span & pred_span)
                    union = len(gt_span | pred_span)
                    iou = intersection / union if union > 0 else 0
                
                best_iou = max(best_iou, iou)
            
            ious.append(best_iou)
        
        return np.mean(ious) if ious else 0.0
    
    def evaluate(
        self,
        model,
        dataset,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        Evaluate model on CMCE benchmark.
        
        Args:
            model: DCMT model to evaluate
            dataset: CMCE dataset
            device: Device to run evaluation on
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        model.to(device)
        
        all_chunk_f1 = []
        all_alignment_acc = []
        all_visual_iou = []
        all_text_iou = []
        
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Evaluating"):
                # Get model predictions
                image = sample['image'].unsqueeze(0).to(device)
                input_ids = sample['input_ids'].unsqueeze(0).to(device)
                
                output = model(image, input_ids, return_chunks=True, return_alignment=True)
                
                # Extract predictions
                visual_boundaries = output.chunks[0, 0].cpu().numpy()
                text_boundaries = output.chunks[0, 1].cpu().numpy()
                alignment = output.alignment[0].cpu().numpy()
                
                # Compute metrics
                gt = sample['annotations']
                
                # Chunk F1
                chunk_metrics = self.compute_chunk_f1(
                    text_boundaries, 
                    np.array([1 if i in [c['start'] for c in gt['text_chunks']] else 0 
                             for i in range(len(text_boundaries))])
                )
                all_chunk_f1.append(chunk_metrics['f1'])
                
                # Alignment accuracy
                align_acc = self.compute_alignment_accuracy(alignment, gt['alignments'])
                all_alignment_acc.append(align_acc)
        
        # Aggregate results
        results = {
            'chunk_f1': np.mean(all_chunk_f1),
            'alignment_acc': np.mean(all_alignment_acc),
            'cmce_score': 0.4 * np.mean(all_chunk_f1) + 0.6 * np.mean(all_alignment_acc)
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on CMCE benchmark')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='CMCE', help='Dataset to evaluate on')
    parser.add_argument('--data_path', type=str, default='data/CMCE/test.json', help='Path to test data')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Load model
    from src.models.dcmt import DCMTModel
    model = DCMTModel.from_pretrained(args.model_path)
    
    # Load dataset
    from src.utils.data_loader import CMCEDataset
    dataset = CMCEDataset(args.data_path)
    
    # Evaluate
    evaluator = CMCEEvaluator()
    results = evaluator.evaluate(model, dataset, args.device)
    
    # Print results
    print("\n" + "="*50)
    print("CMCE Evaluation Results")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'cmce_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path / 'cmce_results.json'}")


if __name__ == '__main__':
    main()
