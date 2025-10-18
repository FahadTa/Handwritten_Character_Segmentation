"""
Evaluation script for handwritten character segmentation.
Evaluates trained model on test set and generates comprehensive reports.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl

from config import load_config
from data import create_datamodule
from training import CharacterSegmentationModule
from models import SegmentationMetrics, InstanceSegmentationMetrics
from utils import SegmentationVisualizer, PostProcessor, save_predictions_grid


def evaluate_model(
    checkpoint_path: str,
    config_path: str = 'config/config.yaml',
    output_dir: str = 'outputs/evaluation',
    visualize: bool = True,
    num_visualizations: int = 20
):
    """
    Evaluate trained model on test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        output_dir: Output directory for results
        visualize: Whether to create visualizations
        num_visualizations: Number of samples to visualize
    """
    print("=" * 80)
    print("HANDWRITTEN CHARACTER SEGMENTATION - EVALUATION")
    print("=" * 80)
    
    config = load_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Loading checkpoint: {checkpoint_path}")
    model = CharacterSegmentationModule.load_from_checkpoint(
        checkpoint_path,
        config=config.to_dict()
    )
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✅ Model loaded on {device}")
    
    print("\n📊 Preparing test data...")
    datamodule = create_datamodule(config.to_dict())
    datamodule.prepare_data()
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    print(f"✅ Test set: {len(datamodule.test_dataset)} samples")
    
    num_classes = config.get('model.unet.num_classes')
    metrics = SegmentationMetrics(num_classes=num_classes)
    instance_metrics = InstanceSegmentationMetrics(iou_threshold=0.5)
    
    if visualize:
        visualizer = SegmentationVisualizer(save_dir=str(output_path / "visualizations"))
        post_processor = PostProcessor(min_area=50, max_area=50000)
    
    all_images = []
    all_predictions = []
    all_ground_truths = []
    
    print("\n🔄 Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            predictions = model(images)
            pred_masks = torch.argmax(predictions, dim=1)
            
            metrics.update(predictions, masks)
            instance_metrics.update(pred_masks, masks)
            
            if visualize and batch_idx < num_visualizations:
                for i in range(min(4, images.shape[0])):
                    img = images[i].cpu()
                    gt = masks[i].cpu().numpy()
                    pred = pred_masks[i].cpu().numpy()
                    
                    img_denorm = visualizer._denormalize_image(img)
                    
                    all_images.append(img_denorm)
                    all_predictions.append(pred)
                    all_ground_truths.append(gt)
    
    print("\n📈 Computing final metrics...")
    final_metrics = metrics.compute_all()
    instance_results = instance_metrics.compute()
    
    results = {
        **final_metrics,
        **instance_results,
        'num_test_samples': len(datamodule.test_dataset),
        'checkpoint': checkpoint_path,
        'evaluation_time': datetime.now().isoformat()
    }
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Test Samples: {results['num_test_samples']}")
    print(f"\nSegmentation Metrics:")
    print(f"  IoU:            {results['iou']:.4f}")
    print(f"  Dice:           {results['dice']:.4f}")
    print(f"  Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"  Precision:      {results['precision']:.4f}")
    print(f"  Recall:         {results['recall']:.4f}")
    print(f"  F1 Score:       {results['f1']:.4f}")
    
    print(f"\nInstance-Level Metrics:")
    print(f"  Instance Precision: {results['instance_precision']:.4f}")
    print(f"  Instance Recall:    {results['instance_recall']:.4f}")
    print(f"  Instance F1:        {results['instance_f1']:.4f}")
    print(f"  Instance IoU:       {results['instance_mean_iou']:.4f}")
    print("=" * 80 + "\n")
    
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {results_file}")
    
    if visualize and len(all_images) > 0:
        print("\n🎨 Creating visualizations...")
        
        grid_path = output_path / "predictions_grid.png"
        save_predictions_grid(
            all_images,
            all_predictions,
            all_ground_truths,
            save_path=str(grid_path),
            max_images=min(16, len(all_images))
        )
        
        for idx in range(min(num_visualizations, len(all_images))):
            vis_path = output_path / "visualizations" / f"sample_{idx:04d}.png"
            visualizer.visualize_prediction(
                all_images[idx],
                all_ground_truths[idx],
                all_predictions[idx],
                title=f"Test Sample {idx}",
                save_path=str(vis_path)
            )
        
        print(f"✅ Saved {len(all_images)} visualizations")
        
        metrics_plot_path = output_path / "metrics_comparison.png"
        metrics_to_plot = {
            'IoU': results['iou'],
            'Dice': results['dice'],
            'Pixel Acc': results['pixel_accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1': results['f1']
        }
        visualizer.plot_metrics_comparison(metrics_to_plot, save_path=str(metrics_plot_path))
    
    confusion_matrix = metrics.get_confusion_matrix()
    cm_path = output_path / "confusion_matrix.npy"
    np.save(cm_path, confusion_matrix)
    print(f"💾 Confusion matrix saved to: {cm_path}")
    
    print("\n✅ Evaluation complete!")
    print(f"📁 All results saved to: {output_path}")
    
    return results


def compare_checkpoints(
    checkpoint_paths: list,
    config_path: str = 'config/config.yaml',
    output_dir: str = 'outputs/comparison'
):
    """
    Compare multiple checkpoints on test set.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        config_path: Path to configuration file
        output_dir: Output directory for comparison
    """
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)
    
    all_results = []
    
    for ckpt_path in checkpoint_paths:
        print(f"\n📊 Evaluating: {ckpt_path}")
        results = evaluate_model(
            checkpoint_path=ckpt_path,
            config_path=config_path,
            output_dir=f"{output_dir}/{Path(ckpt_path).stem}",
            visualize=False,
            num_visualizations=0
        )
        results['checkpoint_name'] = Path(ckpt_path).name
        all_results.append(results)
    
    comparison_file = Path(output_dir) / "checkpoint_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Checkpoint':<50} {'IoU':<8} {'Dice':<8} {'F1':<8}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['checkpoint_name']:<50} {result['iou']:<8.4f} {result['dice']:<8.4f} {result['f1']:<8.4f}")
    print("=" * 80 + "\n")
    
    best_checkpoint = max(all_results, key=lambda x: x['iou'])
    print(f"🏆 Best checkpoint: {best_checkpoint['checkpoint_name']}")
    print(f"   IoU: {best_checkpoint['iou']:.4f}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate handwritten character segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --checkpoint outputs/checkpoints/best_model.ckpt
  python evaluate.py --checkpoint outputs/checkpoints/epoch=50.ckpt --no-visualize
  python evaluate.py --compare outputs/checkpoints/*.ckpt
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=False,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/evaluation',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization generation'
    )
    
    parser.add_argument(
        '--num-vis',
        type=int,
        default=20,
        help='Number of samples to visualize'
    )
    
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple checkpoints'
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            compare_checkpoints(
                checkpoint_paths=args.compare,
                config_path=args.config,
                output_dir=args.output
            )
        elif args.checkpoint:
            evaluate_model(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                output_dir=args.output,
                visualize=not args.no_visualize,
                num_visualizations=args.num_vis
            )
        else:
            latest_checkpoint = Path('outputs/checkpoints').glob('*.ckpt')
            latest_checkpoint = sorted(latest_checkpoint, key=lambda x: x.stat().st_mtime, reverse=True)
            
            if latest_checkpoint:
                print(f"Using latest checkpoint: {latest_checkpoint[0]}")
                evaluate_model(
                    checkpoint_path=str(latest_checkpoint[0]),
                    config_path=args.config,
                    output_dir=args.output,
                    visualize=not args.no_visualize,
                    num_visualizations=args.num_vis
                )
            else:
                print("Error: No checkpoint specified and no checkpoints found")
                parser.print_help()
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()