bone-fracture-detection/
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
├── Makefile                          # Common commands (train, test, lint)
│
├── configs/                          # Configuration files
│   ├── __init__.py
│   ├── base_config.yaml              # Common settings
│   ├── models/
│   │   ├── cnn_baseline.yaml
│   │   ├── resnet50.yaml
│   │   ├── vgg16.yaml
│   │   ├── efficientnet_b0.yaml
│   │   ├── densenet121.yaml
│   │   ├── vit_b16.yaml
│   │   └── yolov8.yaml
│   └── training/
│       ├── default_train.yaml
│       └── hyperparameter_search.yaml
│
├── data/
│   ├── raw/                          # Original dataset (gitignored)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── processed/                    # Preprocessed data
│   └── README.md                     # Dataset documentation
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch Dataset classes
│   │   ├── dataloader.py             # DataLoader factory
│   │   ├── preprocessing.py          # Resize, normalize, augment
│   │   ├── augmentation.py           # Albumentations / torchvision transforms
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py             # Abstract base class — consistent interface for all models
│   │   ├── cnn_baseline.py           # Simple 4-layer CNN (~1M params) — tiny baseline
│   │   ├── alexnet.py                # AlexNet (~60M params) — classic 2012 ImageNet winner
│   │   ├── vgg.py                    # VGG16 (~138M params) — classic deep CNN
│   │   ├── resnet.py                 # ResNet50 (~25M params) — modern benchmark
│   │   ├── densenet.py               # DenseNet121 (~8M params) — medical imaging favorite
│   │   ├── efficientnet.py           # EfficientNet-B3 (~12M params) — strong on FracAtlas
│   │   ├── mobilenet.py              # MobileNetV3 (~5M params) — lightweight / deployable
│   │   ├── convnext.py               # ConvNeXt-Tiny (~28M params) — advanced CNN (SOTA family)
│   │   ├── vit.py                    # ViT-B/16 (~86M params) — standard Vision Transformer
│   │   ├── swin.py                   # Swin-Tiny (~28M params) — advanced hierarchical ViT
│   │   └── model_factory.py          # Registry: build_model("resnet50", num_classes=2)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main training loop
│   │   ├── loss.py                   # Loss functions (CE, Focal, BCE)
│   │   ├── optimizer.py              # Optimizer & scheduler factory
│   │   ├── callbacks.py              # EarlyStopping, Checkpoint, Logging, ...
│   │   └── cross_validation.py       # K-fold CV
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Accuracy, Precision, Recall, F1, AUC
│   │   ├── evaluator.py              # Runs evaluation on a model
│   │   ├── confusion_matrix.py
│   │   ├── roc_curve.py
│   │   └── comparator.py             # Compares multiple models
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py              # Single-image / batch inference
│   │   └── gradcam.py                # Explainability (Grad-CAM heatmaps)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── path.py                     # Added for finding root path
│       ├── dataset_stats.py            # Added for calculating Mean and Std to normalize pixel values
│       ├── logger.py
│       ├── seed.py                   # Reproducibility
│       ├── config_loader.py
│       ├── visualization.py
│       ├── plots.py
│       └── file_io.py
│
├── scripts/                          # Entry-point scripts
│   ├── prepare_data.py               # Preprocess & build splits
│   ├── model_summary.py
│   ├── train.py                      # Train a single model
│   ├── train_all.py                  # Train all models sequentially
│   ├── evaluate.py                   # Evaluate a trained model
│   ├── compare_models.py             # Generate comparison report
│   ├── hyperparameter_search.py      # Optuna / grid search
│   ├── predict.py                    # Run inference on new images
│   └── export_model.py               # Export to ONNX / TorchScript
│
├── notebooks/                        # Exploration & analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_check.ipynb
│   ├── 03_model_training_demo.ipynb
│   ├── 04_results_comparison.ipynb
│   └── 05_visualization_gradcam.ipynb
│   └── smoke_tests.ipynb
│
├── experiments/                      # Training runs (gitignored)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── config.yaml
│       ├── logs/
│       ├── checkpoints/
│       ├── metrics.json
│       └── plots/
│
├── results/                          # Final comparison outputs
│   ├── comparison_table.csv
│   ├── comparison_table.md
│   ├── figures/
│   │   ├── accuracy_bar.png
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   └── training_curves.png
│   └── reports/
│       └── final_report.pdf
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_augmentation.py
│   ├── test_dataloader.py
│   ├── cnn_baseline.py
│   ├── test_dataset.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_training.py
├── weights/                          # 🆕 Final trained model weights (gitignored, kept locally)
│   ├── README.md                     # Explains purpose + how to obtain weights
│   └── <model_name>/                 # One folder per model
│       ├── best.pth                  # The actual trained weights
│       └── config.yaml               # Config that produced these weights
│
└── docs/
    ├── architecture.md
    ├── dataset_description.md
    ├── experiment_protocol.md
    └── results_summary.md