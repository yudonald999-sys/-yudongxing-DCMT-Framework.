> **Archived.** Runnable code for reviewers and experiments has moved to **https://github.com/yudongxing999/DCMT-Framework**. This repository is kept for historical assets (figures/tables); please use the new repo for install, smoke tests, and continued development.

# Measurement Protocols

This directory contains the measurement protocols and analysis scripts used in our study.

## Eye-Tracking Protocol

### Equipment
- **Device**: Tobii Pro Spectrum
- **Sampling Rate**: 1200 Hz
- **Monitor**: 24-inch, 1920×1080 pixels
- **Viewing Distance**: 65 cm (maintained with chin rest)

### Procedure
1. 9-point calibration with validation (average error < 0.5°)
2. Stimulus presentation: 12 seconds per image-text pair
3. Areas of Interest (AOIs) defined a priori for each object
4. Drift correction every 10 trials

### Metrics Extracted
| Metric | Description | Unit |
|--------|-------------|------|
| Fixation Duration | Time spent on each AOI | ms |
| Transition Frequency | Cross-modal gaze shifts | count |
| First Fixation Latency | Time to first fixation on AOI | ms |
| Scanpath Length | Total gaze path distance | pixels |

### Analysis Scripts
- `eye_tracking.py`: Main analysis pipeline
- `aoi_analysis.py`: Area of Interest metrics
- `scanpath_analysis.py`: Scanpath pattern analysis

---

## Neuroimaging Protocol

### Equipment
- **Scanner**: 3T Siemens Prisma
- **Head Coil**: 64-channel
- **Software**: fMRIPrep 20.2.0

### Acquisition Parameters

#### Functional MRI (EPI)
| Parameter | Value |
|-----------|-------|
| TR | 1000 ms |
| TE | 30 ms |
| Flip Angle | 62° |
| Multiband Factor | 6 |
| Voxel Size | 2 mm isotropic |
| Slices | 72 |

#### Structural MRI (MPRAGE)
| Parameter | Value |
|-----------|-------|
| Voxel Size | 1 mm isotropic |
| FOV | 256 × 256 mm |

### Preprocessing Pipeline
1. Motion correction (mcflirt)
2. Slice timing correction
3. Distortion correction (fieldmap-based)
4. Coregistration to T1w
5. Normalization to MNI152 template
6. Spatial smoothing (6 mm FWHM)

### Regions of Interest
- Fusiform Gyrus (visual processing)
- Lateral Occipital Complex (object recognition)
- Superior Temporal Sulcus (multimodal integration)
- Inferior Frontal Gyrus (language processing)

---

## Chunking Measurement Protocol

### Visual Chunking
1. **Semantic Segmentation**: Objects identified using pretrained segmentation
2. **Human Validation**: Three annotators verify/adjust boundaries
3. **Chunk Metrics**:
   - Number of chunks per image
   - Chunk size (bounding box area)
   - Semantic coherence score

### Textual Chunking
1. **Syntactic Parsing**: Initial parse using constituency parser
2. **Human Adjustment**: Annotators mark semantic phrase boundaries
3. **Chunk Metrics**:
   - Number of chunks per text
   - Chunk length (words/characters)
   - Inter-chunk pause probability

### Cross-Modal Alignment
1. Annotators link corresponding visual-textual chunks
2. Confidence rating (1-5) for each alignment
3. Agreement computed using Fleiss' κ

---

## Statistical Analysis Protocol

### Sample Size Justification
- Power analysis: 80% power, α = 0.05, medium effect size (d = 0.5)
- Required n = 34 per group; we recruited n = 48 for eye-tracking

### Multiple Comparison Correction
- Bonferroni correction for benchmark comparisons (4 tests)
- FDR correction for neuroimaging analyses
- Adjusted α = 0.05 / 4 = 0.0125 for benchmark tests

### Effect Size Reporting
All comparisons include:
- Cohen's d for mean differences
- Pearson's r for correlations
- 95% confidence intervals via bootstrap (10,000 iterations)

### Reproducibility
- Random seed: 42 (all experiments)
- All analysis code available in this repository
- Raw data available upon reasonable request

---

## File Descriptions

| File | Description |
|------|-------------|
| `eye_tracking.py` | Eye-tracking data processing and analysis |
| `chunking_metrics.py` | Chunk boundary detection and measurement |
| `statistical_tests.py` | Statistical analysis functions |
| `neural_analysis.py` | fMRI data analysis scripts |
| `mi_estimation.py` | Mutual information estimation (MINE) |

## Usage Example

```python
from measurements.eye_tracking import EyeTrackingAnalyzer
from measurements.chunking_metrics import ChunkingMetrics
from measurements.statistical_tests import StatisticalAnalysis

# Eye-tracking analysis
analyzer = EyeTrackingAnalyzer(data_path="data/eyetracking/")
fixation_stats = analyzer.compute_fixation_statistics()
transition_matrix = analyzer.compute_transition_matrix()

# Chunking metrics
chunker = ChunkingMetrics()
visual_chunks = chunker.detect_visual_chunks(image)
text_chunks = chunker.detect_text_chunks(text)
alignment = chunker.compute_alignment(visual_chunks, text_chunks)

# Statistical tests
stats = StatisticalAnalysis()
t_stat, p_val, cohens_d = stats.paired_ttest(group1, group2)
```

## Contact

For questions about measurement protocols:
- **Email**: yudongxing@sandau.edu.cn
