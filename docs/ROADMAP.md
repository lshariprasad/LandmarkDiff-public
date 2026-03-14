# LandmarkDiff Roadmap

This document tracks planned milestones for LandmarkDiff. Timelines are estimates and may shift based on experimental results, reviewer feedback, and community input.

For discussion and feature requests, see [GitHub Discussions](https://github.com/dreamlessx/LandmarkDiff-public/discussions).

---

## v0.2.x (Current)

**Status:** Released

Core pipeline with TPS and ControlNet inference, six procedure presets, clinical edge-case handling, and Hugging Face Space demo.

- [x] MediaPipe 478-point face mesh extraction
- [x] Gaussian RBF deformation engine with procedure-specific presets
- [x] ControlNet-conditioned Stable Diffusion 1.5 generation
- [x] Neural post-processing (CodeFormer, Real-ESRGAN, ArcFace verification)
- [x] 6 procedures: rhinoplasty, blepharoplasty, rhytidectomy, orthognathic, brow lift, mentoplasty
- [x] 4 inference modes: TPS (CPU), img2img, ControlNet, ControlNet+IP-Adapter
- [x] Clinical flags: vitiligo, Bell's palsy, keloid-prone skin, Ehlers-Danlos
- [x] Fitzpatrick-stratified evaluation protocol
- [x] Bilateral symmetry analysis with per-region scoring
- [x] Gradio demo with multi-angle capture and symmetry comparison
- [x] Docker and Apptainer container support
- [x] Hugging Face Space live demo (CPU)

---

## v0.3.0 -- Data-Driven Training

**Target:** Q3 2026

The focus of this release is moving from hand-tuned displacement vectors to displacements learned from real surgical outcome data, and completing the ControlNet fine-tuning pipeline.

### Data-driven displacement model
- [ ] Fit per-procedure displacement distributions from clinical before/after pairs
- [ ] Anatomically constrained sampling (region-specific variance, bilateral coherence)
- [ ] Validate against expert surgeon rankings on a held-out test set

### Training pipeline
- [ ] ControlNet fine-tuning on 50K+ synthetic TPS pairs (Phase A)
- [ ] Combined loss training on clinical pairs (Phase B): diffusion + landmark L2 + identity (ArcFace) + perceptual (LPIPS)
- [ ] Curriculum training: start with large deformations, anneal toward subtle corrections
- [ ] Multi-GPU DDP training with preemption checkpointing for HPC clusters

### Additional procedures
- [ ] Otoplasty (ear pinning) preset
- [ ] Genioplasty (sliding genioplasty) preset
- [ ] Community-contributed preset validation framework

### Evaluation
- [ ] Benchmark on the Rathgeb et al. plastic surgery database (CVPRW 2020)
- [ ] Populate quantitative results tables (FID, LPIPS, SSIM, NME, identity sim)
- [ ] Ablation studies: loss components, conditioning strategies, displacement models

### Publication
- [ ] MICCAI 2026 workshop paper submission (July 2026)
- [ ] arXiv preprint with supplementary materials

---

## v0.4.0 -- Multi-View and Video

**Target:** Q4 2026

Extend predictions beyond single frontal photos to multiple viewpoints and temporal sequences.

### Multi-view support
- [ ] Consistent predictions across frontal, three-quarter, and profile views
- [ ] Multi-view consistency loss (landmark reprojection across views)
- [ ] FLAME 3D morphable model integration for depth-aware deformation

### Video prediction
- [ ] Temporal coherence for video input (face tracking + per-frame prediction)
- [ ] Smooth interpolation between intensity levels for animated previews
- [ ] GIF/MP4 export from the Gradio demo

### Backbone upgrade
- [ ] FLUX.1-dev or SDXL backbone for higher quality generation at 1024x1024
- [ ] IP-Adapter FaceID v2 for stronger identity preservation
- [ ] LoRA fine-tuning support for domain-specific adaptation

### Clinical tools
- [ ] Measurement overlay: display predicted changes in millimeters
- [ ] Before/after report generation (PDF export for patient consultations)
- [ ] Batch processing API for clinical workflow integration

---

## v1.0.0 -- Clinical Validation

**Target:** 2027

Production-ready release with clinical validation data and regulatory groundwork.

### Clinical validation
- [ ] IRB-approved prospective study: compare predictions to actual surgical outcomes
- [ ] Inter-rater agreement study with board-certified plastic surgeons
- [ ] Statistical validation across Fitzpatrick types I-VI (equity audit)
- [ ] Calibration analysis: does predicted intensity correlate with actual surgical magnitude?

### Regulatory pathway
- [ ] FDA 510(k) pathway exploration for clinical decision support classification
- [ ] HIPAA-compliant deployment architecture (on-premise, no patient data leaves the facility)
- [ ] Documentation for SaMD (Software as a Medical Device) qualification

### Physics-informed modeling
- [ ] Finite element method (FEM) soft tissue simulation for physically plausible deformations
- [ ] Patient-specific tissue parameters estimated from skin elasticity measurements
- [ ] Scar formation prediction for incision planning

### Deployment
- [ ] Cloud deployment with Triton Inference Server
- [ ] React Native mobile app for standardized clinical photo capture
- [ ] DICOM integration for radiology workflows
- [ ] On-premise Docker stack for hospital IT environments

---

## How to Contribute

We welcome contributions at every level. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

- **New procedure presets** -- define custom displacement vectors for procedures not yet supported
- **Clinical validation data** -- if you have access to before/after surgical datasets, we would like to hear from you
- **Evaluation benchmarks** -- run LandmarkDiff on your own data and share results
- **Feature requests** -- open an issue or start a [Discussion](https://github.com/dreamlessx/LandmarkDiff-public/discussions)

Significant contributions earn co-authorship on the MICCAI 2026 paper. See the [Contributors table](../README.md#contributors) for recognition tiers.
