Official PyTorch implementation of GCM-CLIP, as presented in the paper "ForVA and GCM-CLIP: A Million-Scale Multimodal Dataset and Representation Learning Framework for Virtual Autopsy".

This repository is built upon the robust  framework and customized for high-precision, domain-specific multimodal contrastive learning in forensic pathology.

📖 Introduction
Intelligent virtual autopsy faces a profound semantic misalignment driven by scarce multimodal data and insufficient fine-grained cognitive mapping, leaving generic vision-language foundation models vulnerable to complex post-mortem noise (e.g., autolysis, putrefaction) and catastrophic "shortcut learning."

To bridge this misalignment, we propose GCM-CLIP, a semantics-enhanced contrastive learning framework. Powered by a novel Generalized Category Mining (GCM) mechanism—which functions as a high-precision "semantic filter" through Dynamic Semantic Decoupling (DSD) and Adaptive Balanced Clustering (ABC)—GCM-CLIP adaptively extracts implicit fine-grained pathological attributes from unstructured hierarchical reports, effectively isolating redundant anatomical background noise.

✨ Key Features
ForVA Dataset: The first professional-grade, million-scale virtual autopsy dataset containing 1,257,349 forensically validated image-text pairs, covering 9 core cause-of-death categories, 40 lesion typologies, and 8 anatomical regions.

GCM Mechanism: Incorporates explicit semantic anchors and mines implicit fine-grained attributes to construct high-purity supervision signals.

Exceptional Disentanglement: Drastically reduces intra- and inter-class pathological feature overlap (from 0.830/0.709 to 0.566/0.452).

State-of-the-Art Performance: Achieves a 25% relative performance gain in zero-shot classification, 6–8% improvement in cross-modal retrieval, and demonstrates formidable out-of-distribution (OOD) generalization on external clinical datasets (ROCOv2, MIMIC-CXR, PMC-OA).

Real-World Forensic Utility: In blinded trials, GCM-CLIP empowers junior practitioners to leapfrog the diagnostic precision of unassisted senior experts (0.703 vs. 0.433 baseline).

