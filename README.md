# FMNV: A Dataset of Media-Published News Videos for Fake News Detection

This repository contains the dataset **FMNV (Fake Media News Videos)**, as presented in the paper *"FMNV: A Dataset of Media-Published News Videos for Fake News Detection"* (ICIC 2025).

Unlike existing datasets that focus on user-generated content (UGC), FMNV focuses on **professionally produced news videos** from mainstream media outlets. It is designed to benchmark multimodal fake news detection models against high-quality, deceptive content.

## 📥 Download Dataset

 The dataset is hosted on Baidu Netdisk.

- **Link:** [Download FMNV Dataset](https://pan.baidu.com/s/1o-xqR5lyzkEEyU60tkTd9g?pwd=2i94)
- **Password:** `2i94`

---

## 📊 Dataset Overview

Most existing fake news video datasets consist of low-quality, user-generated clips. FMNV addresses the gap in detecting **high-impact fake news** disseminated by media organizations, which often features professional editing and higher perceived credibility.

### Statistics
- **Total Videos:** 2,393
- **Real News:** 893 (Sourced from verified Twitter & YouTube media accounts)
- **Fake News:** 1,500 (Generated via LLM-assisted augmentation)
- **Average Duration:** ~73.8 seconds (Longer than typical short-video datasets)

### Categories of Manipulation
The dataset categorizes fake news videos into four distinct types based on cross-modal inconsistency:

1.  **Contextual Dishonesty (CD):** The video title misrepresents the actual events shown in the video (semantic mismatch).
2.  **Cherry-picked Editing (CE):** Critical video segments are selectively removed to create a biased or false narrative.
3.  **Synthetic Voiceover (SV):** The original audio is replaced with AI-generated speech that contradicts or fabricates the visual context.
4.  **Contrived Absurdity (CA):** Videos that maintain surface-level consistency but present exaggerated, illogical, or "common sense" defying claims.

| Category | Count | Description |
| :--- | :--- | :--- |
| **Real** | 893 | Authentic media-published news |
| **Fake (CD)** | 600 | Title-Video mismatch |
| **Fake (CE)** | 450 | Visual information deletion |
| **Fake (SV)** | 300 | Audio falsification |
| **Fake (CA)** | 150 | Logical absurdity |

---

## 📝 Citation

If you use this dataset in your research, please cite our paper:

```bibtex
@inproceedings{wang2025fmnv,
  title={FMNV: A Dataset of Media-Published News Videos for Fake News Detection},
  author={Wang, Yihao and Qian, Zhong and Li, Peifeng},
  booktitle={International Conference on Intelligent Computing (ICIC)},
  pages={321--332},
  year={2025},
  publisher={Springer},
  doi={10.1007/978-981-96-9794-6_27}
}
