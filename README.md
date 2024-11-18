
# **Argentinian Sign Language Recognition**

## **Description**
This repository contains the implementation of a model for recognizing signs from the Argentinian Sign Language dataset (LSA64). The project aims to facilitate sign language recognition to promote inclusivity and accessibility for individuals with hearing impairments.

---

## **Dataset**
We use the **LSA64: A Dataset of Argentinian Sign Language**, created by Ronchetti et al., 2016. It contains isolated Argentinian sign gestures for 64 words, captured from a single signer under controlled conditions.

**Citation**:
```bibtex
@Article{Ronchetti2016,
  author = "Ronchetti, Franco and Quiroga, Facundo and Estrebou, Cesar and Lanzarini, Laura and Rosete, Alejandro",
  title = "LSA64: A Dataset of Argentinian Sign Language",
  journal = "XX II Congreso Argentino de Ciencias de la Computación (CACIC)",
  year = "2016"
}
```

The original paper is available here: [Ronchetti et al., 2016](https://facundoq.github.io/datasets/lsa64/).

---

## **Training**

### **Step 1: Download the Dataset**
Download the isolated sign dataset (Cut Version) from the official website:
[Download LSA64 Dataset](https://facundoq.github.io/datasets/lsa64/)

- Extract the dataset and ensure the `all_cut` folder is placed in the root directory of this project.

### **Step 2: Train the Model**
Run the training script to start the training process. Make sure to configure the necessary hyperparameters and dataset paths in the script:
```bash
python train.py
```

---

## **How to Make Predictions**
1. Ensure the trained model is available in the `models` directory or provide the path to your saved model.
2. Use the inference script to make predictions on new sign gestures:
```bash
python predict.py --input <path_to_video> --model <path_to_model>
```

Replace `<path_to_video>` with the video of the sign gesture and `<path_to_model>` with the path to your trained model file.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---
