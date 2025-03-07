# 🔍 Interpretable Deep Learning with Kolmogorov-Arnold Networks (KANs)

## 🚀 About This Repository
This repository explores **Kolmogorov-Arnold Networks (KANs)** in deep learning, focusing on their **interpretability and effectiveness** compared to traditional architectures. The project includes multiple experiments evaluating KANs in both **fully connected and convolutional settings**, with visualization techniques like **Grad-CAM** and **PCA** to better understand their decision-making process.

### **Key areas of investigation**
- 🔹 **Neural Network Design**: Comparing KAN-based architectures with standard deep learning models.
- 🔹 **Interpretability**: Using visualization techniques to analyze learned representations.
- 🔹 **Performance**: Benchmarking KANs on classification tasks (MNIST & EMNIST).
- 🔹 **Computational Trade-offs**: Assessing expressivity vs. efficiency.

---

## 📌 Experiments

### 🔹 1. Convolutional KANs vs. CNNs
**Goal:** Evaluate convolutional KANs (KaNet-5) against standard CNNs (LeNet-5) on EMNIST using **Grad-CAM** interpretability analysis.

✅ KAN filters dynamically adapt to complex patterns.  
✅ Grad-CAM shows KANs have broader, more flexible attention maps.  

#### **Grad-CAM Visualization**
![GradCAM](https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/Images%20%26%20Videos/DemoGradCAMFeatureMAPExample.gif)

![Click here to view the interactive Grad-CAM visualizer!](http://leonardomagliolo.altervista.org/visualizer/Visualizer.html)
---

### 🔹 2. Spline Evolution in KANs
**Goal:** Visualize how spline-based transformations evolve during training.

✅ Splines adjust dynamically, offering localized feature adaptation.  
✅ Unlike fixed activations, they provide smoother function approximation.  

#### **Visualization of Spline Evolution**
![Spline Evolution](https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/Images%20%26%20Videos/DemoSplineEvolution.gif)

---

### 🔹 3. Fully Connected KANs vs. MLPs
**Goal:** Compare standard MLPs (LeNet-300) with KAN-based models (KaNet-300) on MNIST to analyze performance and generalization.

✅ KANs match or exceed standard networks while using fewer layers.  
✅ Splines help reduce overfitting and improve feature representation.  

#### **MLP vs. KAN Representation**
![MLP vs KAN](https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/Images%20%26%20Videos/MLP_vs_KAN.png)

---

### 🔹 4. PCA-Based Feature Representation
**Goal:** Use PCA to compare how standard networks and KANs encode learned feature distributions.

#### **PCA Visualization**
![PCA](https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/Images%20%26%20Videos/PCADemo.png)

---

## 📈 Key Takeaways
- ⚡ **Expressivity**: KANs model functions with more flexibility.
- 👁 **Interpretability**: PCA and Grad-CAM provide deeper insights.
- 🚀 **Scalability**: Balancing efficiency and computational cost.

---

## 📜 Future Work
- 📊 Apply KANs to CIFAR-10 & ImageNet.
- ⚙ Optimize efficiency for real-time applications.
- 🤖 Combine KANs with attention mechanisms.

