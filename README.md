<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>🔍 Interpretable Deep Learning with Kolmogorov-Arnold Networks (KANs)</h1>

<h2>🚀 About This Repository</h2>
<p>
This repository explores <b>Kolmogorov-Arnold Networks (KANs)</b> in deep learning, focusing on their <b>interpretability and effectiveness</b> 
compared to traditional architectures. The project includes multiple experiments evaluating KANs in both <b>fully connected and convolutional settings</b>, 
with visualization techniques like <b>Grad-CAM</b> and <b>PCA</b> to better understand their decision-making process.
</p>

<p>
Key areas of investigation:
</p>

<ul>
    <li>🔹 <b>Neural Network Design</b>: Comparing KAN-based architectures with standard deep learning models.</li>
    <li>🔹 <b>Interpretability</b>: Using visualization techniques to analyze learned representations.</li>
    <li>🔹 <b>Performance</b>: Benchmarking KANs on classification tasks (MNIST & EMNIST).</li>
    <li>🔹 <b>Computational Trade-offs</b>: Assessing expressivity vs. efficiency.</li>
</ul>

<hr>

<h2>📌 Experiments</h2>

<h3>🔹 1. Fully Connected KANs vs. MLPs</h3>
<p>
<b>Goal:</b> Compare standard MLPs (LeNet-300) with KAN-based models (KaNet-300) on MNIST to analyze performance and generalization.
</p>
<ul>
    <li>✅ KANs match or exceed standard networks while using fewer layers.</li>
    <li>✅ Splines help reduce overfitting and improve feature representation.</li>
</ul>

<hr>

<h3>🔹 2. Convolutional KANs vs. CNNs</h3>
<p>
<b>Goal:</b> Evaluate convolutional KANs (KaNet-5) against standard CNNs (LeNet-5) on EMNIST using <b>Grad-CAM</b> interpretability analysis.
</p>
<ul>
    <li>✅ KAN filters dynamically adapt to complex patterns.</li>
    <li>✅ Grad-CAM shows KANs have broader, more flexible attention maps.</li>
</ul>

<p><b>Grad-CAM Visualization:</b></p>
<img src="https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/DemoGradCAMFeatureMAPExample.gif" width="600" alt="GradCAM Visualization">

<hr>

<h3>🔹 3. PCA-Based Feature Representation</h3>
<p>
<b>Goal:</b> Use PCA to compare how standard networks and KANs encode learned feature distributions.
</p>

<p><b>PCA Visualization:</b></p>
<img src="https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/PCADemo.png" width="600" alt="PCA Feature Representation">

<hr>

<h3>🔹 4. Spline Evolution in KANs</h3>
<p>
<b>Goal:</b> Visualize how spline-based transformations evolve during training.</p>
<ul>
    <li>✅ Splines adjust dynamically, offering localized feature adaptation.</li>
    <li>✅ Unlike fixed activations, they provide smoother function approximation.</li>
</ul>

<p><b>Visualization of Spline Evolution:</b></p>
<img src="https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/DemoSplineEvolution.gif" width="600" alt="Spline Evolution">

<hr>

<h2>📈 Key Takeaways</h2>
<ul>
    <li>⚡ <b>Expressivity</b>: KANs model functions with more flexibility.</li>
    <li>👁 <b>Interpretability</b>: PCA and Grad-CAM provide deeper insights.</li>
    <li>🚀 <b>Scalability</b>: Balancing efficiency and computational cost.</li>
</ul>

<hr>

<h2>📜 Future Work</h2>
<ul>
    <li>📊 Apply KANs to CIFAR-10 & ImageNet.</li>
    <li>⚙ Optimize efficiency for real-time applications.</li>
    <li>🤖 Combine KANs with attention mechanisms.</li>
</ul>

<hr>

</body>
</html>
