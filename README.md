<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interpretable Deep Learning with Kolmogorov-Arnold Networks (KANs)</title>
</head>
<body>

<h1>🔍 Interpretable Deep Learning with Kolmogorov-Arnold Networks (KANs)</h1>
<h2>Investigating KANs for Expressivity, Efficiency, and Explainability</h2>

<h2>📖 Project Overview</h2>
<p>
This repository presents research on Kolmogorov-Arnold Networks (KANs), a novel class of deep learning architectures inspired by the 
Kolmogorov-Arnold representation theorem. Unlike traditional neural networks that rely on scalar weight parameters, KANs replace them with 
adaptive spline functions, offering a more flexible and interpretable approach to function approximation and feature extraction.
</p>

<p>
This study compares KAN-based architectures with standard deep learning models to evaluate:
</p>

<ul>
    <li><b>Neural Network Design:</b> Exploring fully connected KAN models and convolutional KAN architectures (ConvKAN).</li>
    <li><b>Interpretability & Explainability:</b> Applying visualization techniques such as Grad-CAM and PCA embeddings.</li>
    <li><b>Experimental Evaluation:</b> Benchmarking KANs against MLPs and CNNs on datasets such as MNIST & EMNIST.</li>
    <li><b>Computational Trade-offs:</b> Analyzing the scalability, training efficiency, and computational complexity of KANs.</li>
</ul>

<p>
<b>Why KANs?</b><br>
Traditional deep learning models struggle with interpretability and scalability. KANs aim to bridge this gap by leveraging adaptive, 
learnable activation functions on edges instead of fixed ones on nodes. This approach offers:
</p>

<ul>
    <li>Higher Expressivity: KANs can learn more complex transformations with fewer parameters.</li>
    <li>Improved Interpretability: Splines provide a visualizable, structured way to model transformations.</li>
    <li>Better Generalization: The adaptability of splines helps in reducing overfitting.</li>
</ul>

<p>
Preliminary results suggest that KANs can match or surpass standard architectures in function approximation, while offering 
more interpretability through their flexible structure. This research lays the groundwork for interpretable AI, function-based 
neural networks, and future machine learning advancements.
</p>

<hr>

<h2>📌 Experiments & Research Areas</h2>

<h3>🔹 1. Fully Connected KANs vs. Standard MLPs (LeNet-300 vs. KaNet-300 on MNIST)</h3>
<p>
<b>Goal:</b> Investigate whether KAN-based fully connected networks can match or outperform traditional MLPs, particularly when reducing hidden layers.
</p>

<ul>
    <li><b>Experimental Setup:</b></li>
    <ul>
        <li>Standard LeNet-300 architecture vs. KaNet-300 (KAN version with splines replacing traditional weights).</li>
        <li>Training on the MNIST dataset (handwritten digits).</li>
        <li>Evaluation on accuracy, loss trends, and convergence rate.</li>
    </ul>
    <li><b>Findings:</b></li>
    <ul>
        <li>KAN models achieve comparable or better accuracy with reduced hidden layers.</li>
        <li>Splines provide a smoother learning process, reducing overfitting.</li>
        <li>KANs maintain expressivity even with fewer parameters, making them a compact yet powerful alternative.</li>
    </ul>
</ul>

<hr>

<h3>🔹 2. Convolutional KANs vs. Standard CNNs (LeNet-5 vs. KaNet-5 on EMNIST)</h3>
<p>
<b>Goal:</b> Compare the interpretability and performance of KAN-based convolutional networks (ConvKAN) against traditional CNNs (LeNet-5) on a more complex classification task.
</p>

<ul>
    <li><b>Experimental Setup:</b></li>
    <ul>
        <li>Standard LeNet-5 (CNN with ReLU activations) vs. KaNet-5 (CNN using adaptive spline-based filters).</li>
        <li>Training on EMNIST dataset (letters & digits – 62 classes).</li>
        <li>Grad-CAM used to generate heatmaps for interpretability.</li>
    </ul>
    <li><b>Findings:</b></li>
    <ul>
        <li>ConvKAN filters are more flexible, adapting to feature complexity better than fixed-weight convolutions.</li>
        <li>Grad-CAM visualization shows that KaNet-5 has a broader attention focus, while LeNet-5 focuses on local features.</li>
        <li>Potential trade-off: ConvKANs offer richer feature extraction but require higher computational complexity.</li>
    </ul>
</ul>

<p><b>Grad-CAM Visualization:</b></p>
<img src="https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/DemoGradCAMFeatureMAPExample.gif" width="600" alt="GradCAM Visualization">

<hr>

<h3>🔹 3. PCA-Based Feature Representation Analysis</h3>
<p>
<b>Goal:</b> Use Principal Component Analysis (PCA) to analyze how KAN-based networks represent learned feature distributions compared to standard MLPs/CNNs.
</p>

<p><b>PCA Visualization:</b></p>
<img src="https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/PCADemo.png" width="600" alt="PCA Feature Representation">

<hr>

<h3>🔹 4. Spline Adaptation & Evolution in KANs</h3>
<p>
<b>Goal:</b> Visualize how learnable spline functions evolve over training, understanding how KAN transformations differ from standard deep learning weight updates.
</p>

<p><b>Findings:</b></p>
<ul>
    <li>Spline functions dynamically adapt to fit complex relationships in the data.</li>
    <li>Unlike standard activations (e.g., ReLU), splines allow localized control, making KANs more interpretable.</li>
</ul>

<p><b>Visualization of Spline Evolution:</b></p>
<img src="https://raw.githubusercontent.com/MaglioloLeonardo/InterpretableKAN/main/DemoSplineEvolution.gif" width="600" alt="Spline Evolution">

<hr>

<h2>📈 Key Takeaways</h2>
<ul>
    <li><b>Expressivity:</b> KANs enable more powerful function approximations by using adaptive spline transformations instead of fixed activations.</li>
    <li><b>Interpretability:</b> Heatmap and PCA analyses reveal that KANs distribute attention differently, providing new insights into learned representations.</li>
    <li><b>Computational Trade-offs:</b> While KANs can reduce network depth, they increase per-layer complexity, requiring optimization for real-world deployment.</li>
</ul>

<hr>

<h2>📜 Future Directions</h2>
<ul>
    <li>Applying KAN-based architectures to larger-scale datasets (CIFAR-10, ImageNet).</li>
    <li>Optimizing KAN computational efficiency for real-time applications.</li>
    <li>Integrating KANs with attention mechanisms for hybrid deep learning models.</li>
</ul>

<hr>

<h2>🔗 References</h2>
<ul>
    <li><a href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem">Kolmogorov-Arnold Representation Theorem</a></li>
    <li><a href="https://arxiv.org/abs/2404.19756">Kolmogorov-Arnold Networks (arXiv)</a></li>
</ul>

</body>
</html>
