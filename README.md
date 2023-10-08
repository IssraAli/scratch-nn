# scratch-nn
Neural network from scratch using numpy in python. (Sorry, I did not write a neural network in Scratch)

I did this project because I felt like I didn't really know what was going on under the hood when I was first learning about machine learning. Many online resources start you off using PyTorch or Tensorflow, so you don't understand on a mathematical level what is actually going on in a network—personally, having this level of insight has helped me diagnose problems in more complicated projects that do use high-performance ML frameworks. I think everyone who wants a good intro to machine learning should do this as an exercise, and I'm currently working on a tutorial to walk a beginner through this. 

This is a one-hidden-layer fully connected neural network to classify handwritten digits from the MNIST dataset. After 20 epochs of training the test accuracy was around 80%. 

<img src=https://github.com/IssraAli/scratch-nn/blob/0fb363eac72e663481dbe69b18011f0f845a4b6e/loss_accuracy.png>
<img src=https://github.com/IssraAli/scratch-nn/blob/0fb363eac72e663481dbe69b18011f0f845a4b6e/samples.png>

The hardest part about this project was computing the gradients by hand. 
