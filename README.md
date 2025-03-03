# Generative Adversarial Network (GAN) for Generating Fake Signatures

### Model Architecture
The code implements a GAN with the following components:
- **Generator**: A fully connected neural network that takes a noise vector as input and generates a synthetic signature image. It consists of multiple layers with batch normalization and ReLU activations, with a final layer using a sigmoid activation function to output an image.
- **Discriminator**: A fully connected neural network that takes an image as input (either real or generated) and outputs a probability indicating whether the image is real or fake. It uses Leaky ReLU activations to improve learning stability.

### Data Preprocessing
- The dataset consists of real signatures stored in a directory.
- Images are resized to 64x64 and normalized using torchvision transforms.
- A PyTorch `DataLoader` is used to load the dataset in batches.

### Training Process
1. **Forward Pass**: The generator creates fake images from random noise, and the discriminator evaluates both real and fake images.
2. **Discriminator Loss**: Compares real and fake images, penalizing incorrect classifications using Binary Cross-Entropy (BCE) loss.
3. **Generator Loss**: Encourages the generator to create more realistic images by maximizing the discriminator's misclassification of fake images.
4. **Backpropagation**: Both networks update their weights using Adam optimization.
5. **Image Visualization**: Periodically, generated images are displayed to monitor training progress.
6. **Model Saving**: The trained generator is saved as `generator_model.pth` for future image generation.

### Signature Generation
- The trained generator is loaded and used to generate synthetic signatures by inputting random noise vectors.
- Generated images are visualized and can be saved for further analysis.

### Code Implementation
- `gan-signature-generation.ipynb`: Contains the entire implementation, including data loading, model training, evaluation, and image generation.
- **Key Functions & Classes**:
  - `Generator`: Defines the neural network for generating images.
  - `Discriminator`: Defines the neural network for distinguishing real and fake images.
  - `get_noise()`: Generates random noise for the generator.
  - `disc_loss()` & `gen_loss()`: Compute losses for training.
  - `train_loop`: Runs the training process.
  - `show_tensor_images()`: Displays generated images.

### Output
- Generated signature images are displayed during training and can be found in the output directory.
- The trained generator can be used for further signature synthesis applications.
