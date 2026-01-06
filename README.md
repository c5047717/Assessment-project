#StoryReasoning Image Generation with Autoencoders

A Multimodal Deep Learning Approach

Project Overview

This project presents a multimodal deep learning framework for visual story reasoning and next-scene prediction using both images and textual captions. The model is trained on the StoryReasoning dataset and learns to predict the next image frame and its corresponding caption based on prior story context.

The approach integrates a convolutional autoencoder for visual representation learning with a sequence-based multimodal model to capture temporal and narrative consistency. The system demonstrates strong performance in image reconstruction quality and caption generation accuracy.

Key Features

The project includes a convolutional autoencoder for compact image representation, GRU-based sequence modelling for narrative flow, multimodal fusion of visual and textual features, dual decoding for image and caption generation, and complete experimental evaluation.

Project Structure
data/
  images/
  captions/

models/
  autoencoder.py
  multimodal_model.py

scripts/
  pretrain_ae.py
  train.py
  evaluate.py

checkpoints/
  ae_final.pt
  epoch_9.pt

results/
  loss_curves/
  generated_images/

requirements.txt
README.md

Dataset

The StoryReasoning dataset is used in this project. Each sample consists of a sequence of images paired with textual captions describing the scene.

For training, four sequential frames are used as input context and the fifth frame with its caption is used as the prediction target. Images are resized to 64 by 64 pixels and normalised to the range [0, 1].

Model Architecture
Visual Autoencoder

The visual component is a convolutional autoencoder composed of stacked convolutional layers with batch normalisation and ReLU activations. The encoder compresses each image into a 384-dimensional latent representation. The decoder reconstructs the image using transposed convolution layers. The autoencoder is trained using L1 reconstruction loss.

Text Encoder

Captions are encoded using an embedding layer followed by a GRU network to obtain a compact textual representation.

Multimodal Fusion

Image and text embeddings are concatenated and passed through a fully connected fusion layer. A cross-modal attention mechanism is applied to strengthen alignment between visual and linguistic features.

Sequence Model

A GRU-based sequence model captures temporal dependencies across story frames and learns narrative progression.

Dual Decoder

The model includes an image decoder that reconstructs the next frame from latent representations and a caption decoder that generates the next caption using a GRU and linear output layer.

Training Strategy

Training is performed in two stages.

In the first stage, the convolutional autoencoder is pretrained independently using L1 reconstruction loss to learn meaningful visual representations.

In the second stage, the multimodal sequence model is trained end to end using a combination of caption cross-entropy loss, image L1 reconstruction loss, and latent mean squared error loss. The Adam optimiser is used throughout training.

Results
Quantitative Results

The pretrained autoencoder achieves a validation L1 reconstruction loss in the range of 0.06 to 0.08. The multimodal sequence model shows consistent reduction in validation loss across epochs. Caption generation achieves a masked token accuracy of 99.5 percent.

Qualitative Results

Generated images preserve global structure and colour consistency with the story context. Fine details are slightly blurred due to the low image resolution. Generated captions are contextually accurate and semantically coherent.

Evaluation

Evaluation is conducted on unseen story sequences. Performance is measured using L1 reconstruction loss for images and token-level accuracy for captions, excluding padding tokens. Sample predictions and comparisons with ground truth are included in the results directory.

Requirements

Python version 3.8 or higher is required. The project depends on PyTorch, Torchvision, NumPy, Matplotlib, and tqdm. All dependencies can be installed using the requirements.txt file.

How to Run

To pretrain the autoencoder, run pretrain_ae.py.
To train the multimodal sequence model, run train.py.
To evaluate the trained model, run evaluate.py.
