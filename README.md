# Sangoi Loss Modifier Function

This repository contains the implementation of the **Sangoi Loss Modifier**, a custom loss modifier function designed for use with the **OneTrainer**. The function adjusts the training loss based on the prediction accuracy and the difficulty of the prediction task, encouraging the neural network to focus on making accurate predictions, especially when the task is challenging.

## Table of Contents

- [Overview](#overview)
- [Function Implementation](#function-implementation)
- [Explanation](#explanation)
  - [Clamping the SNR Values](#1-clamping-the-snr-values)
  - [Calculating the Mean Absolute Percentage Error (MAPE)](#2-calculating-the-mean-absolute-percentage-error-mape)
  - [Computing the Weights](#3-computing-the-weights)
  - [Returning the Loss Modifier](#4-returning-the-loss-modifier)
- [Usage](#usage)
- [Notes](#notes)
- [License](#license)

## Overview

The Sangoi Loss Modifier function computes a weight that modifies the loss during training. It considers two main factors:

1. **Prediction Accuracy (MAPE)**: Measures how close the predictions are to the target values.
2. **Prediction Difficulty (SNR)**: Assesses how challenging the prediction task is based on the Signal-to-Noise Ratio.

By combining these factors, the function generates a reward for the neural network, scaling it according to both the accuracy of the predictions and the difficulty level of the task. This encourages the model to perform better, especially on more challenging predictions.

## Function Implementation

```python
def _sangoi_loss_modifier(self, timesteps: Tensor, predicted: Tensor, target: Tensor, gamma: float, device: torch.device) -> Tensor:
    """
    Computes a loss modifier based on the Mean Absolute Percentage Error (MAPE) and the Signal-to-Noise Ratio (SNR).
    This modifier adjusts the loss according to the prediction accuracy and the difficulty of the prediction task.

    Args:
        timesteps (Tensor): The current training step's timesteps.
        predicted (Tensor): Predicted values from the neural network.
        target (Tensor): Ground truth target values.
        gamma (float): A scaling factor (unused in this function).
        device (torch.device): The device on which tensors are allocated.

    Returns:
        Tensor: A tensor of weights per example to modify the loss.
    """

    # Define minimum and maximum SNR values to clamp extreme values
    min_snr = 1e-4
    max_snr = 100

    # Obtain the SNR for each timestep
    snr = self._snr(timesteps, device)
    # Clamp the SNR values to the defined range to avoid extreme values
    snr = torch.clamp(snr, min=min_snr, max=max_snr)

    # Define a small epsilon to prevent division by zero
    epsilon = 1e-8
    # Compute the Mean Absolute Percentage Error (MAPE)
    mape = torch.abs((target - predicted) / (target + epsilon))
    # Normalize MAPE values between 0 and 1
    mape = torch.clamp(mape, min=0, max=1)
    # Calculate the average MAPE per example across spatial dimensions
    mape = mape.mean(dim=[1, 2, 3])

    # Compute the SNR weight using the natural logarithm (adding 1 to avoid log(0))
    snr_weight = torch.log(snr + 1)
    # Invert MAPE to represent accuracy instead of error
    mape_reward = 1 - mape
    # Calculate the combined weight using the negative exponential of the product of MAPE reward and SNR weight
    combined_weight = torch.exp(-mape_reward * snr_weight)

    # Return the tensor of weights per example to modify the loss
    return combined_weight
```

## Explanation

### 1. Clamping the SNR Values

```python
min_snr = 1e-4
max_snr = 100
snr = self._snr(timesteps, device)
snr = torch.clamp(snr, min=min_snr, max=max_snr)
```

- **Purpose**: Prevents extreme SNR values that could destabilize training.
- **Explanation**: The Signal-to-Noise Ratio (SNR) is obtained for each timestep and clamped between `1e-4` and `100` to avoid values that are too small or too large.

### 2. Calculating the Mean Absolute Percentage Error (MAPE)

```python
epsilon = 1e-8
mape = torch.abs((target - predicted) / (target + epsilon))
mape = torch.clamp(mape, min=0, max=1)
mape = mape.mean(dim=[1, 2, 3])
```

- **Purpose**: Measures the prediction error as a percentage.
- **Explanation**:
  - **Epsilon Addition**: A small value `epsilon` is added to the denominator to prevent division by zero.
  - **Clamping MAPE**: The MAPE values are clamped between `0` and `1` to normalize the error.
  - **Averaging**: The mean MAPE is calculated per example across spatial dimensions (e.g., height, width, channels).

### 3. Computing the Weights

```python
snr_weight = torch.log(snr + 1)
mape_reward = 1 - mape
combined_weight = torch.exp(-mape_reward * snr_weight)
```

- **Purpose**: Creates a scaling factor that rewards accurate predictions more when the task is difficult.
- **Explanation**:
  - **SNR Weight**: The natural logarithm of the SNR (plus one) scales the impact of prediction difficulty.
  - **MAPE Reward**: Inverting the MAPE (`1 - mape`) transforms the error into an accuracy measure.
  - **Combined Weight**: The negative exponential of the product of MAPE reward and SNR weight generates the final weight, emphasizing high accuracy in difficult tasks.

### 4. Returning the Loss Modifier

```python
return combined_weight
```

- **Explanation**: The function returns a tensor of weights per example, which can be used to modify the loss during training.

## Usage

To use the Sangoi Loss Modifier in your training process:

1. **Integrate the Function**: Include the `_sangoi_loss_modifier` function in your training class or script.

2. **Compute the Loss Modifier**: During training, calculate the loss modifier using the current `timesteps`, `predicted`, and `target` tensors.

   ```python
   loss_modifier = self._sangoi_loss_modifier(timesteps, predicted, target, gamma, device)
   ```

3. **Apply the Modifier to the Loss**: Multiply your base loss by the `loss_modifier` to adjust the loss according to the prediction accuracy and difficulty.

   ```python
   adjusted_loss = base_loss * loss_modifier
   ```

4. **Proceed with Backpropagation**: Use the `adjusted_loss` for backpropagation to update the model weights.

## Notes

- **SNR Interpretation**: In this context, a higher SNR indicates a prediction task with less noise but more difficulty. The function is designed to give higher rewards (loss modifiers) when the prediction is accurate in these challenging scenarios.

- **Use of Logarithm in SNR Weight**: The logarithm scales down large SNR values to prevent them from disproportionately affecting the loss modifier. Adding `1` inside the `log` function ensures that `log(0)` is avoided.

- **Gamma Parameter**: The `gamma` parameter is included in the function signature but is not used within the function. It can be removed if not needed elsewhere.

- **OneTrainer Compatibility**: The function was developed for use with the `OneTrainer` framework and can replace the `min_snr_gamma` function for ease of integration without significant modifications.

## Acknowledgments

I would like to extend my heartfelt thanks to **OpenAI** and the various **ChatGPT** models I have utilized over the past year. Their assistance has been invaluable in learning and developing this loss modifier function. Special thanks to the latest and most advanced model, **ChatGPT o1-preview**, for helping to create this README and format the function on **October 1, 2024**.

## License
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg