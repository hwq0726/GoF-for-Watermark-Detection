## Overview
This repository contains the code for the paper:

**On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection**

Our implementations of three watermarking algorithms are based on:
- [Gumbel-max](https://simons.berkeley.edu/talks/scott-aaronson-ut-austin-openai-2023-08-17): Implementation based on [link](https://github.com/lx10077/WatermarkFramework).
- [Inverse-Transform](https://arxiv.org/abs/2307.15593): Implementation based on [link](https://github.com/lx10077/WatermarkFramework).
- [SynthID](https://www-nature-com.proxy.library.upenn.edu/articles/s41586-024-08025-4): Implementation based on [MarkLLM](https://github.com/THU-BPM/MarkLLM/tree/main) with significant modifications. 

**Important Notes:**
1. **Implementation Details:** Our implementation uses the continuous version of SynthID where $g$-values follow a uniform distribution on $[0,1]$ for each layer. This differs from both the original SynthID and MarkLLM repositories which use discrete $g$-values following Bernoulli(0.5). While both versions were proposed in the [SynthID paper](https://www-nature-com.proxy.library.upenn.edu/articles/s41586-024-08025-4), we chose the continuous version as its pivotal statistics provides richer information for watermark detection.

2. **Flexibility in Watermark Detection:** For those interested in using our Goodness-of-fit tests for watermark detection, please note that you can generate pivotal statistics using any compatible implementation - *you are not restricted to using our code for this step*. Our detection framework is designed to work with pivotal statistics regardless of their source.


## Usage

### Watermark Detection with GoF Tests
To detect watermarked text using Goodness-of-Fit (GoF) tests:

1. Navigate to the `GoF-tests` directory
2. Prepare your pivotal statistics:
   - Format: numpy array with shape (num_samples, num_tokens)
   - Can be generated using any compatible watermarking method
   - Example data provided in `example_data_gumbel.pkl` (Gumbel-max watermark statistics)
3. Follow the step-by-step guide in `demo.ipynb` to run the detection tests.

### Reproducing Experimental Results
To reproduce the results from our experiments:
- For Gumbel-max and Inverse-Transform watermarks: See `Gumbel&Transform` directory
- For SynthID watermark: See `SynthID` directory

## Repository Structure
### `GoF-tests/`
- **`demo.ipynb`**: A step-by-step guide on how to use the GoF tests to detect watermarked text.
- **`example_data_gumbel.pkl`**: The example data for the pivotal statistics of Gumbel-max watermark.
- **`detect_utils.py`**: Contains the code for the baseline and Goodness-of-fit tests used in the paper.

### `Gumbel&Transform/`
- **`attack.py`**: Tests robustness of watermarked text against various attacks:
  - Word deletion
  - Synonym substitution
  - Dipper paraphrase (didn't use it in the paper)
- **`attack_info.py`**: Perform information-rich attacks in the watermarked text and then perform detection.
- **`detect_human.py`**: Perform detection on human-written text.
- **`generating_samples.py`**: Generates watermarked text using the Gumbel-max or the Inverse-Transform algorithm.
- **`detect_utils.py`**: Contains the code for the baseline and Goodness-of-fit tests used in the paper.
- **`get_score.py`**: Performs watermark detection using different methods.

### `SynthID/`
Contains the implementation of SynthID.
  #### `MarkLLM/`
  - **`synthid_generate.py`**: Generates watermarked text using the SynthID algorithm
  - **`attack.py`**: Tests robustness of watermarked text against various attacks:
    - Word deletion
    - Synonym substitution
    - Dipper paraphrase

  #### `results/`
  All experimental results of SynthID are stored in this directory:
  - **`read_results.py`**: Extracts pivotal values from watermarked texts
  - **`detect_utils.py`**: Contains the code for the baseline and Goodness-of-fit tests used in the paper.
  - **`get_score.py`**: Performs watermark detection using different methods. Also we include the information attack in this code.
