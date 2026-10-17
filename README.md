# Autoregressive Word-Level Text Generation: SimpleRNN vs BiRNN vs LSTM vs GRU

**Course:** Foundations of Generative AI (ICT4403) — Assignment 2
**Author:** Deepak H Suvarna · **Reg. No.:** 220953670

## 1. Problem statement and objective
Build an autoregressive language model that predicts the next word given the previous *L* words,
P(w_t | w_{t-L..t-1}), and uses it to generate text one word at a time. Four recurrent backbones
(Simple RNN, Bidirectional RNN, LSTM, GRU) are trained under identical conditions and compared
on loss, perplexity, accuracy, training time and the quality of generated text at several
sampling temperatures.

## 2. Dataset
- **Name:** *The Adventures of Sherlock Holmes* by Arthur Conan Doyle (public domain)
- **Source:** https://www.gutenberg.org/cache/epub/1661/pg1661.txt (Project Gutenberg #1661)
- **Preprocessing:** strip Gutenberg header/footer → normalise quotes → lowercase → regex
  tokenisation (words incl. apostrophes, plus `. , ! ? ; :`) → vocabulary of the 5,000 most
  frequent **training** words (+ `<pad>`=0, `<unk>`=1).
- **Split:** contiguous 80 / 10 / 10 % of the token stream (train / val / test), so no window
  crosses a split boundary and the vocabulary never sees validation/test text.
- **Sequences:** sliding window, length 20, stride 1. Input = 20 consecutive tokens,
  target = the next token.
- Exact counts: see `results/dataset_statistics.json`.

## 3. Models (shared embedding → recurrent stack → linear softmax head)
| Model | Backbone | Notes |
|---|---|---|
| SimpleRNN | `nn.RNN` (tanh), 2 layers | no gating; short memory, vanishing gradients |
| BiRNN | `nn.RNN`, bidirectional, 2 layers | forward + backward final states concatenated; reads the *given* window in both directions |
| LSTM | `nn.LSTM`, 2 layers | input/forget/output gates + cell state |
| GRU | `nn.GRU`, 2 layers | update/reset gates, fewer parameters than LSTM |

**Shared settings:** embedding 128, hidden 256, dropout 0.3, Adam lr 1e-3, batch 256,
grad-clip 1.0, 12 epochs, seed 42, best epoch chosen by validation loss.

## 4. How to run
```bash
git clone <YOUR REPO URL>
cd <YOUR REPO NAME>
pip install -r requirements.txt
python train_and_evaluate.py --out results        # downloads the book, trains, evaluates, generates
```
A GPU is recommended (Colab/Kaggle: open `autoregressive_text_generation.ipynb`).
Use `--quick` for a one-epoch smoke test. Outputs in `results/`: `results.csv`, `history.csv`,
`samples.csv`, `temperature_summary.csv`, `learning_curves.png`, `temperature_metrics.png`,
`results_tables.md`, `config.json`, `dataset_statistics.json`.

## 5. Results
> Paste the tables from `results/results_tables.md` here after running, and add the plots:
> `![curves](results/learning_curves.png)` `![temps](results/temperature_metrics.png)`

<!-- RESULTS_PLACEHOLDER -->

## 6. Model comparison and evaluation
> Write 1–2 paragraphs based on YOUR numbers: which model has the lowest test loss/perplexity,
> highest accuracy, fastest training; signs of overfitting in `learning_curves.png`; and a
> qualitative review of coherence, diversity, contextual consistency and repetition using
> real excerpts from `samples.csv`.

## 7. Sampling-temperature analysis
Same three seed prompts and three sampling seeds were used for every model at T = 0.5, 1.0, 1.3
(40 generated words each, `<pad>`/`<unk>` masked). Metrics: Distinct-1, Distinct-2,
repeated-trigram fraction, longest back-to-back repeat.
Expected pattern to verify against your data: low T → focused but repetitive; high T → diverse
but less coherent.

> Write your observed analysis here.

## 8. Conclusion
> State which model is best for this dataset and justify it with the evidence above.

## Repository layout
```
textgen_data.py            # data loading, tokenising, windowing, diversity metrics
train_and_evaluate.py      # models, training, evaluation, generation, plots
autoregressive_text_generation.ipynb   # Colab/Kaggle runner
requirements.txt
results/                   # generated outputs (commit these after running)
```
