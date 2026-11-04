# Attention made of neurons

Biocompute hackathon, 3 November 2026. Luqmaan.

**Question:** can the neurons do the work of an attention block [1, 2], rather
than the computer around them?

**Short answer:** in this simulator, partly. The neurons can do the matching step
of attention, but noisily, and they cannot learn. Everything below runs on the
simulator in this repo, so none of it is evidence about real CL1 tissue.

Visual walkthrough of one trial: [`attention-visuals/index.html`](attention-visuals/index.html)
(open it in a browser).

## What attention does

1. **Match.** Score a query against every key. The score is the number of
   positions where both are 1, a sum of products.
2. **Weigh.** Turn the scores into weights, so the best match counts most.
3. **Mix.** Output a weighted blend of the values that belong to those keys.

## Plan A: let the neurons learn key → value pairs

The idea: pulse a group of "key" electrodes, then pulse a group of "value"
electrodes 5 ms later, many times. Real neurons strengthen a connection when
one fires just before the other (spike-timing-dependent plasticity [3]). Once
the pairs are stored, pulsing a key alone should make its value fire. This is a
key → value memory, and attention can be read as exactly that kind of memory [4].

**Try 1: turn on the simulator's learning settings.** The simulator has
`plasticity` and `stdp` settings. With them on, 200 key-then-value pairings
changed the connections by 0.0000:

```bash
uv run python experiments/stock_learning_check.py src/organoid_simulator/lif_data_source.py
```

**Why:** the simulator records which neurons fired together, but it only turns
that record into stronger connections inside `apply_reward()`. Nothing can call
that function. The SDK only talks to the simulator through start, stop, pulses
and reads. The simulator's README says it was copied from an older project; our
guess is that experiments there could call `apply_reward()` directly.

**Try 2: wire learning up ourselves, to test the idea.** We added a setting that
applies the timing rule straight away, without a reward. Then every key recalled
its own value: 43.7 times more spikes on the right value than on the others, and
4 of 4 keys correct. Reversing the timing learned nothing. We also had to make
the background activity quieter, or the network locked into constant firing.

We reverted both changes. Changing the model until the experiment works only
tells you about the change. The code is kept in the history (commit `2c8e1b9`)
and would run unchanged on a real CL1, where learning is real biology.

## Plan B: use each neuron as a tiny multiplier (stock simulator)

This treats the neurons as a fixed system that transforms our input, and only
reads the result. That approach is called reservoir computing [5, 6].

- **Layout:** 4 keys × 8 bits = 32 electrodes (channels 8 to 39). In the viewer,
  each key is one column and each bit is one row.
- **One try:** pulse the electrodes where the key bit is 1. 5 ms later, pulse
  the electrodes where the query bit is 1, on all four keys.
- **Why it multiplies:** each pulse is too weak to fire a neuron reliably. Two
  pulses close together add up and often cross the firing threshold. So a neuron
  fires mostly when both bits are 1. For on/off inputs that is multiplication [7].
- **Answer:** the key whose electrodes fire most is the neurons' pick.
- **Fairness check:** every key has exactly four 1-bits, so counting the pulses
  we send per key cannot find the match. Only the neurons can.

| Picks the best key (seed 42, chance is 25%) | 1 try | 10 tries | 40 tries |
|---|---|---|---|
| Neurons: most spikes | 29.5% | 42.0% | 60.0% |
| Computer only: most pulses | 22.7% | 26.0% | 26.7% |
| Exact maths | 100% | 100% | 100% |

Firing chance in the 2 ms after the query pulse: 27% with both bits on, 14% with
the query bit only, 2% with the key bit only, and under 1% with neither.

```bash
CL_SDK_ACCELERATED_TIME=1 uv run python experiments/attention_reservoir.py --calibrate --no-hold
CL_SDK_ACCELERATED_TIME=1 uv run python experiments/attention_reservoir.py --trials 600 --repeats 1 --no-hold
CL_SDK_ACCELERATED_TIME=1 uv run python experiments/attention_reservoir.py --trials 120 --repeats 40 --no-hold
uv run python experiments/attention_reservoir.py --realtime-demo --repeats 40    # watch in the viewer
```

**What it means:**

- The matching does come from the neurons; the computer alone stays at chance.
- One try is barely better than a guess (about 2 standard errors above chance).
  The simulator's background activity moves each neuron about half way to its
  firing threshold, and we left that alone.
- A plain computer check per electrode gets this right every time, in one go.
  The cells are verifiably doing some of the work, not doing it better.
- Only one random seed was tested. Coupling between neurons (4.0) made it worse.

## Next steps

1. **Ask the simulator's authors** whether learning is meant to work through the
   SDK. If it is, Plan A runs as written.
2. **Run Plan A on a real CL1.** It only uses pulses. Cultured neurons on chips
   have learned from structured stimulation before [8].
3. **Drop our "teacher" pulse.** Instead of pulsing the value ourselves, keep
   pulsing the key and stop when the right value fires. Networks with the timing
   rule learn whatever stops the stimulation [9].
4. **Join the plans.** Plan B finds the best key, Plan A recalls its value:
   match, weigh and mix, all in neurons.
5. **Make Plan B less noisy within the model.** Use several electrodes per bit
   and average them, test more seeds, and compare against a trained readout on
   the raw pulses.

## References

1. Vaswani, A. et al. (2017). Attention Is All You Need. arXiv:1706.03762. https://arxiv.org/abs/1706.03762
2. Bycroft, B. LLM Visualization. https://bbycroft.net/llm
3. Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience* 18(24), 10464–10472. https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998
4. Ramsauer, H. et al. (2020). Hopfield Networks is All You Need. arXiv:2008.02217. https://arxiv.org/abs/2008.02217
5. Maass, W., Natschläger, T. & Markram, H. (2002). Real-time computing without stable states. *Neural Computation* 14(11), 2531–2560. https://doi.org/10.1162/089976602760407955
6. Cai, H. et al. (2023). Brain organoid reservoir computing for artificial intelligence. *Nature Electronics* 6, 1032–1039. https://doi.org/10.1038/s41928-023-01069-w
7. London, M. & Häusser, M. (2005). Dendritic computation. *Annual Review of Neuroscience* 28, 503–532. https://doi.org/10.1146/annurev.neuro.28.061604.135703
8. Kagan, B. J. et al. (2022). In vitro neurons learn and exhibit sentience when embodied in a simulated game-world. *Neuron* 110(23), 3952–3969. https://doi.org/10.1016/j.neuron.2022.09.001
9. Sinapayen, L., Masumori, A. & Ikegami, T. (2017). Learning by stimulation avoidance: a principle to control spiking neural networks dynamics. *PLoS ONE* 12(2), e0170388. https://doi.org/10.1371/journal.pone.0170388
