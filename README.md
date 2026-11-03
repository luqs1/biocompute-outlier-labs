# biocompute-outlier-labs

Closed-loop experiments written against the
[Cortical Labs `cl` SDK](https://github.com/Cortical-Labs/cl-sdk). On a laptop
the SDK runs a simulator instead of a CL1. This repo replaces the simulator's
default random spike generator with a leaky integrate-and-fire (LIF) network
that responds to stimulation (`src/organoid_simulator`). The same experiment
code later runs unchanged on a CL1.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs `cl-sdk==1.0.0` and this repo as an editable package, which puts
`organoid_simulator` on the path for the simulator subprocess.

## Run the example experiment

```bash
uv run python experiments/example1.py
```

The script:

1. Selects the LIF simulator with `coupling_gain_mV=4.0`, so evoked activity spreads to neighbouring channels.
2. Opens the SDK's web viewer in your browser.
3. Runs a 20 Hz loop for 30 s and stimulates channel 27 on every other tick.
4. Prints spike counts on channel 27 for ticks after a stim and ticks after no stim, then the most active channels.
5. Keeps the viewer open until you press Enter.

Expected output (seed 42, 5 s run):

```
Results
     stim:    55 spikes on ch27 over 50 ticks (1.10 per tick)
  no_stim:    12 spikes on ch27 over 50 ticks (0.24 per tick)
```

Options:

| Flag | Default | Effect |
|---|---|---|
| `--seconds` | `30` | Experiment duration |
| `--channel` | `27` | Channel to stimulate. 0, 4, 7, 56 and 63 never spike. |
| `--coupling` | `4.0` | LIF `coupling_gain_mV`. At `0`, only the stimulated channel responds. |
| `--seed` | `42` | Simulator random seed |
| `--no-browser` | off | Print the viewer URL without opening a browser |
| `--no-hold` | off | Exit when the experiment ends |

## The viewer

When `cl.open()` starts in the simulator, the SDK launches a local web viewer
and prints its address:

```
Data visualiser: http://127.0.0.1:8000/visualiser
```

The port increments if 8000 is in use. The viewer shows live raw traces, spikes
and stims across the 8x8 electrode array. It exists only while the script runs,
which is why the example waits for Enter before exiting.

The viewer is disabled when:

- `CL_SDK_ACCELERATED_TIME=1`. Accelerated time runs faster than real time, which the viewer does not support.
- `CL_SDK_VISUALISATION=0`.

Use accelerated time for batch runs and parameter sweeps, and real time when you
want to watch:

```bash
CL_SDK_ACCELERATED_TIME=1 uv run python experiments/example1.py --no-hold
```

## Writing your own experiment

Every experiment follows the same pattern: open the neurons, run a fixed-rate
loop, read spikes from each tick, and stimulate in response.

```python
import cl

with cl.open() as neurons:
    for tick in neurons.loop(ticks_per_second=20, stop_after_seconds=10):
        for spike in tick.analysis.spikes:        # spikes detected during this tick
            print(spike.timestamp, spike.channel)

        neurons.stim(cl.ChannelSet(27), cl.StimDesign(200, -1.5, 200, 1.5))
```

Key pieces of the SDK:

| API | Purpose |
|---|---|
| `cl.open()` | Context manager that returns a `Neurons` handle. In the simulator it also starts the producer and the viewer. |
| `neurons.loop(ticks_per_second, stop_after_seconds=None, stop_after_ticks=None)` | Fixed-rate closed loop. Each `tick` has `iteration`, `analysis.spikes`, `analysis.stims` and raw `frames`. |
| `neurons.stim(channels, design)` | Queue a stimulation pulse on one or more channels. |
| `cl.ChannelSet(*channels)` | Channels to stimulate |
| `cl.StimDesign(us, uA, ...)` | Mono-, bi- or triphasic pulse as (duration in µs, current in µA) pairs. Durations are multiples of 20 µs; currents are at most ±3 µA. |
| `neurons.record(...)` | Write an HDF5 recording of spikes, stims and raw samples |
| `cl.is_simulator()` | True on a laptop, False on a CL1 |

Data format: 64 channels at 25 kHz, int16 samples (0.195 µV per unit).

## Simulator data source

The repo-root `.env` sets `CL_SDK_DATA_SOURCE` to the LIF source. `cl` loads
`.env` on import, so any script run from the repo root uses the LIF network
without code changes. To change its parameters for one script, override it
before `cl.open()`:

```python
if cl.is_simulator():  # raises RuntimeError on a CL1
    cl.sim.set_simulator_data_source(
        "organoid_simulator.lif_data_source:make_lif_source",
        config={"random_seed": 42, "coupling_gain_mV": 4.0},
    )
```

Or per run from the shell:

```bash
CL_SDK_DATA_SOURCE_CONFIG='{"random_seed": 7, "coupling_gain_mV": 4.0}' \
  uv run python experiments/example1.py
```

See [`src/organoid_simulator/README.md`](src/organoid_simulator/README.md) for
the model, its parameters (coupling, connectivity, plasticity, STDP, spatial
mode) and its limitations.

Simulator results are not evidence about CL1 tissue. Report them separately
from device results.
