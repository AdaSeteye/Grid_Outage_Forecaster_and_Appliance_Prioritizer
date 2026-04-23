# Product & business digest — T2.3 Grid forecaster + appliance prioritizer

Audience: small-business owners on **feature phones**, **2G/3G**, and **intermittent mains**. Numbers below are illustrative for the salon archetype in `businesses.json` (7 appliances, **6.5 kW** backup).

## Morning digest — 3 SMS × 160 characters (Kinyarwanda + short French anchor)

**SMS 1 (158 chars)**  
`UBUTUMWA 1/3: Ejo 06-12 risk outage 18% max. Komeza POS + frigo. AC + hair dryer bip 11h-14h. Murakoze — GridDigest`

**SMS 2 (159 chars)**  
`UBUTUMWA 2/3: 14h-20h risk outage 31%. Guma POS; koresha amashanyarazi make. Niba internet irapfuye, koresha plan y’icyumweru (LED 2=red).`

**SMS 3 (156 chars)**  
`UBUTUMWA 3/3: Digicel/Airtel ~12 KB/refresh. RWF ~45k/h saved vs full load if outage hits peak. Reply STOP. (Fr: risque coupure, suivez plan salon.)`

*(Character counts exclude the label line; body text is kept ≤160 per segment.)*

## Internet drops at 13:00 — what the owner sees

- **On device (lite UI / cached HTML):** last successful **24h curve + table** stays on screen; a **red banner** shows `Offline · plan age 3h` with the **UTC timestamp** of the last refresh.
- **Risk budget (staleness):** if **age > 4 h** *or* **calendar day rolls** (local midnight), the UI shows **`Plan stale — shed luxury manually`** and we **stop auto-relying** on hour-specific ON/OFF for **comfort/luxury**; **critical** defaults to **ON** until a new 12 KB fetch succeeds.
- **Maximum staleness before “do not trust hour-level plan”:** **4 hours** wall-clock (configurable to 2h during storm season). After that, only **tier rules** (luxury off, comfort discretionary) remain trustworthy without a refresh.

## Customer who cannot read — chosen mode: **color + icon strip (hardware)**

**Why not voice alone:** salons are noisy; voice prompts are easy to miss and costly to localize.  
**Chosen approach:** a **4-LED strip + 3 large icons** on a wall-mounted box (or the `lite_ui.html` high-contrast view for smartphones when available):

| LED pattern | Meaning |
|-------------|---------|
| **Green solid** | Safe window — normal operations |
| **Yellow blink** | Shed **luxury** first (speaker, hair bank) |
| **Orange blink** | Shed **comfort** next (AC, geyser) |
| **Red solid + slow beep** | **Critical only** (POS, fridge, lights minimum) |

Icons (etched + colored): **plug**, **snowflake**, **scissors** map to shedding tiers for this archetype. A community health worker trains once (**15 minutes**) using the physical card shipped with the relay board.

## Neighbor signal (stretch — optional design)

Crowd SMS “1” to a short code on outage → **Bayesian bump** on `P(outage)` for that cell (+0.05 capped) for the next **6 h**, decaying half-life **90 min**. Keeps model honest: crowd signal is a **prior nudge**, not a replacement for `grid_history.csv`.
