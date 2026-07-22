# Labeling Guide — Human Verification (for teammates)

> Everything you need is in this one page. No code required.

## What is this? (30 seconds)

You read a few dozen bank news/filing articles and pick one of
**positive / negative / neutral** for each. This becomes the **human
answer key** we use to check whether the AI (Llama) labels are trustworthy.
Just fill in the `label` column in Excel / Google Sheets.

The articles are two kinds: **news (gdelt)** and **company filings (edgar)**.

## Three absolute rules

1. **Judge by the bank's RISK DIRECTION, not by tone.** (definitions below)
2. **Judge from the article text only.** Do NOT look up or guess what happened
   to the bank later.
3. **If you're genuinely unsure → `neutral`.** Don't guess a direction.
   (A one-line note in `comment` about why is a bonus.)

## The three classes (risk direction)

- **negative** — the bank's risk is **rising** / health worsening.
  Losses, deposit outflows, enforcement actions, fines, lawsuits,
  risk/finance executive exits, and **euphemistic distress signals** like
  "exploring strategic alternatives".
- **positive** — the bank is getting **stronger**.
  Capital raises, consent orders **lifted**, earnings improvement, rating
  upgrades.
- **neutral** — no clear risk direction.
  Routine announcements, product launches, branch openings, sponsorships,
  or the bank only mentioned in passing.

## Easy-to-miss traps (read this — it's where the skill is)

Cases where **tone and risk disagree**:

| Article | Answer | Why |
|---|---|---|
| "exploring strategic alternatives" | **negative** | calm tone, but a sale/distress signal |
| "regulator lifts consent order" | **positive** | says "regulator" but risk goes **down** |
| "dividend held steady (unchanged)" | **neutral** | not bad news. A *cut* would be negative |
| "bank named marathon sponsor" | **neutral** | unrelated to risk |
| "XYZ Capital buys 1,200 shares of the bank" | **neutral** | someone else buying the stock ≠ a risk signal |

## How to fill it in

1. Open the CSV / sheet you were given. Columns:
   `id, source, title, text_excerpt, label, comment`.
2. Read each row's `title`. If `source` is `edgar`, also read `text_excerpt`
   (the filing excerpt).
3. In `label`, write exactly one of **`positive` / `negative` / `neutral`**
   (lowercase).
4. If you hesitated, add a one-line note in `comment` (optional).
5. **Fill every row.** No blanks.
6. Save and send it back to Jiwon.

## Filled example (do it like this)

| id | source | title | label | comment |
|---|---|---|---|---|
| 1001 | gdelt | Regional Bank reports third straight quarter of deposit outflows | negative | deposits leaving |
| 1005 | gdelt | Federal Reserve lifts consent order against Midwest Bank | positive | order lifted = good news |
| 1017 | edgar | ...dividend of $0.20, unchanged from prior quarter | neutral | held steady, no direction |

## Do NOT

- Write code ❌ / put anything other than pos·neg·neu in `label` ❌
- Search the internet for "what happened to this bank" ❌ (rule 2)
- Leave a row blank because it's hard ❌ → use `neutral` + a `comment`

Stuck? Ask Jiwon.
