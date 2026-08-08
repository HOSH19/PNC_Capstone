# Labeling Guide — Human Verification (for teammates)

> Everything you need is in this one page. No code required.

## What is this? (30 seconds)

You read a few dozen bank news/filing articles and pick one of
**positive / negative / neutral** for each. This becomes the **human
answer key** we use to check whether the AI (Llama) labels are trustworthy.
Just fill in the `label` column in Excel / Google Sheets.

The articles are two kinds: **news (gdelt)** and **company filings (edgar)**.

## Why this matters

The AI (Llama) auto-labels every article, and we then train our real model on
those labels. If the AI's labels are wrong, the model learns the wrong thing —
it can never be better than the labels it trains on. We can't hand-check
thousands of articles, so you label a small sample by hand, **blind to the
AI's answer**. Comparing your labels to the AI's tells us whether the AI is
trustworthy enough to proceed — and, if not, what it's getting wrong. That's
why careful, honest labels matter: a sloppy label here directly corrupts that
check. When unsure, `neutral` + a `comment` beats a guess.

## Four absolute rules

1. **First ask: is a bank the SUBJECT of this article?** If not → `neutral`,
   no matter how dramatic the news. The article has to be *about* a bank (or
   banks), not merely mention one. A non-bank company, a market index, or
   "lenders" as a generic group → `neutral`. Only after a bank passes this
   check do you judge direction. Any country's bank counts — US or not.
2. **Judge by the bank's RISK DIRECTION, not by tone.** (definitions below)
3. **Judge from the article text only.** Do NOT look up or guess what happened
   to the bank later.
4. **If you're genuinely unsure → `neutral`.** Don't guess a direction.
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

Cases where **the bank is in the headline but isn't the subject** (rule 1):

| Article | Answer | Why |
|---|---|---|
| "JPMorgan cuts Redwood Trust's price target" | **neutral** | the bank is the *analyst*; RWT is the subject |
| "S&P 500 rebounds as inflation eases and bank earnings stay positive" | **neutral** | the index is the subject; banks are the explanation |
| "Subprime auto delinquencies hit a 32-year high — what it means for lenders" | **neutral** | "lenders" generically, no bank named |
| "Bank stocks underperform: HDFC, Axis, Yes Bank decline" | **negative** | named banks *are* the subject; non-US still counts |
| "Morgan Stanley sets a new 52-week high" / "Commerce Bancshares downgraded to Sell" | **positive** / **negative** | the bank's *own* share price or rating |

### EDGAR 8-K filings — the one place we disagreed with each other

Rows whose `source` is `edgar` are SEC filings, and the 2026-08-07 gate review
found them to be the least consistent group **on the human side**, in both
directions: some `"Fifth Third Bancorp 8-K"`-style rows were called `positive`
and near-identical ones `neutral`. The rule, so the next pass is consistent:

- A **routine** 8-K — scheduled earnings release, dividend declaration,
  officer appointment, annual-meeting result — with no stated worsening or
  improvement in the bank's condition → **neutral**. An earnings release is
  not `positive` merely for existing.
- Only go directional when the excerpt itself says the condition changed:
  results materially beating or missing, a charge or loss provision, a
  regulatory agreement, an executive departing under pressure.
- Excerpts that are entirely the SEC cover page — address, phone number, the
  Rule 425 / 14a-12 checkboxes, the registered-securities table — contain no
  event at all → **neutral**. There is nothing to read.

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
