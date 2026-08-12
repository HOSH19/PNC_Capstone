<!-- prompt_version: v4 -->
You are given the title of a news item. It may or may not be about a bank.
Your job is to judge, in one word, what the item implies about A BANK'S OWN
RISK — not the article's emotional tone, and not any other company's fortunes.

FIRST, decide whether a bank is the SUBJECT of the item.

A bank is NOT the subject when it merely appears as:
  - the analyst, broker, or research house rating someone else
  - the investor, buyer, seller, or holder of someone else's shares
  - the lender, underwriter, or adviser on someone else's deal
  - a source of commentary or market forecasts
  - background in a story about a non-bank company, an index, or the economy

A bank IS the subject when the action is done TO it, or when what moves is
its own business:
  - it is the company being rated, upgraded, or downgraded by someone else
  - it is being sued, investigated, fined, or placed under a regulatory order
  - its own results, deposits, capital, ratings, or share price move
  - several banks are named together as the ones rising or falling — each of
    them is a subject, and a bank of any country counts

If no bank is the subject, the answer is neutral, however dramatic the news.

ONLY IF a bank is the subject, judge the direction of that bank's risk:

- negative: the bank's own risk is RISING / its health worsening. Losses,
  deposit outflows, enforcement actions against it — including entering a
  formal or written agreement, consent order, or cease-and-desist with a
  regulator (OCC, Fed, FDIC) — lawsuits or investigations targeting it,
  risk/finance executive departures, credit deterioration, its own rating cut,
  a sharp fall in its own share price, and the euphemisms for distress:
  "exploring strategic alternatives", "reviewing options", "engaged advisers".
  The bank must be the party affected. Judge the risk, not the headline
  arithmetic: results that beat estimates while the shares drop sharply are
  negative.
- positive: the bank's own risk is FALLING / its health improving. Capital
  raises, consent orders lifted, earnings or margins improving, its own
  credit rating upgraded, litigation against it resolved in its favour.
- neutral: no clear direction for a bank's own risk. This includes routine
  announcements, product launches, branch openings, sponsorships, and:
  - any item where a bank is the analyst, holder, lender, or commentator
  - stock-market plumbing: short-interest changes, option volumes, fund
    flows, index moves
  - a bank executive's views on markets or the economy
  - crime, property, or local-news stories involving a bank branch or
    building rather than the institution's finances
  - an SEC filing whose text states no deterioration or improvement

The same sentence shape can point either way. What decides it is which side
of the action the bank is on:

Article: "Granite Bank cuts its price target on Apex Industrials to $40"
Label: neutral
Article: "Oppenheim Securities cuts its price target on Granite Bank to $40"
Label: negative
Article: "Apex Industrials cut to Underweight at Granite Bank"
Label: neutral
Article: "Granite Bank cut to Underweight at Oppenheim Securities"
Label: negative
Article: "Granite Bank raises its rating on Apex Industrials to Buy"
Label: neutral
Article: "Granite Bank upgraded to Buy at Oppenheim Securities"
Label: positive

More examples:
Article: "Regional Bank reports third straight quarter of deposit outflows"
Label: negative
Article: "Community Bancorp says it is exploring strategic alternatives"
Label: negative
Article: "Pinnacle Bank's chief risk officer resigns after two years"
Label: negative
Article: "Coastal Trust enters a formal written agreement with the OCC over its BSA/AML program"
Label: negative
Article: "GRB Stock Alert: Rosen Law is investigating whether Granite Bank's board breached its duties"
Label: negative
Article: "Lakeshore Bancorp Q1 profit beats estimates, but the stock crashes 15%"
Label: negative
Article: "Bank stocks slide: Granite Bank, Harbor Bancorp and Summit Bank all decline"
Label: negative
Article: "Federal Reserve lifts consent order against Midwest Bank"
Label: positive
Article: "Summit Bank raises $500M in capital, lifting its Tier 1 ratio"
Label: positive
Article: "Lakeshore Bancorp beats estimates as quarterly profit rises 12%"
Label: positive
Article: "Granite Bank increases its holdings in Apex Industrials"
Label: neutral
Article: "Apex Industrials stock sinks 22% after a rare revenue warning"
Label: neutral
Article: "Granite Bank sees significant decline in short interest"
Label: neutral
Article: "Investors buy large volume of Granite Bank put options"
Label: neutral
Article: "Granite Bank's CEO says markets are underestimating risks"
Label: neutral
Article: "Man arrested over armed robbery at a Granite Bank branch"
Label: neutral
Article: "Coastal Bank opens three new branches in the metro area"
Label: neutral

Now label this article. Answer with exactly one word: positive, negative,
or neutral.

Article: "{{ARTICLE}}"
Label:
