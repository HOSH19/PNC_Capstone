<!-- prompt_version: v1 -->
You label news articles about US banks by the RISK DIRECTION they imply for
the bank — not by the article's emotional tone.

- negative: implies the bank's risk is RISING / health worsening (losses,
  deposit outflows, enforcement actions, lawsuits, risk/finance executive
  exits, "exploring strategic alternatives", and other euphemistic distress
  signals).
- positive: implies risk FALLING / health improving (capital raises, consent
  orders lifted, earnings improvement, rating upgrades).
- neutral: no clear risk direction (routine announcements, product launches,
  branch openings, sponsorships, incidental mentions).

Examples:
Article: "Regional Bank reports third straight quarter of deposit outflows"
Label: negative
Article: "Community Bancorp says it is exploring strategic alternatives"
Label: negative
Article: "Pinnacle Bank's chief risk officer resigns after two years"
Label: negative
Article: "Federal Reserve lifts consent order against Midwest Bank"
Label: positive
Article: "Summit Bank raises $500M in capital, lifting its Tier 1 ratio"
Label: positive
Article: "Coastal Bank opens three new branches in the metro area"
Label: neutral

Now label this article. Answer with exactly one word: positive, negative,
or neutral.

Article: "{{ARTICLE}}"
Label:
